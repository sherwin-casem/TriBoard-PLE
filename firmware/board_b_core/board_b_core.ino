#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include "esp_partition.h"
#include "../common/llm.h"
#include "../common/espnow_protocol.h"

static const uint8_t BOARD_A_PEER[] = BOARD_A_MAC;

#define MAX_SEQ_LEN 256

ModelB model;
Reassembler recv_buf;
Scratch s;

static uint8_t send_seq = 0;
static bool send_done = false;
static float rope_freqs[64];  // precomputed RoPE frequencies (max Dh/2 = 64)

static void espnow_send_cb(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  (void)tx_info;
  send_done = (status == ESP_NOW_SEND_SUCCESS);
}

static bool send_packet(const uint8_t *mac, const uint8_t *data, int len) {
  send_done = false;
  esp_now_send(mac, data, len);
  int wait = 0;
  while (!send_done && wait < 100) { delay(1); wait++; }
  return send_done;
}

static bool send_fragments(const uint8_t *mac, const float *data, int n_floats, uint8_t msg_type) {
  uint8_t pkts[MAX_FRAGS][ESPNOW_MAX_PAYLOAD];
  int sizes[MAX_FRAGS];
  int n = espnow_encode_fragments(data, n_floats, msg_type, send_seq++, pkts, sizes);
  for (int i = 0; i < n; i++) {
    if (!send_packet(mac, pkts[i], sizes[i])) return false;
    delay(2);
  }
  return true;
}

static void compute_ple_b(int token_id, const float *x, float *ple_b_out) {
  int D = model.c.dim, L = model.c.n_layers, Ph = model.c.ple_half;
  float *tmpP = ple_b_out + L * Ph;
  matvec_q(&model.ple_model_proj_b, x, tmpP);
  float dscale = 1.f / sqrtf((float)D);
  for (int i = 0; i < L * Ph; i++) tmpP[i] *= dscale;
  for (int l = 0; l < L; l++)
    rmsnorm(tmpP + l * Ph, model.ple_proj_norm_b, Ph, tmpP + l * Ph);
  float *trow = tmpP + L * Ph;
  deq_row(&model.ple_table_b, token_id, trow);
  float sp = sqrtf((float)Ph), inv2 = 0.70710678f;
  for (int i = 0; i < L * Ph; i++)
    ple_b_out[i] = (tmpP[i] + trow[i] * sp) * inv2;
}

static void process_token(const float *x_in, const float *ple_a, const float *ple_b, float *x_out, int pos) {
  int D = model.c.dim, L = model.c.n_layers, P = model.c.ple_dim;
  int Ph = model.c.ple_half, F = model.c.ffn, H = model.c.n_heads, Dh = D / H;

  float *x = s.x;
  memcpy(x, x_in, D * sizeof(float));

  for (int l = 0; l < L; l++) {
    float *full_ple = s.tmpP;
    memcpy(full_ple, ple_a + l * Ph, Ph * sizeof(float));
    memcpy(full_ple + Ph, ple_b + l * Ph, Ph * sizeof(float));

    rmsnorm(x, model.attn_norm[l], D, s.h);
    matvec_q(&model.qkv[l], s.h, s.qkv);
    float *q = s.qkv, *k = s.qkv + D, *v = s.qkv + 2 * D;

    float rope_c[64], rope_s[64];
    for (int i = 0; i < Dh / 2; i++) {
      float theta = pos * rope_freqs[i];
      rope_c[i] = cosf(theta);
      rope_s[i] = sinf(theta);
    }
    for (int hh = 0; hh < H; hh++) {
      float *qh = q + hh * Dh, *kh = k + hh * Dh;
      for (int i = 0; i < Dh / 2; i++) {
        float c = rope_c[i], sn = rope_s[i];
        float q1 = qh[i], q2 = qh[i + Dh / 2];
        qh[i] = q1 * c - q2 * sn; qh[i + Dh / 2] = q2 * c + q1 * sn;
        float k1 = kh[i], k2 = kh[i + Dh / 2];
        kh[i] = k1 * c - k2 * sn; kh[i + Dh / 2] = k2 * c + k1 * sn;
      }
    }

    int n_pos = pos + 1;
    float *layer_k = s.kcache + ((size_t)l * MAX_SEQ_LEN + pos) * D;
    float *layer_v = s.vcache + ((size_t)l * MAX_SEQ_LEN + pos) * D;
    memcpy(layer_k, k, D * sizeof(float));
    memcpy(layer_v, v, D * sizeof(float));

    memset(s.att, 0, D * sizeof(float));
    for (int hh = 0; hh < H; hh++) {
      float *qh = q + hh * Dh;
      float maxv = -1e20f;
      for (int j = 0; j < n_pos; j++) {
        float *kj = s.kcache + ((size_t)l * MAX_SEQ_LEN + j) * D + hh * Dh;
        float dot = 0;
        for (int d = 0; d < Dh; d++) dot += qh[d] * kj[d];
        s.scores[j] = dot / sqrtf((float)Dh);
        if (s.scores[j] > maxv) maxv = s.scores[j];
      }
      float sum = 0;
      for (int j = 0; j < n_pos; j++) {
        s.scores[j] = expf(s.scores[j] - maxv);
        sum += s.scores[j];
      }
      float inv_sum = 1.f / sum;
      for (int j = 0; j < n_pos; j++) s.scores[j] *= inv_sum;
      for (int j = 0; j < n_pos; j++) {
        float *vj_h = s.vcache + ((size_t)l * MAX_SEQ_LEN + j) * D + hh * Dh;
        float w = s.scores[j];
        for (int d = 0; d < Dh; d++) s.att[hh * Dh + d] += w * vj_h[d];
      }
    }

    matvec_q(&model.attn_proj[l], s.att, s.h);
    for (int i = 0; i < D; i++) x[i] += s.h[i];

    rmsnorm(x, model.ffn_norm[l], D, s.h);
    matvec_q(&model.gate[l], s.h, s.g1);
    matvec_q(&model.up[l], s.h, s.g2);
    for (int i = 0; i < F; i++) s.g1[i] = silu(s.g1[i]) * s.g2[i];
    matvec_q(&model.down[l], s.g1, s.h);
    for (int i = 0; i < D; i++) x[i] += s.h[i];

    matvec_q(&model.ple_gate[l], x, s.g2);
    for (int i = 0; i < P; i++) s.g2[i] = gelu(s.g2[i]) * full_ple[i];
    matvec_q(&model.ple_proj[l], s.g2, s.h);
    rmsnorm(s.h, model.ple_norm[l], D, s.h);
    for (int i = 0; i < D; i++) x[i] += s.h[i];
  }

  memcpy(x_out, x, D * sizeof(float));
}

static volatile bool pending_embed = false;
static int pending_token_id;
static float pending_x_in[128];
static float pending_ple_a[768];

static void espnow_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (reassembler_add(&recv_buf, data, len)) {
    if (recv_buf.msg_type == MSG_SEQ_START) {
      s.klen = 0;
      reassembler_reset(&recv_buf);
    } else if (recv_buf.msg_type == MSG_EMBED_SEND) {
      int D = model.c.dim;
      int L = model.c.n_layers;
      int Ph = model.c.ple_half;
      int expected = (1 + D + L * Ph) * sizeof(float);

      if (recv_buf.total_bytes >= expected && !pending_embed) {
        const float *msg = recv_buf.buffer;
        pending_token_id = (int)msg[0];
        memcpy(pending_x_in, msg + 1, D * sizeof(float));
        memcpy(pending_ple_a, msg + 1 + D, L * Ph * sizeof(float));
        pending_embed = true;
      }
      reassembler_reset(&recv_buf);
    }
  }
}

void *ps(size_t n) {
  void *p = malloc(n);
  if (!p) { while(1) delay(1000); }
  return p;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== Board B: Core ===");

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP("BOARD-B-CORE", NULL, 1, 0, 1);
  Serial.println("WiFi AP+STA mode, channel 1");

  esp_now_init();
  Serial.println("ESP-NOW init");
  esp_now_register_send_cb(espnow_send_cb);
  esp_now_register_recv_cb(espnow_recv_cb);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BOARD_A_PEER, 6);
  peer.channel = 1; peer.encrypt = false; peer.ifidx = WIFI_IF_AP;
  esp_err_t peer_ret = esp_now_add_peer(&peer);
  Serial.printf("add_peer: %d\n", peer_ret);

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("ERROR: no model partition"); while(1) delay(1000); }
  Serial.println("model partition found");
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("ERROR: mmap %d\n", err); while(1) delay(1000); }
  Serial.printf("mmap: %u KB\n", (unsigned)(part->size / 1024));

  if (llm_load_b((const uint8_t *)base, &model)) { Serial.println("ERROR: bad model"); while(1) delay(1000); }
  Cfg *c = &model.c;
  Serial.printf("Model: V=%d D=%d L=%d\n", c->vocab, c->dim, c->n_layers);

  int D = c->dim, F = c->ffn, P = c->ple_dim, Ph = c->ple_half, L = c->n_layers, H = c->n_heads, Dh = D / H;
  s.x = (float *)ps(D * 4);
  s.h = (float *)ps((F > D ? F : D) * 4);
  s.qkv = (float *)ps(3 * D * 4);
  s.att = (float *)ps(D * 4);
  s.g1 = (float *)ps(F * 4);
  s.g2 = (float *)ps((P > F ? P : F) * 4);
  s.tmpP = (float *)ps(P * 4);
  s.ple = NULL; s.trow = NULL;

  s.kcache = (float *)ps((size_t)L * MAX_SEQ_LEN * D * sizeof(float));
  s.vcache = (float *)ps((size_t)L * MAX_SEQ_LEN * D * sizeof(float));
  s.scores = (float *)ps((size_t)MAX_SEQ_LEN * sizeof(float));
  s.klen = 0;
  for (int i = 0; i < Dh / 2; i++)
    rope_freqs[i] = powf(c->rope_theta, -2.f * i / Dh);

  reassembler_reset(&recv_buf);
  Serial.println("READY.\n");
}

static float ple_b_buf[2304];
static float x_out_buf[128];

void loop() {
  if (pending_embed) {
    pending_embed = false;
    int D = model.c.dim;
    int pos = s.klen;
    if (pos < MAX_SEQ_LEN) {
      Serial.printf("process tok %d at pos %d\n", pending_token_id, pos);
      Serial.printf("  in_x[0..4]=%.4f %.4f %.4f %.4f %.4f\n", pending_x_in[0], pending_x_in[1], pending_x_in[2], pending_x_in[3], pending_x_in[4]);
      compute_ple_b(pending_token_id, pending_x_in, ple_b_buf);
      Serial.printf("  ple_b[0..4]=%.4f %.4f %.4f %.4f %.4f\n", ple_b_buf[0], ple_b_buf[1], ple_b_buf[2], ple_b_buf[3], ple_b_buf[4]);
      process_token(pending_x_in, pending_ple_a, ple_b_buf, x_out_buf, pos);
      s.klen = pos + 1;
      Serial.printf("  out_x[0..4]=%.4f %.4f %.4f %.4f %.4f\n", x_out_buf[0], x_out_buf[1], x_out_buf[2], x_out_buf[3], x_out_buf[4]);
      Serial.printf("  -> sending %d floats\n", D);
      send_fragments(BOARD_A_PEER, x_out_buf, D, MSG_CORE_RESULT);
    } else {
      Serial.printf("drop tok %d (seq full)\n", pending_token_id);
    }
  }
  delay(10);
}
