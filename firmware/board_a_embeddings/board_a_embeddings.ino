#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include "esp_partition.h"
#include "../common/llm.h"
#include "../common/espnow_protocol.h"
#include "vocab.h"

#define TOP_K 40

static const uint8_t BOARD_B_PEER[] = BOARD_B_MAC;
static const uint8_t BOARD_C_PEER[] = BOARD_C_MAC;

ModelA model;
Reassembler recv_buf;
static uint8_t send_seq = 0;
static bool send_done = false;

static void espnow_send_cb(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  (void)tx_info;
  send_done = (status == ESP_NOW_SEND_SUCCESS);
}

static bool send_packet(const uint8_t *mac, const uint8_t *data, int len) {
  send_done = false;
  esp_err_t ret = esp_now_send(mac, data, len);
  if (ret != ESP_OK) { Serial.printf("TX send failed: %d\n", ret); return false; }
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

static int tokenize_simple(const char *text, int *ids, int max_ids) {
  int n = 0;
  const char *p = text;
  while (*p && n < max_ids) {
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
    int wlen = p - start;
    int found = -1;
    for (int i = 0; i < VOCAB_N; i++) {
      int vlen = VOCAB_OFF[i + 1] - VOCAB_OFF[i];
      if (vlen == wlen && memcmp(VOCAB_BLOB + VOCAB_OFF[i], start, wlen) == 0) {
        found = i; break;
      }
    }
    ids[n++] = found >= 0 ? found : 0;
  }
  return n;
}

static bool request_ple_row(int token, float *ple_out, int timeout_ms) {
  float tmsg[1] = {(float)token};
  if (!send_fragments(BOARD_C_PEER, tmsg, 1, MSG_PLE_REQUEST)) {
    Serial.println("  ERROR: MSG_PLE_REQUEST send failed");
    return false;
  }
  int waited = 0;
  while (waited < timeout_ms) {
    if (recv_buf.ready && recv_buf.msg_type == MSG_PLE_RESULT) {
      memcpy(ple_out, recv_buf.buffer, model.c.n_layers * model.c.ple_half * sizeof(float));
      reassembler_reset(&recv_buf);
      return true;
    }
    delay(1); waited++;
  }
  Serial.println("  ERROR: timeout waiting for MSG_PLE_RESULT");
  return false;
}

static void compute_ple_a(int token, const float *x, float *ple_out) {
  int D = model.c.dim, L = model.c.n_layers, Ph = model.c.ple_half;
  float *trow = ple_out + L * Ph;
  if (!request_ple_row(token, trow, 10000)) {
    memset(ple_out, 0, L * Ph * sizeof(float));
    return;
  }
  float *tmpP = trow + L * Ph;
  matvec_q(&model.ple_model_proj_a, x, tmpP);
  float dscale = 1.f / sqrtf((float)D);
  for (int i = 0; i < L * Ph; i++) tmpP[i] *= dscale;
  for (int l = 0; l < L; l++)
    rmsnorm(tmpP + l * Ph, model.ple_proj_norm_a, Ph, tmpP + l * Ph);
  float sp = sqrtf((float)Ph), inv2 = 0.70710678f;
  for (int i = 0; i < L * Ph; i++)
    ple_out[i] = (tmpP[i] + trow[i] * sp) * inv2;
}

static void compute_head(const float *x, int *top_ids, float *logits_buf) {
  int D = model.c.dim, V = model.c.vocab;
  float *xn = logits_buf + V;
  rmsnorm(x, model.out_norm, D, xn);
  for (int v = 0; v < V; v++) {
    float row[128];
    deq_row(&model.tok_emb, v, row);
    float acc = 0.f;
    for (int i = 0; i < D; i++) acc += row[i] * xn[i];
    logits_buf[v] = acc;
  }
  for (int k = 0; k < TOP_K; k++) {
    int best = 0;
    float bv = -1e30f;
    for (int v = 0; v < V; v++) {
      if (logits_buf[v] > bv) {
        bool used = false;
        for (int j = 0; j < k; j++) { if (top_ids[j] == v) { used = true; break; } }
        if (!used) { bv = logits_buf[v]; best = v; }
      }
    }
    top_ids[k] = best;
  }
}

static bool wait_for_board_b(float *x_out, int timeout_ms) {
  int waited = 0;
  while (waited < timeout_ms) {
    if (recv_buf.ready && recv_buf.msg_type == MSG_CORE_RESULT) {
      if (recv_buf.total_bytes >= model.c.dim * 4) {
        memcpy(x_out, recv_buf.buffer, model.c.dim * sizeof(float));
        reassembler_reset(&recv_buf);
        return true;
      }
    }
    delay(1); waited++;
  }
  return false;
}

static void espnow_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  Serial.printf("RX type=%d len=%d\n", data[0], len);
  reassembler_add(&recv_buf, data, len);
}

static float *logits_buf = NULL;

static int sample_token(float *logits, int V, float temperature) {
  float max_l = logits[0];
  for (int i = 1; i < V; i++) if (logits[i] > max_l) max_l = logits[i];
  float sum = 0;
  for (int i = 0; i < V; i++) { logits[i] = expf((logits[i] - max_l) / temperature); sum += logits[i]; }
  float r = (float)rand() / (float)RAND_MAX;
  float cum = 0;
  for (int i = 0; i < V; i++) { cum += logits[i] / sum; if (r < cum) return i; }
  return V - 1;
}

static void process_espnow_prompt() {
  int V = model.c.vocab, D = model.c.dim, Ph = model.c.ple_half, L = model.c.n_layers;
  char *prompt = (char *)recv_buf.buffer;
  int plen = recv_buf.total_bytes;
  if (plen <= 0 || plen >= 512) return;
  prompt[plen] = '\0';
  Serial.printf("PROMPT (%d bytes): \"%s\"\n", plen, prompt);

  int ids[128];
  int n_tokens = tokenize_simple(prompt, ids, 128);
  Serial.printf("TOKENS (%d): ", n_tokens);
  for (int i = 0; i < n_tokens; i++) Serial.printf("%d ", ids[i]);
  Serial.println();

  uint8_t pkt[ESPNOW_MAX_PAYLOAD];
  PacketHeader *hdr = (PacketHeader *)pkt;
  hdr->msg_type = MSG_SEQ_START; hdr->frag_index = 0; hdr->frag_total = 1;
  hdr->seq_num = send_seq++;
  send_packet(BOARD_B_PEER, pkt, 4);
  send_packet(BOARD_C_PEER, pkt, 4);

  float *x = logits_buf + V;
  float *ple = x + D;
  float *msg = ple + L * Ph;

  int top_ids[TOP_K];
  float top_f[TOP_K];
  for (int t = 0; t < n_tokens; t++) {
    deq_row(&model.tok_emb, ids[t], x);
    compute_ple_a(ids[t], x, ple);

    msg[0] = (float)ids[t];
    memcpy(msg + 1, x, D * sizeof(float));
    memcpy(msg + 1 + D, ple, L * Ph * sizeof(float));
    send_fragments(BOARD_B_PEER, msg, 1 + D + L * Ph, MSG_EMBED_SEND);

    if (!wait_for_board_b(logits_buf, 30000)) {
      Serial.println("  ERROR: timeout from Board B"); break;
    }
    compute_head(logits_buf, top_ids, logits_buf);
  }

  float temp = 0.5f;
  int gen_tok = sample_token(logits_buf, model.c.vocab, temp);
  for (int g = 0; g < 128 && gen_tok > 0 && gen_tok < VOCAB_N; g++) {
    deq_row(&model.tok_emb, gen_tok, x);
    compute_ple_a(gen_tok, x, ple);

    msg[0] = (float)gen_tok;
    memcpy(msg + 1, x, D * sizeof(float));
    memcpy(msg + 1 + D, ple, L * Ph * sizeof(float));
    send_fragments(BOARD_B_PEER, msg, 1 + D + L * Ph, MSG_EMBED_SEND);

    if (!wait_for_board_b(logits_buf, 30000)) {
      Serial.println("  ERROR: gen timeout from Board B"); break;
    }
    compute_head(logits_buf, top_ids, logits_buf);

    float single = (float)sample_token(logits_buf, model.c.vocab, temp);
    if (gen_tok >= 0 && gen_tok < VOCAB_N) {
      int s = VOCAB_OFF[gen_tok];
      int e = VOCAB_OFF[gen_tok + 1];
      int tl = (e - s) < 63 ? (e - s) : 63;
      char tbuf[64]; memcpy(tbuf, VOCAB_BLOB + s, tl); tbuf[tl] = '\0';
      Serial.printf("  gen_%d=%d '%s' -> C\n", g, gen_tok, tbuf);
    } else {
      Serial.printf("  gen_%d=%d -> C\n", g, gen_tok);
    }
    send_fragments(BOARD_C_PEER, &single, 1, MSG_LOGITS_SEND);

    int waited = 0;
    gen_tok = -1;
    while (waited < 30000) {
      if (recv_buf.ready && recv_buf.msg_type == MSG_TOKEN_READY) {
        memcpy(&gen_tok, recv_buf.buffer, sizeof(int));
        reassembler_reset(&recv_buf);
        break;
      }
      delay(1); waited++;
    }
    if (gen_tok < 0) { Serial.println("  ERROR: gen timeout from Board C"); break; }
  }

  hdr->msg_type = MSG_SEQ_END;
  send_packet(BOARD_B_PEER, pkt, 4);
  send_packet(BOARD_C_PEER, pkt, 4);
  Serial.println("DONE.");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== Board A: Embeddings + Head ===");

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP("BOARD-A-LIVE", NULL, 1, 0, 1);
  Serial.println("WiFi AP+STA mode, channel 1");

  esp_now_init();
  esp_now_register_send_cb(espnow_send_cb);
  esp_now_register_recv_cb(espnow_recv_cb);

  for (const uint8_t *mac : {BOARD_B_PEER, BOARD_C_PEER}) {
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, mac, 6);
    peer.channel = 1; peer.encrypt = false; peer.ifidx = WIFI_IF_AP;
    esp_now_add_peer(&peer);
  }
  Serial.println("ESP-NOW peers added");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("ERROR: model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("ERROR: mmap failed %d\n", err); return; }
  Serial.printf("Model partition: %u KB\n", (unsigned)(part->size / 1024));

  if (llm_load_a((const uint8_t *)base, &model)) { Serial.println("ERROR: bad model"); return; }
  Cfg *c = &model.c;
  Serial.printf("Model: V=%d D=%d L=%d Ph=%d\n", c->vocab, c->dim, c->n_layers, c->ple_half);

  logits_buf = (float *)malloc((c->vocab + c->dim + 3 * c->n_layers * c->ple_half) * sizeof(float));
  if (!logits_buf) { Serial.println("ERROR: logits_buf alloc failed"); return; }
  srand(esp_random());
  reassembler_reset(&recv_buf);
  Serial.println("READY.\n");
}

void loop() {
  if (recv_buf.ready && recv_buf.msg_type == MSG_EMBED_REQUEST) {
    process_espnow_prompt();
    reassembler_reset(&recv_buf);
  }
  delay(10);
}
