#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include "esp_http_server.h"
#include "esp_partition.h"
#include "../common/llm.h"
#include "../common/espnow_protocol.h"
#include "vocab.h"

static const uint8_t BOARD_A_PEER[] = BOARD_A_MAC;

ModelC model;
Reassembler recv_buf;
static char generated_text[4096];
static int generated_len = 0;
static volatile bool new_token_ready = false;
static volatile bool sequence_done = false;
static int last_token_id = -1;
static int top_ids[40];
static bool top_ids_valid = false;

static uint8_t send_seq = 0;
static bool send_done = false;
static bool ple_request_pending = false;
static int ple_token_id = 0;

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

static void espnow_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (reassembler_add(&recv_buf, data, len)) {
    if (recv_buf.msg_type == MSG_SEQ_END) {
      Serial.println("RX MSG_SEQ_END");
      sequence_done = true;
      reassembler_reset(&recv_buf);
    } else if (recv_buf.msg_type == MSG_SEQ_START) {
      Serial.println("RX MSG_SEQ_START - RESET");
      sequence_done = false;
      generated_len = 0; generated_text[0] = '\0';
      reassembler_reset(&recv_buf);
    } else if (recv_buf.msg_type == MSG_LOGITS_SEND) {
      int n_ids = recv_buf.total_bytes / sizeof(float);
      if (n_ids > 40) n_ids = 40;
      const float *top_f = recv_buf.buffer;
      for (int i = 0; i < n_ids; i++) top_ids[i] = (int)top_f[i];
      top_ids_valid = (n_ids > 0);

      if (top_ids_valid) {
        last_token_id = top_ids[0];
        new_token_ready = true;

        if (last_token_id >= 0 && last_token_id < VOCAB_N &&
            generated_len < (int)sizeof(generated_text) - 32) {
          int start = VOCAB_OFF[last_token_id];
          int end = VOCAB_OFF[last_token_id + 1];
          int tlen = end - start;
          char token_text[64];
          int copy_len = tlen < 63 ? tlen : 63;
          memcpy(token_text, VOCAB_BLOB + start, copy_len);
          token_text[copy_len] = '\0';
          Serial.printf("  tok=%d \"%s\"\n", last_token_id, token_text);

          if (generated_len > 0 && tlen > 0 && isalpha((unsigned char)token_text[0])) {
            char last_c = generated_text[generated_len - 1];
            if (isalpha((unsigned char)last_c) || isdigit((unsigned char)last_c)) {
              if (generated_len + 1 < (int)sizeof(generated_text) - 1) {
                generated_text[generated_len] = ' ';
                generated_len++;
              }
            }
          }
          if (generated_len + tlen < (int)sizeof(generated_text) - 1) {
            memcpy(generated_text + generated_len, VOCAB_BLOB + start, tlen);
            generated_len += tlen;
            generated_text[generated_len] = '\0';
          }
        }
      }

      uint8_t ack_pkt[ESPNOW_MAX_PAYLOAD];
      PacketHeader *ah = (PacketHeader *)ack_pkt;
      ah->msg_type = MSG_TOKEN_READY; ah->frag_index = 0; ah->frag_total = 1;
      ah->seq_num = send_seq++;
      memcpy(ack_pkt + 4, &last_token_id, sizeof(int));
      send_packet(BOARD_A_PEER, ack_pkt, 4 + sizeof(int));
      reassembler_reset(&recv_buf);
    } else if (recv_buf.msg_type == MSG_PLE_REQUEST) {
      ple_token_id = (int)((float *)recv_buf.buffer)[0];
      ple_request_pending = true;
    }
  }
}

static httpd_handle_t server = NULL;

static esp_err_t index_handler(httpd_req_t *req) {
  const char *html = "<!DOCTYPE html><html><head>"
    "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>ESP32 Distributed AI - 56M</title>"
    "<style>"
    "body{font-family:monospace;background:#111;color:#0f0;margin:20px;}"
    "h1{color:#0ff;}textarea{width:100%;height:80px;background:#222;color:#0f0;"
    "border:1px solid #0f0;font-family:monospace;font-size:14px;padding:10px;}"
    "button{background:#0f0;color:#111;border:none;padding:10px 20px;"
    "font-family:monospace;font-size:16px;cursor:pointer;margin:10px 0;}"
    "button:hover{background:#0ff;}"
    "#output{background:#222;border:1px solid #0f0;padding:15px;min-height:200px;"
    "white-space:pre-wrap;font-size:14px;margin-top:10px;}"
    ".status{color:#ff0;margin:5px 0;}"
    ".info{color:#888;font-size:12px;}</style></head><body>"
    "<h1>ESP32 Distributed AI</h1>"
    "<p class='status'>3x ESP32-S3 | 56M PLE TinyLM | WikiText-103</p>"
    "<p class='info'>Board protocol: Split-PLE (128+128 per layer)</p>"
    "<textarea id='prompt' placeholder='Type your prompt here...'>The history of</textarea><br>"
    "<button onclick='generate()'>Generate</button>"
    "<button onclick='clearOutput()'>Clear</button>"
    "<div id='status'></div><div id='output'></div>"
    "<script>"
    "let es = null;"
    "function generate() {"
    "  const p = document.getElementById('prompt').value;"
    "  document.getElementById('status').textContent = 'Generating...';"
    "  document.getElementById('output').textContent = '';"
    "  fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},"
    "    body:JSON.stringify({prompt:p})}).then(r=>r.json()).then(d=>{}).catch(e=>{});"
    "  if (es) es.close();"
    "  es = new EventSource('/stream');"
    "  es.onmessage = function(e) {"
    "    if (e.data === '[DONE]') { es.close(); "
    "      document.getElementById('status').textContent = 'Done!'; return; }"
    "    document.getElementById('output').textContent += e.data;"
    "  };"
    "}"
    "function clearOutput() {"
    "  document.getElementById('output').textContent = '';"
    "  document.getElementById('status').textContent = '';"
    "}"
    "</script></body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, strlen(html));
}

static esp_err_t stream_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/event-stream");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
  httpd_resp_set_hdr(req, "Connection", "keep-alive");
  int last_len = 0;
  for (int i = 0; i < 1200; i++) {
    if (generated_len > last_len) {
      int n = generated_len - last_len;
      if (n > 60) n = 60;
      char chunk[64];
      memcpy(chunk, generated_text + last_len, n);
      chunk[n] = '\0';
      char msg[128];
      snprintf(msg, sizeof(msg), "data: %s\n\n", chunk);
      httpd_resp_send_chunk(req, msg, strlen(msg));
      last_len += n;
    }
    if (sequence_done && last_len >= generated_len) break;
    delay(50);
  }
  Serial.printf("stream done: seq=%d gen_len=%d\n", sequence_done, generated_len);
  httpd_resp_send_chunk(req, "data: [DONE]\n\n", 14);
  httpd_resp_send_chunk(req, NULL, 0);
  return ESP_OK;
}

static esp_err_t generate_handler(httpd_req_t *req) {
  char buf[1024];
  int ret = httpd_req_recv(req, buf, sizeof(buf) - 1);
  if (ret <= 0) return ESP_FAIL;
  buf[ret] = '\0';
  char *p = strstr(buf, "\"prompt\"");
  if (!p) { httpd_resp_sendstr(req, "{\"error\":\"no prompt\"}"); return ESP_OK; }
  p = strchr(p + 8, '"');
  if (!p) { httpd_resp_sendstr(req, "{\"error\":\"bad format\"}"); return ESP_OK; }
  p++;
  char *end = strchr(p, '"');
  if (!end) { httpd_resp_sendstr(req, "{\"error\":\"unterminated\"}"); return ESP_OK; }
  *end = '\0';

  generated_len = 0; generated_text[0] = '\0'; new_token_ready = false; sequence_done = false;

  uint8_t pkt[ESPNOW_MAX_PAYLOAD];
  PacketHeader *h = (PacketHeader *)pkt;
  h->msg_type = MSG_EMBED_REQUEST; h->frag_index = 0; h->frag_total = 1;
  h->seq_num = send_seq++;
  int plen = strlen(p);
  if (plen > ESPNOW_DATA_SIZE) plen = ESPNOW_DATA_SIZE;
  memcpy(pkt + 4, p, plen);
  Serial.printf("Sending MSG_EMBED_REQUEST to A (%d bytes)\n", plen);
  bool ok = send_packet(BOARD_A_PEER, pkt, 4 + plen);
  Serial.printf("send_packet: %s\n", ok ? "OK" : "FAIL");

  httpd_resp_set_type(req, "application/json");
  httpd_resp_sendstr(req, "{\"status\":\"generating\"}");
  return ESP_OK;
}

static esp_err_t status_handler(httpd_req_t *req) {
  char json[256];
  snprintf(json, sizeof(json),
    "{\"text\":\"%s\",\"tokens\":%d,\"ready\":%s}",
    generated_text, generated_len / 8, new_token_ready ? "true" : "false");
  httpd_resp_set_type(req, "application/json");
  httpd_resp_sendstr(req, json);
  return ESP_OK;
}

void start_web_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 8;
  httpd_start(&server, &config);
  static const httpd_uri_t uri_root = { .uri = "/", .method = HTTP_GET, .handler = index_handler };
  static const httpd_uri_t uri_gen  = { .uri = "/generate", .method = HTTP_POST, .handler = generate_handler };
  static const httpd_uri_t uri_str  = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler };
  static const httpd_uri_t uri_stat = { .uri = "/status", .method = HTTP_GET, .handler = status_handler };
  httpd_register_uri_handler(server, &uri_root);
  httpd_register_uri_handler(server, &uri_gen);
  httpd_register_uri_handler(server, &uri_str);
  httpd_register_uri_handler(server, &uri_stat);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== Board C: Decoder ===");

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP("ESP32-DIST-AI", "ai123456", 1, 0, 4);
  Serial.println("softAP started");
  esp_err_t en_ret = esp_now_init();
  Serial.printf("esp_now_init: %d\n", en_ret);

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

  if (llm_load_c((const uint8_t *)base, &model)) { Serial.println("ERROR: bad model"); while(1) delay(1000); }
  Serial.println("model loaded");

  reassembler_reset(&recv_buf);
  start_web_server();
  Serial.println("READY.\n");
}

void loop() {
  if (ple_request_pending) {
    int ple_row_sz = model.c.n_layers * model.c.ple_half;
    float *ple_row = (float *)malloc(ple_row_sz * sizeof(float));
    if (ple_row) {
      if (ple_token_id >= 0 && ple_token_id < model.c.vocab) {
        deq_row(&model.ple_table_a, ple_token_id, ple_row);
        send_fragments(BOARD_A_PEER, ple_row, ple_row_sz, MSG_PLE_RESULT);
      }
      free(ple_row);
    }
    reassembler_reset(&recv_buf);
    ple_request_pending = false;
  }
  delay(10);
}
