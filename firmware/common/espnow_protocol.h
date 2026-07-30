// ESP-NOW communication protocol for distributed AI inference.
// Defines packet types, fragmentation, and the message format.
#ifndef ESPNOW_PROTOCOL_H
#define ESPNOW_PROTOCOL_H

#include <stdint.h>
#include <string.h>

// --- Board addresses ---
// Measured from each board's hardware.
#define BOARD_A_MAC {0x14, 0xc1, 0x9f, 0x2a, 0xac, 0xc8}
#define BOARD_B_MAC {0x14, 0xc1, 0x9f, 0x2c, 0x91, 0x10}
#define BOARD_C_MAC {0x28, 0x84, 0x85, 0x51, 0xdc, 0x10}

// --- Message types ---
#define MSG_EMBED_REQUEST    0x01  // A -> A: tokenize prompt, begin inference
#define MSG_EMBED_SEND       0x02  // A -> B: token embedding x[D] + ple[L*P]
#define MSG_CORE_RESULT     0x03  // B -> A: hidden state x[D] after transformer
#define MSG_LOGITS_SEND      0x04  // A -> C: top-K logits or token id
#define MSG_TOKEN_READY      0x05  // C -> A: done decoding, ready for next
#define MSG_TEXT_CHUNK       0x06  // C -> WiFi: text chunk for web output
#define MSG_PLE_REQUEST      0x30  // A -> C: request PLE table row for token
#define MSG_PLE_RESULT       0x31  // C -> A: PLE table row response
#define MSG_SEQ_START        0x10  // A -> B,C: new sequence starting
#define MSG_SEQ_END          0x11  // A -> B,C: sequence finished
#define MSG_ACK              0x20  // any -> any: acknowledge receipt
#define MSG_HEARTBEAT        0xFF  // any -> all: alive ping

// --- Packet structure ---
// ESP-NOW max payload: 250 bytes
// Header: 4 bytes | Payload: up to 246 bytes

#define ESPNOW_MAX_PAYLOAD  250
#define ESPNOW_HEADER_SIZE    4
#define ESPNOW_DATA_SIZE    (ESPNOW_MAX_PAYLOAD - ESPNOW_HEADER_SIZE)

// Fragment header (packed into first 4 bytes of each packet)
typedef struct __attribute__((packed)) {
  uint8_t  msg_type;      // MSG_* constant
  uint8_t  frag_index;    // fragment number (0-based)
  uint8_t  frag_total;    // total fragments for this message
  uint8_t  seq_num;       // sequence number (incremented per message)
} PacketHeader;

// Static assert: header is 4 bytes
_Static_assert(sizeof(PacketHeader) == 4, "PacketHeader must be 4 bytes");

// --- Fragmentation helpers ---

// Max bytes per fragmented message = 250 * 255 = ~63KB (enough for D=96 + L*P=768)
#define MAX_FRAGS  64

// Encode a float buffer into ESP-NOW packets (returns number of packets)
static inline int espnow_encode_fragments(
    const float *data, int n_floats,
    uint8_t msg_type, uint8_t seq_num,
    uint8_t out_pkts[][ESPNOW_MAX_PAYLOAD], int *out_sizes
) {
  int total_bytes = n_floats * sizeof(float);
  int n_frags = (total_bytes + ESPNOW_DATA_SIZE - 1) / ESPNOW_DATA_SIZE;
  if (n_frags > MAX_FRAGS) n_frags = MAX_FRAGS;

  const uint8_t *src = (const uint8_t *)data;
  for (int i = 0; i < n_frags; i++) {
    PacketHeader *h = (PacketHeader *)out_pkts[i];
    h->msg_type = msg_type;
    h->frag_index = i;
    h->frag_total = n_frags;
    h->seq_num = seq_num;

    int offset = i * ESPNOW_DATA_SIZE;
    int chunk = total_bytes - offset;
    if (chunk > ESPNOW_DATA_SIZE) chunk = ESPNOW_DATA_SIZE;
    memcpy(out_pkts[i] + ESPNOW_HEADER_SIZE, src + offset, chunk);
    out_sizes[i] = ESPNOW_HEADER_SIZE + chunk;
  }
  return n_frags;
}

// Reassemble fragments into a float buffer (call with each received packet)
// Returns 1 when all fragments collected, 0 otherwise.
typedef struct {
  float buffer[768 * 32];  // enough for largest message
  uint8_t received[MAX_FRAGS];
  uint8_t total_frags;
  uint8_t msg_type;
  uint8_t seq_num;
  int total_bytes;
  int ready;
} Reassembler;

static inline void reassembler_reset(Reassembler *r) {
  memset(r, 0, sizeof(Reassembler));
}

static inline int reassembler_add(Reassembler *r, const uint8_t *pkt, int pkt_len) {
  if (pkt_len < ESPNOW_HEADER_SIZE) return 0;
  const PacketHeader *h = (const PacketHeader *)pkt;

  // New message?
  if (r->total_frags == 0 || h->seq_num != r->seq_num) {
    reassembler_reset(r);
    r->msg_type = h->msg_type;
    r->seq_num = h->seq_num;
    r->total_frags = h->frag_total;
  }

  int idx = h->frag_index;
  if (idx >= r->total_frags) return 0;
  if (r->received[idx]) return 0;  // duplicate

  r->received[idx] = 1;
  int offset = idx * ESPNOW_DATA_SIZE;
  int chunk = pkt_len - ESPNOW_HEADER_SIZE;
  memcpy((uint8_t *)r->buffer + offset, pkt + ESPNOW_HEADER_SIZE, chunk);

  // Check if all received
  int all = 1;
  for (int i = 0; i < r->total_frags; i++) {
    if (!r->received[i]) { all = 0; break; }
  }
  if (all) {
    r->total_bytes = r->total_frags > 0 ?
      ((r->total_frags - 1) * ESPNOW_DATA_SIZE +
       (pkt_len - ESPNOW_HEADER_SIZE)) : 0;
    r->ready = 1;
    return 1;
  }
  return 0;
}

#endif
