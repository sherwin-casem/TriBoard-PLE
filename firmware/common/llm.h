#ifndef LLM_H
#define LLM_H
#include <stdint.h>
#include <math.h>
#include <string.h>

#define LLM_MAGIC 0x504C4532u
#define RMS_EPS 1e-6f

typedef struct {
  int vocab, dim, n_layers, n_heads, ffn, ple_dim, ple_half, seq_len, group;
  float rope_theta;
} Cfg;

typedef struct {
  const uint8_t  *codes;
  const uint16_t *scales;
  int rows, cols, group, n_groups, row_bytes;
  int nbits;  // 4 or 8
} QT;

static inline float half2float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
  if (exp == 0) {
    if (man == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3FF; f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; memcpy(&out, &f, 4); return out;
}

typedef struct {
  Cfg c;
  QT tok_emb;
  QT ple_model_proj_a;
  const float *ple_proj_norm_a;
  const float *out_norm;
} ModelA;

typedef struct {
  Cfg c;
  const float *attn_norm[32];
  QT qkv[32], attn_proj[32];
  const float *ffn_norm[32];
  QT gate[32], up[32], down[32];
  QT ple_gate[32], ple_proj[32];
  const float *ple_norm[32];
  QT ple_model_proj_b;
  const float *ple_proj_norm_b;
  QT ple_table_b;
} ModelB;

typedef struct {
  Cfg c;
  QT ple_table_a;
} ModelC;

typedef struct {
  float *x, *h, *qkv, *att, *g1, *g2, *logits, *scores, *kcache, *vcache, *ple, *tmpP, *trow;
  int klen;
} Scratch;

static const uint8_t *bind_q(const uint8_t *p, QT *t, int rows, int cols) {
  int32_t group; memcpy(&group, p, 4); p += 4;
  t->rows = rows; t->cols = cols; t->group = group;
  t->n_groups = (cols + group - 1) / group;
  t->row_bytes = (cols + 1) / 2;
  t->nbits = 4;
  t->codes = p;  p += (size_t)rows * t->row_bytes;
  t->scales = (const uint16_t *)p;  p += (size_t)rows * t->n_groups * 2;
  return p;
}

static const uint8_t *bind_q8(const uint8_t *p, QT *t, int rows, int cols) {
  int32_t group; memcpy(&group, p, 4); p += 4;
  t->rows = rows; t->cols = cols; t->group = group;
  t->n_groups = (cols + group - 1) / group;
  t->row_bytes = cols;
  t->nbits = 8;
  t->codes = p;  p += (size_t)rows * t->row_bytes;
  t->scales = (const uint16_t *)p;  p += (size_t)rows * t->n_groups * 2;
  return p;
}

static const uint8_t *bind_f(const uint8_t *p, const float **t, int n) {
  *t = (const float *)p;  return p + (size_t)n * sizeof(float);
}

static inline void deq_row(const QT *t, int r, float *out) {
  const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
  const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
  if (t->nbits == 8) {
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group;
      int end = begin + t->group;
      if (end > t->cols) end = t->cols;
      float scale = half2float(sc[gi]);
      for (int j = begin; j < end; j++) {
        out[j] = (float)((int)row[j] - 128) * scale;
      }
    }
    return;
  }
  for (int gi = 0; gi < t->n_groups; gi++) {
    int begin = gi * t->group;
    int end = begin + t->group;
    if (end > t->cols) end = t->cols;
    float scale = half2float(sc[gi]);
    int j = begin;
    if ((j & 1) && j < end) {
      out[j] = (float)((row[j >> 1] >> 4) - 8) * scale;
      j++;
    }
    for (; j + 1 < end; j += 2) {
      uint8_t byte = row[j >> 1];
      out[j] = (float)((byte & 0xF) - 8) * scale;
      out[j + 1] = (float)((byte >> 4) - 8) * scale;
    }
    if (j < end) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      out[j] = (float)(code - 8) * scale;
    }
  }
}

static inline void matvec_q(const QT *t, const float *x, float *y) {
  for (int r = 0; r < t->rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float acc = 0.f;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group;
      int end = begin + t->group;
      if (end > t->cols) end = t->cols;
      float scale = half2float(sc[gi]);
      float group_acc = 0.f;
      int j = begin;
      if ((j & 1) && j < end) {
        group_acc += (float)((row[j >> 1] >> 4) - 8) * x[j];
        j++;
      }
      for (; j + 1 < end; j += 2) {
        uint8_t byte = row[j >> 1];
        group_acc += (float)((byte & 0xF) - 8) * x[j];
        group_acc += (float)((byte >> 4) - 8) * x[j + 1];
      }
      if (j < end) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        group_acc += (float)(code - 8) * x[j];
      }
      acc += group_acc * scale;
    }
    y[r] = acc;
  }
}

static inline void rmsnorm(const float *x, const float *w, int n, float *out) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / n + RMS_EPS);
  for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

static inline float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.f + expf(-x)); }

static int llm_load_a(const uint8_t *base, ModelA *m) {
  const uint8_t *p = base;
  uint32_t magic; memcpy(&magic, p, 4); p += 4;
  if (magic != LLM_MAGIC) return -1;
  int32_t hv[9]; memcpy(&hv, p, 36); p += 36;
  m->c.vocab = hv[0]; m->c.dim = hv[1]; m->c.n_layers = hv[2]; m->c.n_heads = hv[3];
  m->c.ffn = hv[4]; m->c.ple_dim = hv[5]; m->c.ple_half = hv[6];
  m->c.seq_len = hv[7]; m->c.group = hv[8];
  memcpy(&m->c.rope_theta, p, 4); p += 4;
  int D = m->c.dim, L = m->c.n_layers, Ph = m->c.ple_half, V = m->c.vocab;
  p = bind_q8(p, &m->tok_emb, V, D);
  p = bind_q(p, &m->ple_model_proj_a, L * Ph, D);
  p = bind_f(p, &m->ple_proj_norm_a, Ph);
  p = bind_f(p, &m->out_norm, D);
  return 0;
}

static int llm_load_b(const uint8_t *base, ModelB *m) {
  const uint8_t *p = base;
  uint32_t magic; memcpy(&magic, p, 4); p += 4;
  if (magic != LLM_MAGIC) return -1;
  int32_t hv[9]; memcpy(&hv, p, 36); p += 36;
  m->c.vocab = hv[0]; m->c.dim = hv[1]; m->c.n_layers = hv[2]; m->c.n_heads = hv[3];
  m->c.ffn = hv[4]; m->c.ple_dim = hv[5]; m->c.ple_half = hv[6];
  m->c.seq_len = hv[7]; m->c.group = hv[8];
  memcpy(&m->c.rope_theta, p, 4); p += 4;
  int D = m->c.dim, L = m->c.n_layers, F = m->c.ffn, P = m->c.ple_dim, Ph = m->c.ple_half, V = m->c.vocab;
  for (int i = 0; i < L; i++) {
    p = bind_f(p, &m->attn_norm[i], D);
    p = bind_q(p, &m->qkv[i], 3 * D, D);
    p = bind_q(p, &m->attn_proj[i], D, D);
    p = bind_f(p, &m->ffn_norm[i], D);
    p = bind_q(p, &m->gate[i], F, D);
    p = bind_q(p, &m->up[i], F, D);
    p = bind_q(p, &m->down[i], D, F);
    p = bind_q(p, &m->ple_gate[i], P, D);
    p = bind_q(p, &m->ple_proj[i], D, P);
    p = bind_f(p, &m->ple_norm[i], D);
  }
  p = bind_q(p, &m->ple_model_proj_b, L * Ph, D);
  p = bind_f(p, &m->ple_proj_norm_b, Ph);
  p = bind_q(p, &m->ple_table_b, V, L * Ph);
  return 0;
}

static int llm_load_c(const uint8_t *base, ModelC *m) {
  const uint8_t *p = base;
  uint32_t magic; memcpy(&magic, p, 4); p += 4;
  if (magic != LLM_MAGIC) return -1;
  int32_t hv[9]; memcpy(&hv, p, 36); p += 36;
  m->c.vocab = hv[0]; m->c.dim = hv[1]; m->c.n_layers = hv[2]; m->c.n_heads = hv[3];
  m->c.ffn = hv[4]; m->c.ple_dim = hv[5]; m->c.ple_half = hv[6];
  m->c.seq_len = hv[7]; m->c.group = hv[8];
  memcpy(&m->c.rope_theta, p, 4); p += 4;
  int V = m->c.vocab, L = m->c.n_layers, Ph = m->c.ple_half;
  p = bind_q(p, &m->ple_table_a, V, L * Ph);
  return 0;
}

#endif
