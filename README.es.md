# ESP32-S3 IA Distribuida

**Inferencia distribuida de micro-LLM en tres placas ESP32-S3 N16R8 con comunicación ESP-NOW.**

---

## Descripción General

Este proyecto implementa un **sistema de IA distribuida** que ejecuta un **modelo de lenguaje de 56 millones de parámetros** en tres microcontroladores ESP32-S3. Inspirado por [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai), que demostró correr TinyStories en una sola placa, este proyecto extiende la arquitectura a un sistema multi-placa con interacción vía web.

El modelo se entrena en **WikiText-103** (corpus de Wikipedia) usando **Per-Layer Embeddings (PLE)** de la arquitectura Gemma de Google, se cuantiza a 4 bits y se divide entre tres placas que se comunican mediante el protocolo inalámbrico ESP-NOW. La tabla PLE de 50.3M parámetros se divide entre Board A y Board B (**Split-PLE**) para caber en los 16MB de flash por placa. Board B mantiene una **KV cache** en PSRAM (1.5 MB para 256 posiciones), permitiendo que el transformer atienda toda la secuencia generada en lugar de operar token por token.

![Tres placas ESP32-S3 N16R8 conectadas a energía](docs/images/boards.jpeg)

---

## Arquitectura

```
┌───────────────────┐      ESP-NOW      ┌───────────────────┐      ESP-NOW     ┌───────────────────┐
│    Board A        │ ◄──────────────► │    Board B        │ ◄──────────────► │    Board C        │
│   Embeddings +    │                  │   Core + KV Cache │                  │   PLE_A + Decoder │
│   PLE_Proj + Head │                  │                   │                  │   + WiFi          │
│                   │                  │                   │                  │                   │
│ • Tokenizer BPE   │                  │ • 6 capas Attn    │                  │ • Tabla PLE A     │
│ • tok_emb 8-bit   │                  │ • 6 capas FFN     │                  │   (12.5 MB)       │
│ • ple_model_proj  │                  │ • KV Cache (256)  │                  │ • Sampling        │
│ • out_norm + head │                  │ • PLE_B (mitad)   │                  │ • Servidor Web    │
│                   │                  │ • PLE Gating      │                  │ • Heurística      │
│ Flash: 4.31 MB    │                  │ Flash: 13.71 MB   │                  │ Flash: 13.37 MB   │
└───────────────────┘                  └───────────────────┘                  └───────────────────┘
```

### Flujo de Inferencia (completo)

```
Navegador ──WiFi──► Board C ──ESP-NOW──► Board A ──ESP-NOW──► Board B
                           ◄─────────────── ◄──────────────
```

1. **Usuario** escribe un prompt en el navegador (conectado al WiFi AP de Board C) y hace clic en "Generate"
2. **Board C** recibe el prompt vía HTTP POST → lo reenvía a Board A por ESP-NOW (`MSG_EMBED_REQUEST`)
3. **Board A** tokeniza el texto → por cada token:
   a. Busca el embedding `x[D]` (tok_emb 8-bit)
   b. Calcula la proyección PLE local: `tmpP = ple_model_proj_a @ x`, aplica RMS-norm
   c. Solicita la fila de la tabla PLE a **Board C** vía ESP-NOW: envía `token_id` → C busca `ple_table_a[token]` → envía la fila de vuelta
   d. Combina: `ple = (tmpP + trow * sqrt(Ph)) / sqrt(2)` → envía `token_id + x[D] + ple[L×Ph]` a **Board B**
   e. **Board B** calcula PLE_B[L×Ph] de su tabla local → combina PLE_A + PLE_B en PLE completo[L×P] → ejecuta 6 capas transformer (atención + FFN + PLE gating) → envía el estado oculto `x[D]` de vuelta a Board A
   f. **Board A** aplica `out_norm` → calcula el output head: `logits[V] = tok_emb^T · rmsnorm(x)` → envía los top-40 IDs de token a Board C
   g. **Board C** samplea el siguiente token → decodifica vía vocabulario BPE → agrega al texto generado
4. **Board C** envía el texto generado al navegador en tiempo real vía SSE (`/stream`)
5. **Usuario** ve el texto aparecer en la página web

### Split-PLE

La tabla PLE de 50.3M parámetros (vocab × capas × 256) se divide en dos mitades de 128 dimensiones:
- **PLE_A** (Board C, 12.5 MB, 4-bit group=64): primeras 128 dims por capa — se recarga desde Board C en tiempo de ejecución vía ESP-NOW (MSG_PLE_REQUEST / MSG_PLE_RESULT)
- **PLE_B** (Board B, 12.6 MB, 4-bit group=128): últimas 128 dims por capa — local en Board B

Board A mantiene el `ple_model_proj` (~50 KB, 4-bit) y `ple_proj_norm` (~0.5 KB, fp32) para la proyección local. La tabla PLE de 12.5 MB se movió de Board A a Board C para solucionar el desbordamiento de partición en Board A y permitir la actualización de `tok_emb` a 8-bit para mejor calidad de inferencia.

### KV Cache en Board B

Anteriormente, cada token en el bucle autorregresivo pasaba por el transformer de Board B con `seq_len=1` — la atención solo veía el token actual, no el histórico. Esto perjudicaba la coherencia porque el modelo fue diseñado para `seq_len=128` de contexto.

Board B ahora mantiene una **KV cache** en PSRAM (1.5 MB para 256 posiciones):

- Para cada nuevo token, Q se calcula normalmente, mientras que K y V se agregan a los cachés por capa
- Los scores de atención se calculan contra **todas las posiciones almacenadas** (no solo el token actual)
- RoPE usa el índice de posición real en lugar de `pos=0` — por primera vez el transformer ve atención consciente de la posición
- El caché se reinicia en `MSG_SEQ_START` (nuevo prompt) y tiene un límite de `seq_len=256`
- No hay cambios en A, C ni en el protocolo inalámbrico

Esta es la mejora de calidad más importante: el transformer finalmente funciona como fue diseñado, atendiendo a toda la secuencia generada en lugar de operar token por token de forma aislada.

### Direcciones MAC de las Boards

| Board | Dirección MAC | Puerto Serial |
|-------|---------------|---------------|
| A (Embeddings + PLE_Proj + Head) | `14:c1:9f:2a:ac:c8` | `/dev/cu.usbmodem5C372059631` |
| B (Core + KV Cache) | `14:c1:9f:2c:91:10` | `/dev/cu.usbmodem5C372065471` |
| C (PLE_A + Decoder + WiFi AP) | `28:84:85:51:dc:10` | `/dev/cu.usbmodem5C4D0363671` |

### Protocolo de Comunicación

- **ESP-NOW** inalámbrico peer-to-peer (sin router)
- Paquetes ≤ 250 bytes con fragmentación automática
- Protocolo custom con números de secuencia y acuses de recibo
- Latencia estimada: 1–5ms por round-trip

---

## Estructura del Proyecto

```
esp32s3/
├── README.md                       # Este archivo (inglés)
├── README.es.md                    # Versión en español
├── pyproject.toml                  # Dependencias Python
│
├── src/                            # Pipeline de entrenamiento (PC/Mac)
│   ├── model.py                    # Arquitectura PLE TinyLM (PyTorch)
│   ├── dataset.py                  # Carga de datos TinyStories/WikiText-103
│   ├── train.py                    # Loop de entrenamiento
│   ├── quantize.py                 # Cuantización 4-bit group-wise
│   ├── export.py                   # Exportar a 3 binarios por board
│   └── gen_assets.py               # Generar vocab.h para firmware
│
├── firmware/                       # Firmware ESP32-S3 (Arduino IDE)
│   ├── common/
│   │   ├── llm.h                   # Runtime C de inferencia distribuida
│   │   └── espnow_protocol.h       # Protocolo de mensajes ESP-NOW
│   │
│   ├── board_a_embeddings/         # Firmware Board A
│   │   ├── board_a_embeddings.ino  # Sketch principal
│   │   ├── partitions.csv          # Tabla de particiones flash
│   │   └── vocab.h                 # Vocabulario del tokenizador
│   │
│   ├── board_b_core/               # Firmware Board B
│   │   ├── board_b_core.ino        # Sketch principal
│   │   └── partitions.csv
│   │
│   ├── board_c_decoder/            # Firmware Board C
│   │   ├── board_c_decoder.ino     # Sketch principal (WiFi + Web UI)
│   │   ├── partitions.csv
│   │   └── vocab.h                 # Vocabulario del tokenizador
│   │
│   └── model/                      # Binarios del modelo exportado
│       ├── board_a.bin             # 4.31 MB (tok_emb 8-bit + ple_proj + out_norm)
│       ├── board_b.bin             # 13.71 MB (capas transformer + PLE_B)
│       ├── board_c.bin             # 13.37 MB (tabla PLE A, 4-bit group=64)
│       ├── golden.npz              # Logits de referencia para verificación
│       └── golden.txt              # Referencia golden en texto
│
├── tools/                          # Scripts de utilidad
│   ├── flash_all.sh                # Flashear las 3 boards
│   ├── verify_models.py            # Verificar binarios exportados
│   └── setup_env.sh                # Instalar dependencias Python
│
├── data/                           # Dataset
│   ├── tinystories/                # Texto raw de TinyStories
│   ├── wikitext103/                # Texto raw de WikiText-103
│   ├── tokenizer/                  # Tokenizador BPE entrenado
│   ├── train.bin                   # Datos de entrenamiento tokenizados (112M tokens)
│   └── val.bin                     # Datos de validación tokenizados
│
└── runs/                           # Checkpoints de entrenamiento
    ├── ple-wiki60m-s0.pt            # Checkpoint del modelo 56M
    ├── ple-wiki60m-s0.json          # Historial de entrenamiento
    └── train.log                   # Log de salida del entrenamiento
```

---

## Especificaciones del Modelo

| Parámetro | Valor |
|---|---|
| Arquitectura | Transformer decoder-only con Split-PLE |
| Total de Parámetros | 56.0M almacenados |
| Core (denso, SRAM) | 1.5M |
| Tabla PLE (flash, dividida) | 50.3M (25.2M por board) |
| Output Head (compartido) | 4.2M |
| Tamaño del Vocabulario | 32,768 (BPE, WikiText-103) |
| d_model | 128 |
| n_layers | 6 |
| n_heads | 4 |
| ffn_hidden | 223 |
| ple_dim | 256 (128 por board) |
| Cuantización | tok_emb: 8-bit group=128, tabla PLE A: 4-bit group=64, resto: 4-bit group=128 |
| Tamaño del Binario | 31.39 MB total (A: 4.31, B: 13.71, C: 13.37) |
| Dataset | WikiText-103 (Wikipedia) |
| Steps de Entrenamiento | 12,000 |
| Batch Size | 8 |
| Longitud de Secuencia | 128 |

### Resultados de Entrenamiento

| Métrica | Valor |
|---|---|
| Loss de Validación Final (fp32) | 5.26 |
| Perplejidad (fp32) | 192 |
| Degradación por Cuantización 4-bit | +0.62 nats |
| Perplejidad en 4-bit | 358 |
| Tokens de Entrenamiento | 12.3M |
| Tiempo de Entrenamiento | ~6.9 horas (Mac i7 CPU) |

---

## Inicio Rápido

### Prerrequisitos

- Python 3.11+
- Gestor de paquetes [uv](https://docs.astral.sh/uv/)
- Arduino IDE con paquete de board ESP32
- 3 placas ESP32-S3 N16R8

### 1. Configurar Entorno

```bash
source .venv/bin/activate
python src/dataset.py --dataset wikitext103
```

### 2. Entrenar Modelo

```bash
python src/train.py --arm ple --d-model 128 --n-layers 6 --ple-dim 256 \
  --target-core 1500000 --batch-size 8 --seq-len 128 --steps 12000 --tag wiki60m
```

### 3. Cuantizar y Exportar

```bash
python src/quantize.py --tag wiki60m
python src/export.py ple-wiki60m-s0
python src/gen_assets.py
```

### 3.1 Buscar los devices
```bash
ls /dev/cu.usbmodem*
/dev/cu.usbmodem5C372059631  /dev/cu.usbmodem5C372065471  /dev/cu.usbmodem5C4D0363671
```


### 4. Flashear Boards

```bash
# Conectar cada board una a la vez
./tools/flash_all.sh /dev/cu.usbmodemXXXX
```

### 5. Usar

1. Encender las tres boards
2. Conectar tu celular/PC a la WiFi: **ESP32-DIST-AI** (contraseña: `ai123456`)

   ![WiFi conectada a la red ESP32-DIST-AI](docs/images/net1.jpeg)

3. Abrir `http://192.168.4.1` en el navegador

   ![Detalles de red mostrando IP 192.168.4.1](docs/images/net2.jpeg)

4. Escribir un prompt y hacer clic en "Generate"

   ![Captura de la app con prompt y texto generado](docs/images/inference.jpeg)

---

## Lo que está Listo

<details>
<summary>Arquitectura — PLE TinyLM v2, Split-PLE, 56M params</summary>

- PLE TinyLM v2 con d_model=128, n_layers=6, ple_dim=256
- Split-PLE: tabla PLE dividida entre Board A y Board C (128+128 por capa)
- Tabla PLE A movida de Board A a Board C (solucionó desbordamiento de partición)
- Petición/respuesta A↔C por token vía ESP-NOW
- KV cache en Board B: atención causal multi-head con K/V caching en PSRAM (1.5 MB para 256 posiciones), RoPE usa índices reales
- MAX_SEQ_LEN=256, loop de generación=128, soporta 30+ palabras
- srand(esp_random()) para sampling no-determinista
</details>

<details>
<summary>Entrenamiento — WikiText-103, PPL 192, 12K steps, 12.3M tokens</summary>

- Descarga y tokenización BPE de WikiText-103 (112M tokens)
- Entrenamiento: 12,000 steps, batch_size=8, seq_len=128, 12.3M tokens
- Loss fp32 final: 5.26, Perplejidad: 192
- Tiempo de entrenamiento: ~6.9 horas (Mac i7 CPU)
</details>

<details>
<summary>Cuantización y Exportación — 4-bit, 31.39 MB total</summary>

- Cuantización 4-bit group-wise (+0.62 nats de degradación, PPL 4-bit 358)
- tok_emb: 8-bit group=128, tabla PLE A: 4-bit group=64, resto: 4-bit group=128
- Exportación a 3 binarios: A: 4.31 MB, B: 13.71 MB, C: 13.37 MB
- Logits golden de referencia para verificación
</details>

<details>
<summary>Firmware — 3 boards, ESP-NOW, Web UI, loop autorregresivo</summary>

- Board A: tokenizer + tok_emb 8-bit + proyección PLE + output head
- Board B: 6 capas transformer + PLE_B + PLE gating + KV cache
- Board C: tabla PLE A (4-bit group=64) + decodificador + WiFi AP + Web UI
- Protocolo ESP-NOW con fragmentación, números de secuencia y acuses de recibo
- Web UI para input de texto, output streaming y endpoint SSE
- Loop autorregresivo (hasta 128 tokens) entre las 3 boards
- Heurística de espacios en Board C
- Compatibilidad Arduino core 3.3.11, partición factory 1.25 MB
</details>

<details>
<summary>Flash y Deploy — 3 boards flasheadas con firmware + modelo</summary>

- Board A: MAC `14:c1:9f:2a:ac:c8` / puerto `5C372059631`
- Board B: MAC `14:c1:9f:2c:91:10` / puerto `5C372065471`
- Board C: MAC `28:84:85:51:dc:10` / puerto `5C4D0363671`
- Scripts de flasheo y verificación
</details>

<details>
<summary>Testing — Inferencia local confirma texto coherente (~30 palabras)</summary>

- Prueba de inferencia local confirmó ~30 palabras coherentes
- Confirmado que la cuantización 4-bit es el cuello de botella (no el protocolo)
- Comparación entre fp32, cuantizado e inferencia en dispositivo
</details>

## Lo que Falta

- [ ] **Tokenizador BPE en C** — Reemplazar búsqueda bruta en vocabulario por BPE con merges
- [ ] **MACs ESP-NOW estáticas** — Configurar MACs conocidas en vez de broadcast
- [ ] **Mejoras Web UI** — Display streaming de tokens, temperatura/top-k, indicadores de estado
- [ ] **Manejo de errores** — Retransmisión ESP-NOW y timeouts por pérdida de paquetes
- [ ] **Optimización de energía** — Deep sleep entre requests de inferencia
- [ ] **Salida de audio** — Altavoz I2S (futuro)

---

## Hardware Necesario

| Componente | Cantidad | Notas |
|---|---|---|
| ESP32-S3 N16R8 | 3 | 512KB SRAM, 8MB PSRAM, 16MB flash |
| Cables USB-C | 3 | Para flashear firmware |
| Computadora | 1 | Mac/Linux/Windows con Arduino IDE |

---

## Referencia

Este proyecto extiende el trabajo de:

- **[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)** — Ejecutando un LLM de 28.9M parámetros en un solo ESP32-S3 usando Per-Layer Embeddings
- **Google Gemma** — Arquitectura Per-Layer Embeddings
- **WikiText-103** — Dataset de Salesforce Research ([arXiv:1609.07843](https://arxiv.org/abs/1609.07843))
- **TinyStories** — Dataset de Eldan & Li (Microsoft Research, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759))
- **[karpathy/llama2.c](https://github.com/karpathy/llama2.c)** — Inspiración para correr LMs pequeños en C puro

---

## Licencia

MIT
