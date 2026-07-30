#!/bin/bash
# Flash all 3 boards with their firmware and model binaries.
# Usage: ./flash_all.sh /dev/cu.usbmodemXXXX
#
# Prerequisites:
#   1. arduino-cli installed (brew install arduino-cli)
#   2. ESP32 board package installed (arduino-cli core install esp32:esp32)
#   3. Model exported (python src/export.py)
#   4. All 3 boards connected via USB

set -e

PORT=${1:?"Usage: $0 <serial-port>"}
FQBN="esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info"
BUILD_DIR="/tmp/esp32-dist-build"
MODEL_DIR="firmware/model"

if [ ! -f "$MODEL_DIR/board_a.bin" ]; then
    echo "Error: Model not exported. Run: python src/export.py"
    exit 1
fi

echo "=== Flashing Board A (Embeddings) ==="
echo "Connect Board A to $PORT and press Enter..."
read

arduino-cli compile \
    --fqbn "$FQBN" \
    --build-property compiler.optimization_flags=-O3 \
    --build-path "$BUILD_DIR/board_a" \
    firmware/board_a_embeddings

arduino-cli upload \
    -p "$PORT" \
    --fqbn "$FQBN" \
    --input-dir "$BUILD_DIR/board_a"

esptool.py --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash 0x150000 "$MODEL_DIR/board_a.bin"

echo "Board A flashed!"
echo ""

echo "=== Flashing Board B (Core) ==="
echo "Connect Board B to $PORT and press Enter..."
read

arduino-cli compile \
    --fqbn "$FQBN" \
    --build-property compiler.optimization_flags=-O3 \
    --build-path "$BUILD_DIR/board_b" \
    firmware/board_b_core

arduino-cli upload \
    -p "$PORT" \
    --fqbn "$FQBN" \
    --input-dir "$BUILD_DIR/board_b"

esptool.py --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash 0x150000 "$MODEL_DIR/board_b.bin"

echo "Board B flashed!"
echo ""

echo "=== Flashing Board C (Decoder + WiFi) ==="
echo "Connect Board C to $PORT and press Enter..."
read

arduino-cli compile \
    --fqbn "$FQBN" \
    --build-property compiler.optimization_flags=-O3 \
    --build-path "$BUILD_DIR/board_c" \
    firmware/board_c_decoder

arduino-cli upload \
    -p "$PORT" \
    --fqbn "$FQBN" \
    --input-dir "$BUILD_DIR/board_c"

esptool.py --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash 0x150000 "$MODEL_DIR/board_c.bin"

echo ""
echo "=== All 3 boards flashed! ==="
echo "Power cycle all boards, then monitor serial:"
echo "  Board A: arduino-cli monitor -p $PORT"
echo "  Board C: Connect to WiFi 'ESP32-DIST-AI' (pass: ai123456)"
echo "           Open http://192.168.4.1"
