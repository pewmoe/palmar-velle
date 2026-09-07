#!/usr/bin/env bash
# Downloads the MediaPipe hand-landmark model (ONNX port maintained by the
# OpenCV Zoo project) that gesture-control needs.
#
# Source: https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe
# Mirror used here: https://huggingface.co/opencv/handpose_estimation_mediapipe
#   (GitHub's raw.githubusercontent.com only serves this file's Git-LFS
#   *pointer*, not the actual weights -- Hugging Face's /resolve/ endpoint
#   redirects to the real binary, which is what we want.)
# License: Apache 2.0 (see the LICENSE file alongside the model in that repo)
set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$DEST_DIR"

URL="https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
OUT="$DEST_DIR/handpose_estimation_mediapipe_2023feb.onnx"

echo "Downloading hand landmark model to $OUT ..."
curl -fL "$URL" -o "$OUT"

SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
if [ "$SIZE" -lt 1000000 ]; then
    echo "ERROR: downloaded file is only $SIZE bytes -- that's too small to be" >&2
    echo "the real model (expected ~4 MB). You probably got an LFS pointer" >&2
    echo "instead of the binary. Try downloading it by hand from:" >&2
    echo "  $URL" >&2
    rm -f "$OUT"
    exit 1
fi

echo "Done ($SIZE bytes). Run with: cargo run --release -- --model $OUT"
