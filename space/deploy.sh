#!/usr/bin/env bash
# Upload this folder plus the engine package to the Hugging Face Space (Gradio + ZeroGPU).
set -euo pipefail
cd "$(dirname "$0")"
SPACE=${SPACE:-datafreak/laya-chess}
HF=${HF:-../engine/.venv/bin/hf}
STAGE=$(mktemp -d)
cp app.py README.md requirements.txt packages.txt "$STAGE"/
rsync -a --exclude __pycache__ ../engine/laya_chess "$STAGE"/
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --commit-message "${1:-Update LayaChess Space}"
rm -rf "$STAGE"
echo "https://huggingface.co/spaces/$SPACE"
