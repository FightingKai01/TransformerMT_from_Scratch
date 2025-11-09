#!/bin/bash
# ==========================
# Script: test.sh
# Function: Test trained Transformer model on IWSLT2017 EN-DE test dataset
# ==========================

set -e

echo "🔍 Starting test ..."

python test.py \
  --dataset_name "iwslt2017" \
  --dataset_config "iwslt2017-en-de" \
  --src_lang "en" \
  --tgt_lang "de" \
  --model_name "Helsinki-NLP/opus-mt-en-de" \
  --max_len 1024 \
  --batch_size 64 \
  --model_ckpt "./checkpoints/best_checkpoint.pt"

echo "✅ Test finished!"
