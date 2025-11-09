#!/bin/bash
# ==========================
# Script: train.sh
# Function: Train and evaluate TransformerMT model on IWSLT2017 EN-DE train and validation dataset
# ==========================

set -e

echo "🚀 Starting training ..."

python train.py \
  --dataset_name "iwslt2017" \
  --dataset_config "iwslt2017-en-de" \
  --src_lang "en" \
  --tgt_lang "de" \
  --model_name "Helsinki-NLP/opus-mt-en-de" \
  --max_len 1024 \
  --batch_size 64 \
  --limit_train_samples 0 \
  --num_layers 2 \
  --d_model 256 \
  --num_heads 8 \
  --d_ffn 1024 \
  --dropout_emb 0.1 \
  --dropout_atten 0.1 \
  --dropout_ffn 0.1 \
  --num_epochs 50 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --save_dir "./checkpoints" \
  --result_dir "../results"

echo "✅ Training finished!"
