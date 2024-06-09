#!/bin/bash
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g ge43
#PJM -j
set -x
source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load cuda/12.1
source .venv/bin/activate

ratarmount corpus/hq-youtube.tar /dev/shm/hq-youtube_mount
TYPE=hubert
CKPT_PATH=models/hubert/iter2_ckpt/checkpoints/checkpoint_last.pt
LAYER=12
MANIFEST=manifest/valid.tsv
KM_MODEL_PATH=models/km_model.bin
OUT_QUANTIZED_FILE=quantized/quantized_valid
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
  --feature_type $TYPE \
  --kmeans_model_path $KM_MODEL_PATH \
  --acoustic_model_path $CKPT_PATH \
  --layer $LAYER \
  --manifest_path $MANIFEST \
  --out_quantized_file_path $OUT_QUANTIZED_FILE \
  --extension ".flac"
