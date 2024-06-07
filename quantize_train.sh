#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -g ge43
#PJM -j
set -x
source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load cuda/12.1
source .venv/bin/activate

ratarmount corpus/hq-youtube.tar /dev/shm/hq-youtube_mount
rank=0
for file in manifest/train_split/split_*; do
  TYPE=hubert
  CKPT_PATH=models/hubert/iter2_ckpt/checkpoints/checkpoint_last.pt
  LAYER=12
  MANIFEST=$file
  KM_MODEL_PATH=models/km_model.bin
  OUT_QUANTIZED_FILE=quantized/quantized_$(basename $file)
  CUDA_VISIBLE_DEVICES=$rank python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".flac" &
  rank=$(expr $rank + 1)
done
wait
