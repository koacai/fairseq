#!/bin/bash
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g ge43
#PJM -L jobenv=singularity
#PJM -j
set -x
source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load cuda/12.1
module load singularity/3.9.5

singularity exec container.sif mkdir /dev/shm/hq-youtube_mount
TYPE=hubert
CKPT_PATH=models/hubert/iter2_ckpt/checkpoints/checkpoint_last.pt
LAYER=12
MANIFEST=manifest/valid.tsv
KM_MODEL_PATH=models/km_model.bin
OUT_QUANTIZED_FILE=quantized/quantized_valid
export PYTHONPATH=.
singularity exec --bind `pwd` --nv -B corpus/hq-youtube.sqfs:/dev/shm/hq-youtube_mount:image-src=/ container.sif python3 examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
  --feature_type $TYPE \
  --kmeans_model_path $KM_MODEL_PATH \
  --acoustic_model_path $CKPT_PATH \
  --layer $LAYER \
  --manifest_path $MANIFEST \
  --out_quantized_file_path $OUT_QUANTIZED_FILE \
  --extension ".flac"
