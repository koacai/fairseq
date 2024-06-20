#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g ge43
#PJM -j

set -x
source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load cuda/12.1
source .venv/bin/activate

ratarmount corpus/hq-youtube.tar.xz /dev/shm/hq-youtube_mount

N_CLUSTERS=500
TYPE=hubert
CKPT_PATH=models/hubert/iter2_ckpt/checkpoints/checkpoint_last.pt
LAYER=12
MANIFEST=manifest/train.tsv
KM_MODEL_PATH=models/km_model.bin

PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH
