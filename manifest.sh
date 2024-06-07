#!/bin/bash
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g ge43
#PJM -j
source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load cuda/12.1
source .venv/bin/activate

ratarmount corpus/hq-youtube.tar /dev/shm/hq-youtube_mount
python3 examples/wav2vec/wav2vec_manifest.py /dev/shm/hq-youtube_mount \
  ----dest manifest \
  ----ext flac \
  ----valid-percent 0.01
