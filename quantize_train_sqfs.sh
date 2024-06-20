#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -g ge43
#PJM -L jobenv=singularity
#PJM -j

set -x
source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load cuda/12.1
module load singularity/3.9.5

singularity exec container.sif mkdir /dev/shm/hq-youtube_mount
singularity exec --bind `pwd` --nv -B corpus/hq-youtube.sqfs:/dev/shm/hq-youtube_mount:image-src=/ container.sif bash quantize.sh