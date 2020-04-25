#!/usr/bin/env bash
cd src && \
python main.py --task show --config 'configs/atari_riverraid.yaml' \
  resume True resume_ckpt '../pretrained/atari_riverraid.pth' device $1 \
  show.indices "[0, 1, 2, 3, 4]"

