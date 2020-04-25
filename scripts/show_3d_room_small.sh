#!/usr/bin/env bash
cd src && \
python main.py --task show --config 'configs/3d_room_small.yaml' \
  resume True resume_ckpt '../pretrained/3d_room_small.pth' device $1 \
  show.indices "[0, 1, 2, 3, 4]"

