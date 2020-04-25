#!/usr/bin/env bash
cd src && \
python main.py --task show --config 'configs/atari_joint.yaml' \
  resume True resume_ckpt '../pretrained/atari_joint.pth' device $1 \
  show.indices "[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]"

