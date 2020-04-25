#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo "Plese run this script from the project root"
  exit
fi
[ -d data/OBJ3D_LARGE ] && [ -f pretrained/3d_room_large.pth ] && sh scripts/show_3d_room_large.sh 'cuda:0'
[ -d data/OBJ3D_SMALL ] && [ -f pretrained/3d_room_small.pth ] && sh scripts/show_3d_room_small.sh 'cuda:0'
[ -d data/ATARI ] && [ -f pretrained/atari_joint.pth ] && sh scripts/show_atari_joint.sh 'cuda:0'
[ -d data/ATARI ] && [ -f pretrained/atari_riverraid.pth ] && sh scripts/show_atari_riverraid.sh 'cuda:0'
[ -d data/ATARI ] && [ -f pretrained/atari_spaceinvaders.pth ] && sh scripts/show_atari_spaceinvaders.sh 'cuda:0'

