#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo "Please run this script from the project root"
  exit
fi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gE3kr_ZLdsMRr263K2v1HhyfJehhW7Pi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gE3kr_ZLdsMRr263K2v1HhyfJehhW7Pi" -O data/OBJ3D_LARGE.tar.gz && rm -rf /tmp/cookies.txt && \
echo "Decompressing..." ; tar -xzf data/OBJ3D_LARGE.tar.gz -C data && \
rm data/OBJ3D_LARGE.tar.gz


