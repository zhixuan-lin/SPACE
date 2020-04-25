#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo "Plese run this script from the project root"
  exit
fi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gUvLTfy5pKeLa6k3RT8GiEXWiGG8XzzD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gUvLTfy5pKeLa6k3RT8GiEXWiGG8XzzD" -O pretrained/pretrained.tar.gz && rm -rf /tmp/cookies.txt && \
echo "Decompressing..."; tar -xzf pretrained/pretrained.tar.gz -C pretrained && \
rm pretrained/pretrained.tar.gz
