#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo "Please run this script from the project root"
  exit
fi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vzFVFhJZDZMkJ8liROtIyzOiUY42r4TZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vzFVFhJZDZMkJ8liROtIyzOiUY42r4TZ" -O data/ATARI.tar.gz && rm -rf /tmp/cookies.txt && \
echo "Decompressing..." ; tar -xzf data/ATARI.tar.gz -C data && \
rm data/ATARI.tar.gz
