#!/usr/bin/env bash
set -euo pipefail

rm docker/md_exp.tar || true
rm md_exp.zip || true
rm -r md_exp || true

docker image build -t md_exp -f docker/Dockerfile .
docker save -o docker/md_exp.tar md_exp

mkdir -p md_exp/
cp docker/md_exp.tar md_exp/
cp docker/scripts/enter.sh md_exp/
cp docker/scripts/kill.sh md_exp/
cp docker/scripts/load.sh md_exp/
cp docker/scripts/run.sh md_exp/

zip -r md_exp.zip md_exp/
rm docker/md_exp.tar || true
# rm -rf md_exp/
