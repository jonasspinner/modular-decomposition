#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


RAW_PATH="data/01-raw"
DOWNLOAD_PATH="${RAW_PATH}/girg"


mkdir -p "${DOWNLOAD_PATH}"

if [ ! -f "${DOWNLOAD_PATH}/edge_lists_girg.zip" ]
then
  wget -P "${DOWNLOAD_PATH}" https://zenodo.org/records/8058432/files/edge_lists_girg.zip
fi

if [ ! -f "${DOWNLOAD_PATH}/edge_lists_girg/girg_n=50000_deg=10_dim=2_ple=2.25_T=0.0_seed=321" ]
then
  unzip "${DOWNLOAD_PATH}/edge_lists_girg.zip" -d "${DOWNLOAD_PATH}"
fi