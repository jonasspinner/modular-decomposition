#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


RAW_PATH="data/01-raw"
DOWNLOAD_PATH="${RAW_PATH}/girg_deg_scaling"


mkdir -p "${DOWNLOAD_PATH}"

if [ ! -f "${DOWNLOAD_PATH}/edge_lists_girg_deg_scaling.zip" ]
then
  wget -P "${DOWNLOAD_PATH}" https://zenodo.org/records/8058432/files/edge_lists_girg_deg_scaling.zip
fi

if [ ! -f "${DOWNLOAD_PATH}/edge_lists_girg_deg_scaling/girg_deg_scaling_n=50000_deg=8_dim=2_ple=5.1_T=0.62_seed=1010" ]
then
  unzip "${DOWNLOAD_PATH}/edge_lists_girg_deg_scaling.zip" -d "${DOWNLOAD_PATH}"
fi