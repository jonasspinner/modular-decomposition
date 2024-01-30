#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


RAW_PATH="data/01-raw"
DOWNLOAD_PATH="${RAW_PATH}/real"


mkdir -p "${DOWNLOAD_PATH}"

if [ ! -f "${DOWNLOAD_PATH}/edge_lists_real.zip" ]
then
  wget -P "${DOWNLOAD_PATH}" https://zenodo.org/records/8058432/files/edge_lists_real.zip
fi

if [ ! -f "${DOWNLOAD_PATH}/edge_lists_real/208bit" ]
then
  unzip "${DOWNLOAD_PATH}/edge_lists_real.zip" -d "${DOWNLOAD_PATH}"
fi