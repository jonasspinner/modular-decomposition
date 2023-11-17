#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


RAW_PATH="data/01-raw"
PACE2023_PATH="${RAW_PATH}/pace2023"

mkdir -p "${PACE2023_PATH}"

if [ ! -f "${PACE2023_PATH}/exact-public.zip" ]
then
  wget -P "${PACE2023_PATH}" https://cloudtcs.tcs.uni-luebeck.de/index.php/s/Dm5pZfzxkoP8cL6/download/exact-public.zip
fi

if [ ! -f "${PACE2023_PATH}/exact-private.zip" ]
then
  wget -P "${PACE2023_PATH}" https://cloudtcs.tcs.uni-luebeck.de/index.php/s/eitXacoY5ACHqtx/download/exact-private.zip
fi

if [ ! -f "${PACE2023_PATH}/heuristic-public.zip" ]
then
  wget -P "${PACE2023_PATH}" https://cloudtcs.tcs.uni-luebeck.de/index.php/s/QMxJFWgDZF4bEo2/download/heuristic-public.zip
fi

if [ ! -f "${PACE2023_PATH}/heuristic-private.zip" ]
then
  wget -P "${PACE2023_PATH}" https://cloudtcs.tcs.uni-luebeck.de/index.php/s/PPpJsy6J3XGipHP/download/heuristic-private.zip
fi

GRAPH_NAMES=()
for i in $(seq 100)
do
  GRAPH_NAMES+=("$(printf "exact_%03d.gr" "$i")")
  GRAPH_NAMES+=("$(printf "heuristic_%03d.gr" "$i")")
done

MISSING_FILES=""
for file in "${GRAPH_NAMES[@]}"
do
  if [ ! -f "${PACE2023_PATH}/${file}" ]
  then
    echo "${file} file not found"
    MISSING_FILES="${MISSING_FILES} ${file}"
  fi
done

if [ -n "${MISSING_FILES}" ]
then
    unzip "${PACE2023_PATH}/exact-public.zip" -d "${PACE2023_PATH}"
    unzip "${PACE2023_PATH}/exact-private.zip" -d "${PACE2023_PATH}"
    unzip "${PACE2023_PATH}/heuristic-public.zip" -d "${PACE2023_PATH}"
    unzip "${PACE2023_PATH}/heuristic-private.zip" -d "${PACE2023_PATH}"
    mv "${PACE2023_PATH}"/exact-public/*.gr.xz "${PACE2023_PATH}"
    mv "${PACE2023_PATH}"/exact-private/*.gr.xz "${PACE2023_PATH}"
    mv "${PACE2023_PATH}"/heuristic-public/*.gr.xz "${PACE2023_PATH}"
    mv "${PACE2023_PATH}"/heuristic-private/*.gr.xz "${PACE2023_PATH}"
    rm -r "${PACE2023_PATH}"/exact-public
    rm -r "${PACE2023_PATH}"/exact-private
    rm -r "${PACE2023_PATH}"/heuristic-public
    rm -r "${PACE2023_PATH}"/heuristic-private
    rm -r "${PACE2023_PATH}"/__MACOSX

    for file in "${PACE2023_PATH}"/*.gr.xz
    do
      [ -f "${file%.xz}" ] || xz --decompress "${file}"
    done
fi
