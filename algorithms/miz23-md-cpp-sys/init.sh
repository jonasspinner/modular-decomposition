#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MIZ23_DIR="${SCRIPT_DIR}/extern"

if [ ! -d "${MIZ23_DIR}" ]
then
  git clone git@github.com:mogproject/twinwidth-2023.git "${MIZ23_DIR}"
fi

FILES=(\
ds/graph/Graph.cpp \
ds/graph/Graph.hpp \
ds/queue/BucketQueue.hpp \
ds/set/ArrayBitset.hpp \
ds/set/basic_set.hpp \
ds/set/FastSet.cpp \
ds/set/FastSet.hpp \
ds/set/SortedVectorSet.hpp \
ds/tree/IntRootedForest.hpp \
modular/MDTree.cpp \
modular/MDTree.hpp \
modular/compute/assembly.cpp \
modular/compute/MDComputeNode.cpp \
modular/compute/MDComputeNode.hpp \
modular/compute/MDSolver.cpp \
modular/compute/MDSolver.hpp \
modular/compute/misc.cpp \
modular/compute/pivot.cpp \
modular/compute/promotion.cpp \
modular/compute/refinement.cpp \
readwrite/edge_list.cpp \
readwrite/edge_list.hpp \
readwrite/pace_2023.cpp \
readwrite/pace_2023.hpp \
util/logger.cpp \
util/logger.hpp \
util/profiler.cpp \
util/profiler.hpp \
util/Random.cpp \
util/Random.hpp \
util/util.cpp \
util/util.hpp)

for f in "${FILES[@]}"; do
  source="${MIZ23_DIR}/src/main/cpp/${f}"
  target="${SCRIPT_DIR}/src/${f}"
  if [ ! -s "${target}" ]
  then
    mkdir -p "$(dirname "${target}")"
    cp "${source}" "${target}"
  else
    echo "${target} already exists"
  fi
done

cd "${SCRIPT_DIR}" || exit
git apply "patch-1.patch"

