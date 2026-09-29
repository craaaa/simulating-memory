#!/bin/bash
# Validate heldout.sbatch's output-path logic: repeats of one candidate get distinct dirs,
# the first occurrence keeps the plain name, and TAG separates run generations so a new
# job's baseline cannot overwrite an older one (job 18719683 did exactly that, and runs/ is
# gitignored, so the comparison recorded against the destroyed baseline was unreproducible).
set -u

paths() {
  local _BASE="meta_harness/runs/heldout${TAG:+/$TAG}"
  declare -A _SEEN=()
  local c _N
  for c in "$@"; do
    _SEEN[$c]=$(( ${_SEEN[$c]:-0} + 1 ))
    _N=${_SEEN[$c]}
    if [ "$_N" -eq 1 ]; then
      echo "${_BASE}/${c}"
    else
      echo "${_BASE}/${c}_rep${_N}"
    fi
  done
}

echo "== with TAG=postfix"
TAG=postfix paths respond_first_v2 baseline respond_first_v2 baseline respond_first_v2 baseline

echo "== with no TAG (must match the paths already on disk)"
TAG= paths respond_first baseline respond_first baseline respond_first
