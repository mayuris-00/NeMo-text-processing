#!/bin/bash
# Test whether an OLDER base image (with older g++) can compile openfst 1.7.9's
# fst/fst.h. The current miniconda3:latest ships g++ 14 which rejects the old
# header. We test miniconda3:4.6.14 (Debian stretch, g++ 6.3) -- consistent with
# the Dockerfile's existing "stretch" apt override.

echo "########## Testing base: continuumio/miniconda3:4.6.14 (Debian stretch / g++6) ##########"
docker run --rm continuumio/miniconda3:4.6.14 bash -c '
  echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list
  apt-get update >/dev/null 2>&1
  apt-get install -y --reinstall build-essential >/dev/null 2>&1
  echo "--- installing thrax (conda) ---"
  conda install -c conda-forge thrax=1.3.4 -y 2>&1 | tail -4
  echo "--- g++ version ---"
  g++ --version | head -1
  echo "--- header present? ---"
  ls -la /opt/conda/include/fst/fst.h 2>&1
  printf "#include <fst/fst.h>\nint main(){return 0;}\n" > /tmp/t.cpp
  echo "--- compile test ---"
  if g++ -I/opt/conda/include /tmp/t.cpp -o /tmp/t 2>/tmp/err; then
    echo "RESULT: COMPILE_OK  (this base image fixes the build)"
  else
    echo "RESULT: COMPILE_FAIL"
    head -8 /tmp/err
  fi
'
