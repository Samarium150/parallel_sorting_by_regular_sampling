#!/usr/bin/env bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang '-DCMAKE_CXX_COMPILER=clang++' -G Ninja -S .. -B ../cmake-build-release
cd ../cmake-build-release && cmake --build . --target all -j 16
cd ./src && chmod u+x main && ./main "auto"
if [ -d ../../test/logs ]; then
  rm -rf ../../test/logs
  mkdir ../../test/logs
fi
mv ./*.txt ../../test/logs/
cd ../../test || return
python3 plot.py
