#!/usr/bin/env bash
cmake -DCMAKE_BUILD_TYPE=Release '-DCMAKE_C_COMPILER=clang-15' '-DCMAKE_CXX_COMPILER=clang++-15' -G Ninja -S .. -B ../cmake-build-release
cd ../cmake-build-release && cmake --build . --target all -j 16
cd ./src && chmod u+x main
for i in {0..5}; do
  if [[ $i == 0 ]]; then
    n=32
  else
    n=$((i * 64))
  fi
  while : ; do
    if ./main $n; then
      break;
    fi
  done
done
if [ -d ../../test/logs ]; then
  rm -rf ../../test/logs
  mkdir ../../test/logs
else
  mkdir ../../test/logs
fi
mv ./*.txt ../../test/logs/
cd ../../test || return
python3 plot.py
