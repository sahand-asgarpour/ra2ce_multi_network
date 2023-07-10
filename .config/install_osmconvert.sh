#!/bin/bash
curl -O http://m.m.i24.cc/osmconvert.c
gcc -x c osmconvert.c -o osmconvert -lz -O3
rm osmconvert.c

# wget -O - http://m.m.i24.cc/osmconvert.c | gcc -x c - -lz -O3 -o osmconvert
# curl -O - http://m.m.i24.cc/osmconvert.c | gcc -x c - -lz -O3 -o osmconvert