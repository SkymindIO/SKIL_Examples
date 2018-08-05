#!/usr/bin/env bash

python freeze_model.py

docker build -t skymindops/skil-ce:mnist .

docker run --rm -it -p 9008:9008 -p 8080:8080 -p9100:9100 -p8888:8888 skymindops/skil-ce:mnist