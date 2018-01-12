#!/usr/bin/env bash
set -euo pipefail

export SKIL_HOME=/opt/skil

pushd step3-deployment-test

DATAVEC_ADDRESS="localhost:9200"
INFERENCE_ADDRESS="localhost:9300"

[ $# == 2 ] && DATAVEC_ADDRESS=${1} INFERENCE_ADDRESS=${2}
[ $# == 1 ] && INFERENCE_ADDRESS=${1}

echo "Using: "
echo " - Datavec Address: ${DATAVEC_ADDRESS}"
echo " - Inference Address: ${INFERENCE_ADDRESS}"

mvn exec:java \
    -Dexec.mainClass="io.skymind.skil.tutorial.RestClient" \
    -Dexec.args="${DATAVEC_ADDRESS} ${INFERENCE_ADDRESS}"
