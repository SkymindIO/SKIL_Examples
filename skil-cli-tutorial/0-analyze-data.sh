#!/usr/bin/env bash
set -euo pipefail

mvn clean install -DskipTests

pushd step0-analyze-data

OUTPUT_DIR=${1:-${PWD}}
echo "Using output directory: ${OUTPUT_DIR}"

mvn exec:java \
    -Dexec.mainClass="io.skymind.skil.tutorial.AnalyzeIrisData" \
    -Dexec.args="--input ../data/train --output ${OUTPUT_DIR}"
popd
