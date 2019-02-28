#!/usr/bin/env bash
set -euo pipefail

export SKIL_HOME=/opt/skil

pushd step1-datavec-pipeline

JSON_PATH=${1:?"Missing JSON path; set it as the first argument"}
INPUT_FILE="../step0-analyze-data/iris-analysis.json"
OUTPUT_FILE="iris-inference-transform.json"

echo "Using: "
echo " - JSON path: ${JSON_PATH}"
echo " - Input: ${INPUT_FILE}"
echo " - Output: ${OUTPUT_FILE}"

mvn exec:java \
    -Dexec.mainClass="io.skymind.skil.tutorial.CreateInferenceTransformDescription" \
    -Dexec.args="--input ${INPUT_FILE} --output ${JSON_PATH}"

${SKIL_HOME}/sbin/skil \
    --host localhost \
    --port 9008 \
    datavec \
    --jsonPath ${JSON_PATH}/${OUTPUT_FILE} \
    --dataVecPort 9200 \
    --dataType CSV

popd
