#!/usr/bin/env bash
set -euo pipefail

export SKIL_HOME=/opt/skil

pushd step2-model-trainer

data="../data/"
analysis="../step0-analyze-data/iris-analysis.json"

MODEL_PATH=${1:?"Missing model path; set it as the first argument"}
MODEL_FILE="irismodel.zip"

echo "Using: "
echo " - Model path: ${MODEL_PATH}"
echo " - Data path: ${data}"
echo " - Analysis file: ${analysis}"

mvn exec:java \
    -Dexec.mainClass="io.skymind.skil.tutorial.TrainModel" \
    -Dexec.args="--train ${data}/train --test $data/test --analysis ${analysis} --epochs 1000"

echo "Cleaning models directory ${MODEL_PATH}"
[ -d ${MODEL_PATH} ] && rm -rf ${MODEL_PATH}

${SKIL_HOME}/sbin/skil \
    --host localhost \
    --port 9008 \
    inference \
    --predictServerPort 9300  \
    --modelUri file://${MODEL_PATH} \
    --modelHistoryServerUrl http://localhost:9508

echo "Creating models directory ${MODEL_PATH}"
[ -d ${MODEL_PATH} ] || (mkdir -p ${MODEL_PATH} && chmod 777 ${MODEL_PATH})

echo "Giving SKIL a few seconds to spin up inference server before deploying model."
sleep 30s

# Deploy model to directory to be picked up by model-provider.
echo "Copying model into ${MODEL_PATH}"
cp -v ${MODEL_FILE} ${MODEL_PATH}/

popd
