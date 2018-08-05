#!/usr/bin/env bash

export SKIL_HOME=/opt/skil
#export SKIL_CLASS_PATH=/opt/skil/lib/*:/opt/skil/native/*:/opt/skil/jackson-2.5.1/*
export SKIL_LOG_DIR=/var/log/skil
export SKIL_PID_FILE=/run/skil/skil.pid

export SKIL_BACKEND=cpu
export SKIL_LICENSE_PATH=/etc/skil/license.txt
export SKIL_PUBLIC_KEY_PATH=/etc/skil/publickey.txt

export ZOOKEEPER_HOST=localhost
export ZOOKEEPER_EMBEDDED=true

bash /opt/skil/sbin/pre-start.sh
bash /opt/skil/sbin/start-skil-daemon

echo "Waiting for all SKIL Services to start..."
while [ $(jps | wc -l) != 5 ]
do
	sleep 1s
done

until [ "grep join /var/log/skil/zeppelin-*.log" ]
do
	sleep 1s
done

echo "SKIL Ready"

echo "[hit enter key to exit] or run 'docker stop <container>'"

echo "Deploying model file"

python deploy_model.py

read

bash /opt/skil/sbin/stop-skil-daemon

