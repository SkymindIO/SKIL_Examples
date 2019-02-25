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

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk

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

echo "Deploying model file"

python deploy_model.py

# Uncomment the following lines to run an additional zeppelin servers and interpreters

# echo "Running additional Zeppelin Server"
# $SKIL_HOME/sbin/skil zeppelin --name Zeppelin2 --interpreterPort 6560 --zeppelinPort 8140
#
# echo "Creating the interpreter process"
# $SKIL_HOME/sbin/skil zeppelinInterpreter --interpreterPort 6560

echo "[hit enter key to exit] or run 'docker stop <container>'"
read

bash /opt/skil/sbin/stop-skil-daemon

