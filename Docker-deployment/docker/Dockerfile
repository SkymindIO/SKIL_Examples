FROM skymindops/skil-ce

USER root

ADD deploy_model.py /deploy_model.py
ADD data_directory/output_graph.pb /model.pb
ADD deploy_model.sh /deploy_model.sh

RUN chmod 755 /deploy_model.sh

RUN yum install -y epel-release 
RUN yum install -y python-pip 
RUN pip install numpy skil-client

# PLD
EXPOSE 9008 
# File Server
EXPOSE 9508
# Zeppelin
EXPOSE 8080
# DL4J UI first port
EXPOSE 9002 
# ModelHistoryServer port
EXPOSE 9100

CMD ["/deploy_model.sh"]
