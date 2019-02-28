import pprint
import unittest

import numpy
import skil_client
from skil_client import Configuration, ApiClient, CreateDeploymentRequest
from skil_client import ImportModelRequest, SetState
from skil_client.rest import ApiException

debug = False

host = "localhost" # Rename this to the host you are using 

config = Configuration()
config.host = "{}:9008".format(host)  # change this if you're using a different port number for the general API!
config.debug = debug
api_client = ApiClient(configuration=config)
# create an instance of the API class
api_instance = skil_client.DefaultApi(api_client=api_client)

config_mh = Configuration()
config_mh.host = "{}:9100".format(host)  # change this if you're using a different port number for the model server!
config_mh.debug = debug
api_client_mh = ApiClient(configuration=config_mh)
# create an instance of the Model History API class
api_instance_mh = skil_client.DefaultApi(api_client=api_client_mh)

# authenticate
pp = pprint.PrettyPrinter(indent=4)
try:
    print("Authenticating with SKIL API...")
    credentials = skil_client.Credentials(user_id="admin", password="admin") # Update this with the ID and password you're using for your SKIL server
    token = api_instance.login(credentials)
    pp.pprint(token)
    # add credentials to config
    config.api_key['authorization'] = token.token
    config.api_key_prefix['authorization'] = "Bearer"
    # for model history
    config_mh.api_key['authorization'] = token.token
    config_mh.api_key_prefix['authorization'] = "Bearer"
except ApiException as e:
    print("Exception when calling DefaultApi->login: %s\n" % e)

print("Uploading model, please wait...")
modelFile = "/model.pb"
uploads = api_instance.upload(file=modelFile)
pp.pprint(uploads)

model_file_path = "file://" + uploads.file_upload_response_list[0].path
pp.pprint(model_file_path)

deployment_name = "mnist"
create_deployment_request = CreateDeploymentRequest(deployment_name)
deployment_response = api_instance.deployment_create(create_deployment_request)

pp.pprint(deployment_response)

model_name = "tf_model_mnist"
uris = ["{}/model/{}/default".format(deployment_name, model_name),
        "{}/model/{}/v1".format(deployment_name, model_name)]

deploy_model_request = ImportModelRequest(model_name,
                                          1, 
                                          file_location=model_file_path,
                                          model_type="model",
                                          uri=uris,
                                          input_names=["input_node", "keep_prob_input"], 
                                          output_names=["output_node"])

model_deployment_response = api_instance.deploy_model(deployment_response.id, deploy_model_request)
pp.pprint(model_deployment_response)

model_state_change_response = api_instance.model_state_change(deployment_response.id,
                                                              model_deployment_response.id,
                                                              SetState("start"))
pp.pprint(model_state_change_response)

import time

# Checking if the model is already started
print("\nStart serving model...")
while True:
    time.sleep(5)
    
    # Query the model state
    model_state = api_instance.model_state_change(deployment_response.id, 
                                    model_deployment_response.id, 
                                    SetState("start")).state
    
    if model_state == "started":
      print("Model server started successfully!")
      break
    else:
      print("wait...")
