This folder contains the basic files required for a DevOps team to deploy a single model on a single node SKIL instance.

## Running the script
Make sure you have "[TensorFlow](https://www.tensorflow.org/install/)" installed in your machine. After that run the following script:

```cmd
sh script.sh
```

## File contents
1. **freeze_model.py**: A Python script for training a CNN on MNIST dataset and freezing the model to an output graph named `output_graph.pb` (TensorFlow Protobuff file format) inside the `data_directory` folder. 
2. **deploy_model.py**: A Python script for deploying a model to a SKIL server. (Added to the Dockerfile). 
3. **deploy_model.sh**: A shell script for starting the SKIL server and running the `deploy_model.py` script. (Added to the Dockerfile).
4. **Dockerfile**: Docker file configuration for deploying a model to a single node SKIL server. 
5. **script.sh**: A shell script for freezing a model, building and starting the docker container.