# SKIL Examples

This repository contains public examples for the [SKIL Platform](docs.skymind.ai). It
contains basic tutorials and guides, as well as complete SKIL applications and notebooks
that can be imported and run from SKIL.

## Getting started

For each application shown here you need a running SKIL instance. SKIL CE is free and
can be [downloaded and installed in a few simple steps](https://docs.skymind.ai/docs/docker-image).
You'll find more about the specific requirements of each application in the respective folders.

## Applications and tutorials

- [CLI tutorial](https://github.com/SkymindIO/SKIL_Examples/tree/master/CLI-tutorial): A four-step, lightning introduction to the SKIL command line interface. You'll analyze the input data, deploy a transform process, then deploy a model and finish by testing SKIL's REST client. (_Command line & Java_)
- [Clinical LSTM application]((https://github.com/SkymindIO/SKIL_Examples/tree/master/Clinical-LSTM-app): This example shows how to train a recurrent neural network written with DL4J on electronic health record data and deploy it with SKIL. (_Java_)
- [SKIL deployment with Docker](https://github.com/SkymindIO/SKIL_Examples/tree/master/Docker-deployment): This example shows how import and deploy a TensorFlow model to SKIL, all within a Docker container. (_Docker & Python_)
- [Fraud detection application](https://github.com/SkymindIO/SKIL_Examples/tree/master/Fraud-anomaly-detection-app): In this application you'll learn how to build an anomaly detection system to recognize fraudulent behaviour. (_Python notebook_)
- [CIFAR model deployment](https://github.com/SkymindIO/SKIL_Examples/tree/master/Keras-Cifar-model-deployment): This basic workflow example shows you how to deploy a Keras model trained on the CIFAR dataset. (_Python_).
- [Salesforce app](https://github.com/SkymindIO/SKIL_Examples/tree/master/Salesforce-app): Salesforce SKIL application
- [Sequence classification app](https://github.com/SkymindIO/SKIL_Examples/tree/master/UCI-sequence-classification): This application shows you a detailed example of a sequence classification problem on synthetic control chart time series. (_Java_)
- [Object detection app](https://github.com/SkymindIO/SKIL_Examples/tree/master/YOLO-object-detection-app): In this application you'll learn how the _You only look once_ (YOLO) model can be used for real-time object detection within SKIL. (_Java_)

## Notebooks

All notebooks found in the `notebook` folder of this repository contain Zeppelin notebooks in JSON format that can be [imported into any SKIL experiment](https://docs.skymind.ai/docs/conducting-experiments).

- [Training & deploying Keras and DL4J models in the same notebook](https://github.com/SkymindIO/SKIL_Examples/blob/master/notebooks/end_to_end.json): In this end-to-end example you'll see how to mix and match several deep learning frameworks (and programming languages) in the same Zeppelin notebook. (_Python & Scala_)
- [Keras model deployment](https://github.com/SkymindIO/SKIL_Examples/blob/master/notebooks/python_keras_tf_mnist.json): Deploy a simple Keras model from the SKIL UI. (_Python_)
- [TensorFlow model deployment](https://github.com/SkymindIO/SKIL_Examples/blob/master/notebooks/python_tf_mnist.json): Deploy a TensorFlow model with SKIL. (_Python_)
- [Deploying a KNN model](https://github.com/SkymindIO/SKIL_Examples/blob/master/notebooks/scala_knn_smile_test.json): An example showing you how to deploy a K-nearest-neighbor model with SKIL. (_Scala_)
- [Running a Spark training job with SKIL]((https://github.com/SkymindIO/SKIL_Examples/blob/master/notebooks/spark_example.json): This notebook shows you how to scale-out your DL4J deep learning model on Apache Spark. (_Scala_)
- [UCI sequence classification example](https://github.com/SkymindIO/SKIL_Examples/blob/master/notebooks/uci_quickstart_notebook.json): The notebook corresponding to the above listed sequence classification example. (_Scala_)