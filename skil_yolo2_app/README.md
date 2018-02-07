# YOLO2 Demo on SKIL

This example is meant to show off raw TF model import into SKIL 1.0.2


## Demo Workflow
* Start up SKIL 1.0.2
* Download and convert the YOLO2 Model to the .pb TensorFlow format
* Import the model into the SKIL Model Server
* Run the YOLO2 Client from the local command line

# Get SKIL 1.0.2

Check out SKIL over at: [https://skymind.ai/platform](https://skymind.ai/platform)

You can get SKIL as either an [RPM package](https://docs.skymind.ai/v1.0.2/docs/packages) or a [Docker Image](https://docs.skymind.ai/v1.0.2/docs/docker-image) from the [downloads page](https://docs.skymind.ai/v1.0.2/docs/download)




# Downloading the YOLO2 Tensor Flow Model

Tensor Flow Model Used

This is the official website listed with the yolo900 paper 

https://pjreddie.com/darknet/yolo/

The weights are from here and are listed under YOLOv2 608x608


This repo converts it from darknet to TF and has instructions on how to get the single pb file aka the frozen graph

https://github.com/thtrieu/darkflow

## Specific Steps for Model Conversion


# Import the .pb File into the SKIL Model Server

[ stuff ]
* log into SKIL
* select the "deployments" option on the left side toolbar
* click on the "New Deployment" button
* in the models section of the newly created deployment screen, select "Import" and locate the .pb file we created previously
* For the placeholders options:
   * Names of the Input Placeholders: "input" (make sure to press 'enter' after you enter the name)
   * Names of the Output Placeholders: "output" (also make sure to press 'enter' after you enter the name)
* click on "Import Model" 
* click the "start" button on the endpoint


# Run the SKIL Client Locally

clone this repo with the command:
```
git clone [ here ]
```
build the jar:
```
mvn package
```
now run the jar from the command line:

java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/images/0012.jpg --endpoint http://localhost:9008/endpoints/tf2/model/yolo/default/

where --input can be any input image you choose, and the --endpoint parameter is the endpoint you create when you import the TF .pb file.

The jar can also take local file:// input as well.
