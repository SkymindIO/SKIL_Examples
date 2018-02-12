# Object Detection with YOLO2 Demo on SKIL

This example is meant to show off raw TF model import into SKIL 1.0.2. We've chosen object detection in computer vision as the application we want to demo in the context of Tensor Flow model import on SKIL. For the purposes of demoing computer vision on SKIL, we'll use a YOLO network for object deteciton as the application. The original YOLO2 model is in the darknet framework format, but fortunately we have a way of converting this format to the tensor flow format.

The purpose of this demo on SKIL is two-fold:
1. show off the native TensorFlow model import capabilities of the SKIL platform
2. show off a live real-world computer vision object detection demo on the SKIL platform


## Demo Workflow
* Start up SKIL 1.0.2
* Download and convert the YOLO2 Model to the .pb TensorFlow format
* Import the model into the SKIL Model Server
* Run the YOLO2 Client from the local command line

# Get SKIL 1.0.2

Check out SKIL over at: [https://skymind.ai/platform](https://skymind.ai/platform)

You can get SKIL as either an [RPM package](https://docs.skymind.ai/v1.0.2/docs/packages) or a [Docker Image](https://docs.skymind.ai/v1.0.2/docs/docker-image) from the [downloads page](https://docs.skymind.ai/v1.0.2/docs/download)




# Working with the YOLO Model

![alt text](https://pjreddie.com/media/image/model2.png "Image Courtesy of YOLO Website")

YOLO is a deep network for real-time object detection and classification. 
Paper: 
* [version 1](https://arxiv.org/pdf/1506.02640.pdf)
* [version 2](https://arxiv.org/pdf/1612.08242.pdf)

As described in the first paper:

> We pretrain our convolutional layers on the ImageNet
> 1000-class competition dataset [30]. For pretraining we use
> the first 20 convolutional layers from Figure 3 followed by a
> average-pooling layer and a fully connected layer. We train
> this network for approximately a week and achieve a single
> crop top-5 accuracy of 88% on the ImageNet 2012 validation
> set, comparable to the GoogLeNet models in Caffe’s
> Model Zoo [24]. We use the Darknet framework for all
> training and inference [26].

In the second paper the authors go on to state further accuracy improvements:

> The improved model, YOLOv2, is state-of-the-art on
> standard detection tasks like PASCAL VOC and COCO. Using
> a novel, multi-scale training method the same YOLOv2
> model can run at varying sizes, offering an easy tradeoff
> between speed and accuracy. At 67 FPS, YOLOv2 gets
> 76.8 mAP on VOC 2007. At 40 FPS, YOLOv2 gets 78.6
> mAP, outperforming state-of-the-art methods like Faster RCNN
> with ResNet and SSD while still running significantly faster

Both papers reference the darknet framework as:

> [26] J. Redmon. Darknet: Open source neural networks in c.
> http://pjreddie.com/darknet/, 2013–2016. 3

The specific version of the YOLO model [setup](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg) we use in this example is based on the YOLOv2 architecture was trained on the [COCO dataset](http://cocodataset.org/#home) and can recognize [80 distinct classes](https://github.com/pjreddie/darknet/blob/master/data/coco.names).

## Leveraging the Darknet Framework to Extract the TensorFlow Model

So for this demo we want a TensorFlow model to import into SKIL. For the purposes of demoing computer vision on SKIL, we'll use a YOLO network for object deteciton as the application. The original YOLO2 model is in the darknet framework format, but fortunately we have a way of converting this format to the tensor flow format.

* The official website listed for the yolo9000 paper:
   * https://pjreddie.com/darknet/yolo/
* The github repo for the darknet framework:
   * https://github.com/pjreddie/darknet
* Specific YOLO model weights in darknet format:
   * https://pjreddie.com/darknet/yolo/
   * The weights are from here and are listed under YOLOv2 608x608
      * [weights](https://pjreddie.com/media/files/yolo.weights)
      * [cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg)

Now that we know where to get the darknet-format yolo model, let's move on to converting it to the TensorFlow format.

## Specific Steps for Model Conversion

Now that we have the darknet-format yolo model we need to convert it to the TensorFlow format. The repo linked below converts the darknet-format model from darknet to TensorFlow. The repo has instructions on how to get the single pb file aka the frozen graph.

https://github.com/thtrieu/darkflow

---
> ### Model Conversion is a Lot of Work 
> It's a lot easier if you use the one we already converted to the TensorFlow format for you:
> [yolo pb file](https://github.com/deeplearning4j/dl4j-test-resources/blob/master/src/main/resources/tf_graphs/examples/yolov2_608x608/frozen_model.pb)
---

Skip down to the section on ["Saving the built graph to a protobuff file"](https://github.com/thtrieu/darkflow#save-the-built-graph-to-a-protobuf-file-pb) and check out the specific commands on how to get darknet to save to the TensorFlow format.

As noted in their docs, the name of input tensor and output tensor (e.g., "placeholders") are respectively 'input' and 'output'. For more information on how to use protobuf files, check out the official [docs of Tensorflow on C++ API](https://www.tensorflow.org/api_docs/cc/). 



# Import the TensorFlow Protobuff File into the SKIL Model Server

Now we can log into SKIL and import the TensorFlow protobuff (.pb) file we created in the previous step.

* Log into SKIL
* Select the "deployments" option on the left side toolbar
* Click on the "New Deployment" button
* In the models section of the newly created deployment screen, select "Import" and locate the .pb file we created previously
* For the placeholders options:
   * Names of the Input Placeholders: "input" (make sure to press 'enter' after you enter the name)
   * Names of the Output Placeholders: "output" (also make sure to press 'enter' after you enter the name)
* Click on "Import Model" 
* Click the "start" button on the endpoint

It will take a few seconds for the page to report that the endpoint has successfully started. Once the page lists the endpoint as running, however, you will have access to the model from the listed endpoint on the page. The endpoint URI will look something like:

http://localhost:9008/endpoints/tf2/model/yolo/default/

Now we need a client application to query this endpoint and get object detection predictions.

# Run the SKIL Client Locally with the Sample Client Application

Clone this repo with the command to get the included YOLOv2 sample application that will retrive predictions and render the bounding boxes locally:
```
git clone git@github.com:SkymindIO/SKIL_Examples.git
```
build the jar:
```
cd skil_yolo2_app/client_app
mvn package
```
This will build a jar named "skil-example-yolo2-tf-1.0.0.jar" in the ./target subdirectory of the client_app/ subdirectory.

Now that we have a client application jar, we can run the yolo2 client jar from the command line:
```
java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input [image URI] --endpoint [SKIL Endpoint URI]
```

where 

* `--input` can be any input image you choose (local file with the file:// prefix, or an image file via an internet URI with a http:// prefix)
* `--endpoint` parameter is the endpoint you create when you import the TF .pb file

An example of this command in usage would be:

```
java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/images/0012.jpg --endpoint http://localhost:9008/endpoints/tf2/model/yolo/default/
```


# Reference Material on the YOLO Family of Networks

* Understanding bounding box mechanics in object detection (aka “understanding YOLO output)
   * http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html


