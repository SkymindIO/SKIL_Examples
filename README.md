# SKIL_Examples
Public examples for the SKIL Platform (client examples, notebooks, etc). Versioned for each release.

## Examples Included in the v1.0.1 Release

* Endpoint Examples
* MNIST Keras Notebook (python) and Java Client Example

## Running the MNIST Keras Notebook and Model Server Client

Includes a full SKIL application example with a JSON Keras python notebook and an example client in java that base64 encodes MNIST images and sends them to a SKIL Model Server endpoint for classification, returning labels and probabilities.

### Running Model Server Client Code

(from the client_app subdir)

To send a blank image to the model server to test out a non-MNIST image:

java -jar ./target/skil-example-mnist-tf-1.0.0.jar --input blank --endpoint http://localhost:9008/endpoints/mnist/model/mnistmodel/default/


A few MNIST digit images (28x28 pngs) are included in the example, and we can send those for classification as well:

java -jar ./target/skil-example-mnist-tf-1.0.0.jar --input ./target/classes/mnist_28x28/3/270.png --endpoint http://localhost:9008/endpoints/mnist/model/mnistmodel/default/

A "3 digit" should return something like:

classification return: {"maxOutcomes":["3"],"rankedOutcomes":[["3","9","8","7","6","5","4","2","1","0"]],"probabilities":[[1,0,0,0,0,0,0,0,0,0]]}