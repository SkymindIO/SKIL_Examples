# SKIL_Examples
Public examples for the SKIL Platform (client examples, notebooks, etc). Versioned for each release.

## Examples Included in the v1.0.1 Release

* Endpoint Examples
* MNIST Keras Notebook (python) and Java Client Example

## Running the MNIST Keras Notebook and Model Server Client

Includes a full SKIL application example with a JSON Keras python notebook and an example client in java that base64 encodes MNIST images and sends them to a SKIL Model Server endpoint for classification, returning labels and probabilities.

### Running Model Server Client Code

(from the client_app subdir)

To send an image for classification to the SKIL model server with REST we'll use the included client example that is executed from the command line in the form of:

java -jar ./target/skil-example-mnist-tf-1.0.0.jar --input [image file location] --endpoint [skil endpoint URI]

To send a blank image to the model server to test out a non-MNIST image:

java -jar ./target/skil-example-mnist-tf-1.0.0.jar --input blank --endpoint http://localhost:9008/endpoints/mnist/model/mnistmodel/default/

This should return (along with some other log debug lines):

classification return: {"maxOutcomes":["5"],"rankedOutcomes":[["5","8","3","9","2","7","4","1","6","0"]],"probabilities":[[0.12396843731403351,0.1205655038356781,0.10955488681793213,0.10501493513584137,0.09488383680582047,0.09272368997335434,0.09050352871417999,0.08859185874462128,0.08754267543554306,0.08665058016777039]]}

Where we can see the 5 digit got the highest classification probability of 0.12396843731403351, but this was not much better than the 8-digit with a probability of 0.1205655038356781. Effectively this model returned probabilities for each digit that were fairly close in value, so we know the model recognized this image didn't resemble any of labels well-enough that it was trained on.

Now let's send a real image from MNIST to the model server; A few MNIST digit images (28x28 pngs) are included in the example, and we can send those for classification as well:

java -jar ./target/skil-example-mnist-tf-1.0.0.jar --input ./target/classes/mnist_28x28/3/270.png --endpoint http://localhost:9008/endpoints/mnist/model/mnistmodel/default/

A "3 digit" should return something like:

classification return: {"maxOutcomes":["3"],"rankedOutcomes":[["3","9","8","7","6","5","4","2","1","0"]],"probabilities":[[1,0,0,0,0,0,0,0,0,0]]}