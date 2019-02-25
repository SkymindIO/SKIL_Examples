# SKIL Anomaly Detection Application

Please [click here](https://blog.skymind.ai/multi-tier-system-for-detecting-fraud-with-ai-and-deep-learning/) to view the full tutorial.

## Files

This repository contains three files.

1. `Fraud Autoencoder.json` - An importable Zeppelin notebook containing Python code to train an autoencoder.
2. `Fraud MLP.json` - An importable Zeppelin notebook containing Python code to train a multilayer perceptron.
3. `skil_fraud_client_app` - Client API to query the deployed multilayer perceptron model.

## Client API Instructions

`skil_fraud_client_app` contains a client API, written in Java, which retrieves predictions from the multilayer perceptron deployed in SKIL.

1. Run `mvn package` inside the client_app subdirectory to obtain the JAR file.
2. Type `java -jar skil-example-anomaly-detection-1.0.0.jar --feature fraud.csv --endpoint http://localhost:9008/endpoints/test/model/anomaly-detection/default/` to query the model and obtain a prediction. Remember to replace "localhost" with the correct host if necessary.

## Training Dataset

We use the [PaySim dataset](https://www.kaggle.com/ntnu-testimon/paysim1/data). PaySim is a synthetically created dataset of over 6,000,000 mobile money transaction. To reduce the time it takes to train our models, we pluck a randomized subsample of around 100,000 examples from the full PaySim dataset.

## Data Cleansing

In order for our machine learning models to process the data, we need to transform the raw CSV data into usable vector formats. The features of the PaySim dataset contains both categorical and numerical values which must be converted into floats.

| Feature | Transformation |
| ----- |:--------------------:|
|  *step, isFraud, isFlaggedFraud* | Convert from integers to floats |
|  *type, nameOrig, nameDest* | Convert from strings to categorical values to floats |
|  *amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest* | Already floats. No conversion necessary. |

Now that our data is properly transformed, we scale all values between 0 and 1 to suppress the impact of outliers.

## Methodology

### Multilayer Perceptron (MLP)

An MLP is a *supervised deep learning model* intended to perform binary classification such as `fraud` or `not fraud`. Because we have a relatively large labeled dataset, we can teach an MLP to learn the differences between a `known` fraudulent and a `known` normal transaction. A supervised approach tends to perform very well on well-established and known items of interest.

### Autoencoder

An autoencoder is an *unsupervised deep learning model* which can learn patterns within data without reference to known, or labeled, outcomes. This is valuable because we can teach an autoencoder to measure differences between normal and fraudulent transactions.

Autoencoders are quite useful for two reasons:

1. A labeled dataset is not required.
2. Autoencoders output an evaluation metric called reconstruction error. This metric is an objective baseline in which we can measure how far a transaction deviates from normal activity and thus, determine if it requires further scrutiny.

An unsupervised approach tends to perform much worse, but it is very good at catching novel items (unknown unknowns) which are near impossible to proactively detect.
