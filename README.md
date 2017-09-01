# ml-plants

#### Applying a neural network learning algorithm to classify plant species

This repository aims to provide a neural network that is capable of classifying
plant species based on images of their leaves.

## Neural Network

![Neural Network Architecture](https://docs.google.com/drawings/d/e/2PACX-1vSsRjt3R2eO_xGL4jG_B5N4h98F_dIPTuE5WFtWNDbEQMiKc-7X6V0CcUboyJ1vcgpVQ9SsfRuI7uRZ/pub?w=959&h=451)

> NOTE: Each leaf image is of some dimension L x W, where L = W or L != W. By slicing each images into rows, and appending
> each row to the one above it we form a 1 x (L * W) matrix to pass into the Input layer of our neural network. For example a 10 x 10 > pixel leaf image will be converted to a 1 x 100 matrix.

| Layer | Size |
| --- | --- |
| Input | L * W |
| Hidden-1 | [(L * W) + (C)] / 2 |
| Output | C |

Some helpful notation for describing the neural network architecture design:

| Symbol | Description |
| --- | --- |
| L | length of a leaf image training example |
| W | width of a leaf image training example |
| C | number of labels or classes for neural network |

## Datasets

We'll use open and free datasets to train our learning algorithms.

So far we have tested on the following datasets:

| Name | Info | Download |
| --- | --- | --- |
| One-hundred plant species leaves data set Data Set | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set) | [100 leaves plant species.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00241/100%20leaves%20plant%20species.zip) |
| Leaf Data Set | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Greenhouse+Gas+Observing+Network) | [leaf.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00288/leaf.zip) |
