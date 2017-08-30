# ml-plants

#### Applying a neural network learning algorithm to classify plant species

This repository aims to provide a neural network that is capable of classifying
plant species based on images of their leaves.

## Datasets

We'll use open and free datasets to train our learning algorithms.

So far we have tested on the following datasets:

| Name | Info | Download |
| --- | --- | --- |
| One-hundred plant species leaves data set | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set) | [100 leaves plant species.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00241/100%20leaves%20plant%20species.zip) |

## Learned Contexts

| Hidden Layers | Neurons | Lambda | Context | Test Accuracy |
| --- | --- | --- | --- | --- |
| 1 | 50 | 0.02 | [learnedContext-50.mat](./100-leaves-plant-species/learnedContext-50.mat) | 84.667 |
| 1 | 82 | 0.01 | [learnedContext.mat](./100-leaves-plant-species/learnedContext.mat) | 80.667 |

These are the learned contexts .mat files that you can load in Octave to spin
up a neural network with the learned weights like so:

```
octave:1> cd ml-plants/neural-network
octave:1> load('../100-leaves-plant-species/learnedContext.mat');
octave:1> pred = predict(Theta1, Theta2, Xtest);
octave:1> % Test Accuracy
octave:1> mean(double(pred == ytest)) * 100
```

> Note: You can substitue Xtest and ytest for your own dataset to validate your
> neural network.
> Here, Xtest and ytest are the part of the training set X that
> the neural network hasn't seen yet. So as long as your data matches the format
> and scale as Xtest and ytest, it should be able to make predictions.
