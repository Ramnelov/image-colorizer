# Image Colorizer

## Table of Contents

- [Introduction](#introduction)
- [Method](#method)

## Introduction

This is a project to test whether the **U-Net** convolutional neural network can be used to colarize black and white pictures. The network is trained using the images from the **CIFAR-100** dataset.

## Method

The model is constructed like a regular U-Net architecure with input dimension $32 \times 32 \times 1$ and output dimension $32 \times 32 \times 3$, i.e. the outout has more channels than the input. This basically means that the output needs to learn to extract extra information. Down bellow is an example of the network architecture with similar dimensions.

![U-Net Architecture](./u-net.png)

The model is trained using the mean squared error

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

between the regular image as the target and a corresponding black and white image feeded into the network as the prediction. After each epoch the average validation loss is computed to evaluate the model over time.

The final model is evaluated by plotting targets and predictions side by side and comparing similarity.
