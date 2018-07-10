# PWC_Net_TensorFlow

Tensorflow implementation of Pyramid, Warping and Cost Volume Networks based on the [paper](https://arxiv.org/abs/1709.02371) presented at CVPR 2018.'\n'
Currently, main.py simply downloads the FlyingChairs Dataset and starts training, following the schedule outlined in the [paper](https://arxiv.org/abs/1709.02371).'\n'
This code could easily be adapted to train on other datasets though.'\n\n'

## Tested Environment

Ubuntu 16.04
Python3
Tensorflow 1.8
Cuda 9.0

## Usage

```python
python3 main.py
```

## Example visualisations following training

From left to right, the images below indicate rgb image, ground truth flow, predicted flow, flow error

Examples from the training set:

![Example Training Flow Result 1](readme_images/example_training_flow1.gif)
![Example Training Flow Result 2](readme_images/example_training_flow2.gif)
![Example Training Flow Result 3](readme_images/example_training_flow3.gif)
![Example Training Flow Result 4](readme_images/example_training_flow4.gif)

Examples from the validation set:

![Example Validation Flow Result 1](readme_images/example_validation_flow1.gif)
![Example Validation Flow Result 2](readme_images/example_validation_flow2.gif)
![Example Validation Flow Result 3](readme_images/example_validation_flow3.gif)
![Example Validation Flow Result 4](readme_images/example_validation_flow4.gif)

## Example Training Loss

This is an example of the loss when training on the full flying chairs dataset (no validation was used on this occassion).

![Example Loss](readme_images/example_loss.png)
