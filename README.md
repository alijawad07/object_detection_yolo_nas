# Object Detection using YOLO-NAS
This repository contains scripts for training and testing an object detection model using YOLO-NAS architecture.

## Getting Started
- python 3.x
- super-gradients

## Installation

```
pip install super-gradients==3.1.0
pip install imutils
pip install opencv-python
```
## Dataset
Prepare your dataset by creating **train**, **test**, and **valid** folders in a root folder named dataset. Inside each folder, create a folder for each class, containing images of that class. In addition, create a dataset.yaml file in the root directory to define the dataset structure. An example **dataset.yaml** file is included in this repository.

## Training
- Run the following command to start training:
```
python3 train.py --data --name --epochs --batch_size --num_workers
```
The trained model will be saved in the **'checkpoints'** directory.

**--data**      => .yaml file with dataset and class details

**--name**   => the name of the directory you want to create to store checkpoints

**--epochs** => number of epochs. by default it is 100

**--batch_size** => size of batch, by default it is 4

**--num_workers** => number of workers, by default it is 2

## Testing

- Run the following command to start testing
```
python3 test.py --data --source --output --weights --conf
```
**--data**  => .yaml file with dataset and class details

**--source** => Path to directory containing images

**--output** => Path to save the detection results

**--weights** => Path to checkpoint file

**--conf** => Confidence threshold for bounding boxes

**--video** => Flag indicating whether the source directory contains videos instead of images

This will run the model on the test set and save the results in the runs/detect directory.

## Acknowledgements
Huge shoutout to [Nicolai Høirup Nielsen](https://github.com/niconielsen32) and [Piotr Skalski](https://github.com/SkalskiP) for inspirtation for giving a starting point.

