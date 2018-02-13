# **Behavioral Cloning** 
### by Mohit Arvind Khakharia
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./write_up_images/nvidia_cnn.png "Nvidia Model"
[image3]: ./write_up_images/data_vs_steering.png "Steering Angle Distribution Image"
[image4]: ./write_up_images/flipped_img.png "Flipped Image"
[image5]: ./write_up_images/origina_vs_cropped.png "Original vs Cropped Image"
[image6]: ./write_up_images/training_vs_validation.png "Training vs Validation Image"
[image7]: ./write_up_images/m-summary.png "Model Summary"
[image8]: ./write_up_images/flipped_img.png "Moving Track1"
[image9]: ./write_up_images/bc.png "Banner"
[image10]: ./write_up_images/autonomous_1.gif "autonomous_1"
[image11]: ./write_up_images/autonomous_2.gif "autonomous_2"

---
![Banner][image9]
## Overview
In this project, we will use traditional and convolutional neural networks to clone driving behavior. We will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

This project was done as part of Udacity's Self-Driving Car Nanodegree Program. The model performance has been tested on for resolution of 640x480, and graphic quality selected as 'fantastic'.

## Model performance on track 1

 **DRIVER CAM**                     |  **AERIAL CAM** 
 :-------------------------:|:-------------------------:
 ![DriverCam][image10] |  ![AerialCam][image11]
---

## Goals

* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (also in README.md) summarizing the results
* view_data.ipynb showing various aspects of the data.

#### 2. Functional code
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Usability and Readability

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Attempts to reduce overfitting in the model
- Two Dropout layers
- Early stopping
- Data Augmentation by :   
a) **Using Left and Right Images with correction**  
b) **Image Fliping**                    
 :-------------------------:
![Image Fliping][image4]               

## Model parameter tuning

 
| Hyperparameter         	|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate        			| Used Adam optimizer(learning rate was not tuned manually) 									| 
| Batch Size         			| 32  									| 
| Epoch     				| 5										|
| Training Data Percentage					| 80%								|
| Validation Data Percentage	      			| 20%					 				|
| Steering Correction for Left and Right Cameras				    | 0.2      							|
| Dropout Prob				    | 0.5      							|

## Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road 
 
| Track1         	|     	        					| 
|:---------------------:|:---------------------------------------------:| 
| Forward Lap        			| 3						| 
| Backward Lap         			| 1  									| 
| Recovery Areas     				| 5										|

#### Note - The car was driven using the mouse so that we get the maximum amount of non-zero steering angle frames. The distribution of the steering angle in the training data is shown below.
 
 **Frame count vs Steering Angle**                    
 :-------------------------:
 ![Steering Angle Distribution][image3]
 
 ## Preprocessing
 - Image Normalization : The images are normalized so that the computations are neither too big or small.
 - Image Cropping : The sky and the car dome dont affect the steering angle and so they were cropped.

## Model Architecture and Training Strategy

#### 1. Different Approaches tried

- Convolution neural network model similar to the Alexnet with last layer as fully connected layer with one unit. 
- Convolution neural network model similar to comma.ai ([github link (https://github.com/commaai/research/blob/master/train_steering_model.py)).
- [NVIDIA CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

#### 2. Final Model Architecture

####  Network architecture *is modified from* [NVIDIA CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) is used which consists of 9 layer, including 
- 1 Normalization layer
- 1 Cropping layer
- 3 convolutional layers with subsampling and Rectified Linear Unit(RELU)
- 2 convolutional layers with only Rectified Linear Unit(RELU)
- 2 Dropout layers for generalization
- 3 fully connected layers.

 **NVIDIA CNN**                    
 :-------------------------:
 ![Nvidia Model][image1]
 
  **Modified NVIDIA CNN**                    
 :-------------------------:
 ![Modified CNN][image7]

## Training Strategy
Initially because of the small dataset, the model was memorizing and therefore there was overfitting. But, after suffucient data was collected the validation dataset became better.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added the recovery data to comeback from situations where the car is drifitng on the side.

![TrainingVsValidation][image6]

```
0400/30347 [==============================] - 34s - loss: 0.0049 - val_loss: 0.0031
Epoch 2/5
30358/30347 [==============================] - 32s - loss: 0.0036 - val_loss: 0.0026
Epoch 3/5
30400/30347 [==============================] - 32s - loss: 0.0037 - val_loss: 0.0033
Epoch 4/5
30358/30347 [==============================] - 32s - loss: 0.0036 - val_loss: 0.0031
Epoch 5/5
30400/30347 [==============================] - 32s - loss: 0.0035 - val_loss: 0.0027
```

| Track 1         	|     	        					| 
|:---------------------:|:---------------------------------------------:| 
| Training Loss        			| 0.0035						| 
| Validation Loss         			| 0.0027  									|

## RESULT
The vehicle is able to drive autonomously around the track without leaving the road.

## Future Improvements
- More augmented data using brightness, hue and scaled images so that the CNN learns to identify the road curvature under different lighting conditions.
- Add different types of shadows onto the image frames so that the detection is shadow agnostic.
- Use [CAPSULE NETWORKS](https://github.com/naturomics/CapsNet-Tensorflow) so that the detection can tackle pixel attacks and maintain the positional constraints(E.g - An image of a road in the Advertisement banner on the side of the track should not effect the decision making process)

## Model performance on Track 1

 **DRIVER CAM**                     |  **AERIAL CAM** 
 :-------------------------:|:-------------------------:
 ![DriverCam][image10] |  ![AerialCam][image11]
