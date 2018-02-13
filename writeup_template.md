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

---
![Banner][image9]
## Overview
In this project, we will use traditional and convolutional neural networks to clone driving behavior. We will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

This project was done as part of Udacity's Self-Driving Car Nanodegree Program. The model performance has been tested on for resolution of 640x480, and graphic quality selected as 'fantastic'.

## Model performance on track 1

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Files Submitted & Code Quality

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

### Model Architecture and Training Strategy

#### 1. Network architecture *is modified from* [NVIDIA CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) is used which consists of 9 layer, including 
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

#### 2. Attempts to reduce overfitting in the model
- Two Dropout layers
- Early stopping
- Data Augmentation by : 
a) ** Using Left and Right Images with correction **  
b) **Image Fliping**                    
 :-------------------------:
![Image Fliping][image4]               

#### 3. Model parameter tuning

 
| Hyperparameter         	|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate        			| Used Adam optimizer(learning rate was not tuned manually) 									| 
| Batch Size         			| 32  									| 
| Epoch     				| 5										|
| Training Data Percentage					| 80%								|
| Validation Data Percentage	      			| 20%					 				|
| Steering Correction for Left and Right Cameras				    | 0.2      							|
| Dropout Prob				    | 0.5      							|

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road 
 
| Track1         	|     	        					| 
|:---------------------:|:---------------------------------------------:| 
| Forward Lap        			| 3						| 
| Backward Lap         			| 1  									| 
| Recovery Areas     				| 5										|

### Model Architecture and Training Strategy

#### 1. Different Approaches tried

- Convolution neural network model similar to the Alexnet with last layer as fully connected layer with one unit. 
- Convolution neural network model similar to comma.ai ([github link (https://github.com/commaai/research/blob/master/train_steering_model.py)).
- [NVIDIA CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

Initially because of the small dataset, the model was memorizing and therefore there was overfitting. But, after suffucient data was collected the validation dataset became better.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added the recovery data to comeback from situations where the car is drifitng on the side.

![TrainingVsValidation][image6]


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

- ** Final Approach ** - Modified  [NVIDIA CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
