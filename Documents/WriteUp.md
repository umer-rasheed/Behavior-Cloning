#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavior-Cloning.ipynb containing the script to load, preprocess data, and also create and train the model
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json containing a trained convolution neural network 
* model.h5 containing a trained convolution neural network 
* WriteUp.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py Model.json "Run1" 
```
Please note that "Run1" is the name of the directory to save images of the drive. User can set any other URL in inverted commas ans the code should work. 

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network (model.py line 97 onwards, also 25th code cell of .ipynb file). The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 2 and 16. The model summary can be seen by running Behavior-Cloning.ipynb file

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (See 24th-32nd code cell). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 130, 154). The model summary can be seen by running Behavior-Cloning.ipynb file

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 16, 78, 16)        448       
_________________________________________________________________
activation_1 (Activation)    (None, 16, 78, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 76, 8)         1160      
_________________________________________________________________
activation_2 (Activation)    (None, 14, 76, 8)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 74, 4)         292       
_________________________________________________________________
activation_3 (Activation)    (None, 12, 74, 4)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 72, 2)         74        
_________________________________________________________________
activation_4 (Activation)    (None, 10, 72, 2)         0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 36, 2)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 36, 2)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 360)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                5776      
_________________________________________________________________
activation_5 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 16)                272       
_________________________________________________________________
activation_6 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272       
_________________________________________________________________
activation_7 (Activation)    (None, 16)                0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 17        
=================================================================
Total params: 8,311
Trainable params: 8,311
Non-trainable params: 0

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Unfortunately, my training on the data provided was not as successful. I realized that it was very difficult for the car to recover in case it moved away from the center. Therefore, I collected 2 rounds of data. In the first round, I drove in the center of the lane as accurately as possible. In the second round, I collected data while recovering from the edge of the road. Actual size of data is 7257. However, images were flipped to create additional data after reviewing the steering angle histogram (See 5th code cell in .ipynb file).
Hence, the total data contains 14514 samples. The training, test and validation data is divided as follows. 
Training Data Size: 9796
Test Data Size: 1452
Validation Data Size: 3266
Please note that the data is shuffled and randomized as shown in 18th code cell in .ipynb file.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop a CNN based model 

My first step was to use a convolution neural network model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I collected more training data which focused on recovering from the side of the road. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 16, 78, 16)        448       
_________________________________________________________________
activation_1 (Activation)    (None, 16, 78, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 76, 8)         1160      
_________________________________________________________________
activation_2 (Activation)    (None, 14, 76, 8)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 74, 4)         292       
_________________________________________________________________
activation_3 (Activation)    (None, 12, 74, 4)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 72, 2)         74        
_________________________________________________________________
activation_4 (Activation)    (None, 10, 72, 2)         0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 36, 2)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 36, 2)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 360)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                5776      
_________________________________________________________________
activation_5 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 16)                272       
_________________________________________________________________
activation_6 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272       
_________________________________________________________________
activation_7 (Activation)    (None, 16)                0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 17        
=================================================================
Total params: 8,311
Trainable params: 8,311
Non-trainable params: 0


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recoverin case it swerves from the center. 
For now, I only collected data from the first track. 

To augment the data sat, I also flipped images and angles thinking that this would make the steering angle data symmetrical. The examples of flipped images are shown in .ipynb file.

The images have a lot of unnecessary features like, grass, trees, rocks etc which act as noise. Hence, I removed some region from the top and some region from below since it captured the bonnet of the car. After that, I tested my model with various downsample factors. I downsampled the image size by 4. I initially converted images to grayscale, but that did not yield very good results. I also tested with YUV channels. However, finally I reverted back to RGB. Hence, the shape of the image was (18,80) with 3 channels.


After the collection process, I had 14514 number of data points. 
The training, test and validation data is divided as follows. 
Training Data Size: 9796
Test Data Size: 1452
Validation Data Size: 3266
Please note that the data is shuffled and randomized as shown in 18th code cell in .ipynb file. 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used the number of epochs to be 15, just to make sure the data is properly trained. Although 10 epochs gave good results as well. I used an adam optimizer so that manually training the learning rate wasn't necessary.
