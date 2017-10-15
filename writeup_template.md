#**Behavioral Cloning** 


The goals / steps of this project are the following:

1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
5. Summarize the results with a written report

## Installation Notes


I decided to try to install the Docker container via the instructions from CarND-Term1-Starter-Kit this time for educational and informational reasons. This turned out to be a nontrivial effort, and things didn't work right out of the box. For anyone who's interested in learning about how to do it, here are my trials and tribulations.

The basic problem is that cuDNN v5.1 is not installed as a library. Because of that, you're likely to run into something like:

`
tensorflow/stream_executor/cuda/cuda_dnn.cc:221] Check failed: s.ok() could not find cudnnCreate in cudnn DSO; dlerror: /root/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow.so: undefined symbol: cudnnCreate Aborted
`

A quick note about what *didn't work*. Scouring through online forums, I came across the note to replace the first line (where the container inherits from) that currently looks like:

```
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
```

to

```
FROM gcr.io/tensorflow/tensorflow.
```

Since we're using NVIDIA's cuDNN v5.1, I would have thought.

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

* generator.py containing the ingestion of data. This assumes:
   * the metadata and labels are stored in a comma separated file `filename.csv`,
   * the CSV file has labels and the paths to the images,
   * iterators to be passed to the training module
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In actuality, I had a function called `random_augmentation`, which randomly chose the camera perspective (left,, right, or center). This is probably a matter of semantics, since indeed, data *was* collected on either side of the car, but for my purposes, I simply called a single point in time a sample. This means that through a single epoch, each point in time would only be used once. The code to do so is documented in my `generator.py` code, which is the off-GPU code that I used to prepare the data before handing it to Keras.

Additionally, I drove the car both clockwise and counter-clockwise on the track so that the car effectively sees an entirely new track, which can be thought of as a new dataset.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

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

After the collection process, I had 24k number of data points from the original "sample" collection, and additional 44k samples from my own collection. I then preprocessed this data by mean-shifting and variance normalizing the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set, while training on the remaining 80% of the data. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
