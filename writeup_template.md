# **Behavioral Cloning** 


The goals / steps of this project are the following:

1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
  * This model is stored in [model.h5](model.h5).
  * The video of this model at work is in this [movie](drive.mp4).
5. Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center-driving.gif "Center Image"
[image3]: ./examples/center.jpg "Recovery Image"
[image4]: ./examples/left.jpg "Recovery Image"
[image5]: ./examples/right.jpg "Recovery Image"
[image6]: ./examples/unflipped-im.jpg "Normal Image"
[image7]: ./examples/flipped-im.jpg "Flipped Image"
[lossplots]: ./examples/lossplots.png "Losses"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* [generator.py](generator.py) containing the ingestion of data. This assumes:
   * the metadata and labels are stored in a comma separated file `collections_log.csv`, the collection of all the driving logs for each individual run,
   * the CSV file has labels and the paths to the images,
   * iterators to be passed to the training module
* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* [drive.mp4](drive.mp4) for a video of the successful driving log
* [writeup_report.md](writeup_report.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). I thought about a `tanh` activation at the end to bound between -1 and 1, but ended up just using a linear activation (doing nothing) with an MSE loss cost function.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting ([model.py](model.py) line 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. I used Keras's built in training splits with:

```
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Among the issues that I found to be particularly difficult to remedy was if I trained to completion. My neural network just had too many parameters. For that reason, I ensured that I stopped early, where the earliest checkpoint was actually a single epoch.

#### 3. Model validation curves

The model used an Adam optimizer, so the learning rate was set to 0.01 (model.py line 64).

In the Keras modeling effort, and so I needed to implement checkpointing with callbacks. 

```
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('model.ckpt.h5', verbose=1, save_best_only=True)
```

Additionally, in order to produce training/validation plots on a per batch basis below, I required an additional callback. Here, I show the result of the training and validation losses. 

![alt text][lossplots]

This was done by defining `LossHistory` as a callback.

```
# Loss history callback
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In actuality, I had a function called `random_augmentation`, which randomly chose the camera perspective (left,, right, or center). This is probably a matter of semantics, since indeed, data *was* collected on either side of the car, but for my purposes, I simply called a single point in time a sample. This means that through a single epoch, each point in time would only be used once. The code to do so is documented in my `generator.py` code, which is the off-GPU code that I used to prepare the data before handing it to Keras.

Additionally, I drove the car both clockwise and counter-clockwise on the track so that the car effectively sees an entirely new track, which can be thought of as a new dataset.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to research an acceptable MSE-based optimizing deep learning architecture. I looked online for various architectures. In actuality, I wanted to come up with something on my own, though, which I did and is described below.

My first step was to use a convolution neural network model similar to LeNet that I'd previously ran. I thought this model might be appropriate because it offered a considerable amount of complexity. In reality, it seemed as though I'm predicted a single number (whereas LeNet had an output dimensionality of ten), and I'm doing more of a wholistic "classification".

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that additional ReLU's were inputted, and then I added Dropout at all different levels. Then, because the model was still very complex, I did a lot of early stopping (2-3 epochs), which ultimately provided the best results.

For optimization, I used the Adam optimizer. Originally, since I thought that the steering angle was bounded by [-1, 1], I attached a `tanh` function but that turned out not to work out so well, since I required more minute steering values.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected more data around those areas. That didn't seem to help that much, but I figured the extra data wouldn't hurt. 

I did end up cutting down the number of data points in which we were driving straight, since those were relatively boring in terms of learning criteria. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image | 
| Convolution 8x8     	| 4x4 stride, valid padding, 8 output maps |
| Dropout		|		Rate=0.5			|
| RELU					|						|
| Max pooling	      	| 2x2 stride |
| Convolution 4x4	    | 4x4 stride, valid padding, 8 output maps    |
| Dropout		|		Rate=0.5			|
| RELU					|						|
| Max pooling	      	| 2x2 stride |
| Convolution 5x5	    | 1x1 stride, valid padding, 6 output maps    |
| Dropout		|		Rate=0.5			|
| RELU					|						|
| Max pooling	      	| 2x2 stride |
| Fully connected		|  Outputs 128 flat neurons|
| Dropout		|		Rate=0.5			|
| RELU					|						|
| Fully connected		|  Outputs 64 flat neurons|
| RELU					|						|
| Fully connected		|  Outputs 1 flat neurons|
| Tanh			| Nonlinearity between -1 and 1 |
| MSE | Mean squared error |
 

It looks like the number of parameters is adequate and it's currently not overfitting, as the two losses converge to the same point.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to know how to deal with situations in which it needed to recover. These images show what a recovery looks like starting from the right curb:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize the driving to be able to extend to a variety of situations. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 24k number of data points from the original "sample" collection, and additional 44k samples from my own collection. I then preprocessed this data by mean-shifting and variance normalizing the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set, while training on the remaining 80% of the data. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
