#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* augment the data set by adding new images out of the existing ones by transforming them(scale/shift/rotate)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample.png " sample input after grayscale"
[image2]: ./test_web.png "test images from the web"
[image3]: ./softmax.png "softmax probabilities"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


You're reading it! and here is a link to my [project code](https://github.com/localmaxima/sdc_p2/blob/master/Traffic_Sign_Classifier.ipynb)



###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used python to count frequencies of training data set and to calculate summary statistics of the traffic
signs data set:

* The size of training set is 106345
* The size of the validation set is 52380
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It's rendering of all unique classes a long with their frequence and class id.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

After visualizing the data set , I found that some labels have more images than the others (2200 images for the highest one vs 200 images for the lowest one), I used an image transformation lib to randomly generate a transformed image out of the original one and add it to the data set. For each label, I calcualte the number of transformations that should be applied on it based on how far it is from the label with max frequency, for example, if the 20 km speed limit sign has 2000 images and turn right has just 200, for each images in the turn right sign, I makde 10 transformations and add it to the training set. 

I decided to convert the images to grayscale because it's easier to manipulate one channel than the three RGB channels.
After that, I normalized the image data because this way it's easier for weights to be shared accorss different images with different intensities.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set on a 2/1 scale. I did this by using the train_test_split method from sklearn.cross_validation. I kept the test set unused until I became sure the model has learnt enough.

My final test set had 12630 images. My validation set had over 52380 images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscal image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| DROPOUT					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Fully connected		| input 400,  outputs 120								|
| RELU					|												|
| Fully connected		| input 120,  outputs 84								|
| RELU					|									
| Fully connected		| input 84,  outputs 43								| 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used softmax cross entropy to calculate the loss, then the reduce_mean to minimize it. the optimizer is Adam with learning rate 0.001, I tried 0.003 and it gave good results too. A correct prediction is the highest probability cell of a hot_y vector.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
*training set accuracy up to %93
* validation set accuracy of %86
* test set accuracy 91%

Training and validation accuracy were doing ok, The testing accuracy reached 91% which means that the training set could cover a lot of cases in the test test

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? 
I choose the LeNet architecture

* Why did you believe it would be relevant to the traffic sign application? 
Using a combination of convolutions and activations and full connected networks, we can teach the model to learn about most important features of each traffic sign without explicitly telling the model how each sign looks like, i.e without telling it that the number 8 for example is composed of 2 small circles one above the other or the no-entry sign is a big red circle with a white recatangle in the middle.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The model's accuracy on training data is very high and on validation data is less accurate since it's not seeing validation data as much as it sees training data, and of course the accuracy of test data is much lower since it came from a completely different data set.
 
 What I would do next is to sample from the training data using a normal distribution where in the middle of the distribution lies the signs with highest frequencies, and in the both ends, lies the signs with less frequencies, this permits signs with lower frequencies to be seen enough during the training phase.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]
The first image might be difficult to classify because maybe it was brighter than similar class images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook. The accuracy of the images downloaded from the  web (using google street view) is about %79.

Here are the results of the prediction:
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 0.000
Test Accuracy = 1.000
Test Accuracy = 0.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 1.000
Test Accuracy = 0.000

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30 km      		| 100%   									| 
| Speed limit 30 km     			| 100% 										|
| Speed limit 30 km					| 100%										|
| Speed limit 50 km      		| 0% 				 				|
| Speed limit 50 km		| 100%  							|
|Right-of-way at the next intersection      		| 0%   									| 
| Yield     			| 100% 										|
| STOP				| 100%										|
| Vehicles over 3.5 metric tons prohibited     		|100% 				 				|
| No entry		| 100%  							|
| Roadwork      		| 100%   									| 
| Children crossing     			| 100% 										|
| Go straight or right				| 100%										|
| Keep right     		| 0% 				 				|


The model was able to correctly guess 11 of the 14 traffic signs, which gives an accuracy of 79%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

I created some visulaization that will explain how sure was the model of the predicted sign and which were the next probable choices

![alt text][image3]




