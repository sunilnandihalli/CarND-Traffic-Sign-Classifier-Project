#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/00000.ppm "16 - Vehicles over 3.5 metric tons prohibited"
[image5]: ./examples/00001.ppm " 1 -  Speed limit (30km/h)"
[image6]: ./examples/00002.ppm "38 - Keep right"
[image7]: ./examples/00003.ppm "33 - Turn right ahead"
[image8]: ./examples/00004.ppm "11 - Right-of-way at the next intersection"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sunilnandihalli/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I didn't preprocess all the data ahead of time. I included preprocessing as part of the network itself. This would increase the compute but just seemed simpler. The preprocessing steps I took the image converted to both hsv and grascale images. I then concatenated them so that the number of channels is now 4. While I don't have a strong justification for this, it just seemed to make sense. Also, in order to augment the dataset with images, I added noise gaussian(mean=0.0,std=0.05) noise to 30% of input in everybatch.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:



| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
|concatenate hsv and grayscale images | outputs 32x32x4 |
| apply per image standardization | no change in output-shape |
| Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x6 |
| RELU | |
| Convolution 5x5 | 1x1 stride, valid padding, output 28x28x6 |
| RELU | |
| max pooling | 2x2 stride | output 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, output 10x10x16 |
| RELU | |
| max pooling | 2x2 stride | output 5x5x16 |
| Flatten |  | input:5x5x16 output 400 |
| Fully connected |  input:400 output:120 |
| RELU | |
| Fully connected |  input:120 output:120 |
| RELU | |
| Dropout | keep_prob : 0.5(training) 1.0(evaluation) | 
| Fully connected | input:120 output:84 |
| RELU | |
| Dropout | keep_prob : 0.5(training) 1.0(evaluation) |
| Fully connected | input:84 output:43 |
| RELU | |
| Dropout | keep_prob : 0.5(training) 1.0(evaluation) |
| Fully connected | input:43 output 43 |
| Softmax				|        									|
 
This is simply a LeNet model with a few added layers.  The layers were added purely for experimental reasons.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using an adamoptimizer with a batch size of 512 for 90 epochs. I used a learning rate of 0.001 and a dropout regularization of 0.5 during training. I prefered to use dropout over l2 regularization as scaling the regularization-loss would be a non-issue in case of dropout regularization.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used LeNet architecture as the starting point for this problem. LeNet was used as this was also designed for image classification. The model took about 10 minutes to train on a GPU. 

My final model results were:
* training set accuracy of 99.3 %
* validation set accuracy of 93.8 %
* test set accuracy of 92.5 %


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited      		| Vehicles over 3.5 metric tons prohibited   									| 
| Speed limit (30km/h)     			| Yield										|
| Keep right					| Keep right											|
| Turn right ahead	      		|  Stop					 				|
| Right-of-way at the next intersection		| Right-of-way at the next intersection |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .78         			| Vehicles over 3.5 metric tons prohibited  									| 
| .21     				| End of no passing 										|
| .003					| No passing											|
| .0006	      			| End of no passing by vehicles over 3.5 metric tons			 				|
| .0001			    | Speed limit (80km/h)      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .68         			| Yield   									| 
| .24     				| Speed limit (50Km/h) 										|
| .04					| Speed limit (30Km/h)											|
| .01	      			| No Vehicles					 				|
| .006			    | Priority Road    							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep Right   									| 
| 0.0    				| General Caution 										|
| 0.0					|  Ahead only								|
| 0.0	      			| End of no passing					 				|
| 0.0			    | Go Straight or right    							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			|   Stop 									| 
| .23     				| Turn Left ahead 										|
| .13					| No entry								|
| .10	      			| Yield				 				|
| .045			    | Turn right ahead     							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection									| 
| 0     				| Beware of ice/snow							|
| 0					| Speed limit (20Km/h)											|
| 0	      			| Speed limit(30Km/h)					 				|
| 0			    | Speed limit (50Km/h)						|



From the above tables, we can see that the model is quiet uncertain for the examples which it did not 
predict correctly. For the ones that it did predict correctly, it seems quiet certain.



