# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./examples/bar.png "Visualization"
[image1.1]: ./examples/low_contrast.png "Low Contrast"
[image2]: ./examples/gray.png "Grayscaling"
[image3]: ./examples/equalized.png "Equalized"
[image3.1]: ./examples/flipped.png "Flipped"
[image4]: ./web-GTS/33.jpeg "Traffic Sign 1"
[image5]: ./web-GTS/22.jpeg "Traffic Sign 2"
[image6]: ./web-GTS/31.jpeg "Traffic Sign 3"
[image7]: ./web-GTS/12.jpeg "Traffic Sign 4"
[image8]: ./web-GTS/28.jpeg "Traffic Sign 5"
[image9]: ./web-GTS/13.jpeg "Traffic Sign 6"
[image10]: ./web-GTS/27.jpeg "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the traffic sign classes is distributed in training set.

![alt text][image1]

As from above visualization, we can clearly state that the data is highly unbalanced which can make our model more biased towards the classes with more number of images e.g. class 40, 13 etc.

![alt text][image1.1]

From above fig, we can see that in dataset there are images which have very low contrast.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduce the complexity of the model, and also color doesn't give that much of information in traffic sign images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because as we know neural networks learn its weights by adding gradient error vectors (multiplied by learning rate) computed from backpropagation to various weight matrices throughout the network as training examples are passed through.

If we didn't scale our feature then range of distribution of features values for each feature will likely to be very different. So, weight compensating for one feature may cause undercompensating the other. This can make our loss function in oscilliating state. 

I decided to generate additional data because of unbalanced data. As we visualized earlier some of the classes has less than 400 images, and some have more than 1500 which can make our model biased.

To add more data to the the data set, I used the techniques resampling and flipping because some of the images are same after the flipping which can be benficial for us.

Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image3.1]

The difference between the original data set and the augmented data set is the following

In original training dataset, we have 32799 image but after data augmentation we has more than 107500 images in our dataset i.e. 20 times of the original dataset.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 36x36x1 GRAY image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x20 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 12x12x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x80 				    |
| Fully connected		| outputs 512        						    |
| RELU  				|           									|
| Dropout				| Keep = 0.5									|
| Fully connected		| outputs 128        						    |
| RELU  				|           									|
| Dropout				| Keep = 0.5									|
| Fully connected		| outputs 43        						    |
| Softmax  				|           									| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**Optimizer:** 
To train the model, I used Adam optimizer s it converges fast, and the learning speed of the model is quiet fast and efficient. It also rectifies the problem of other learning algorithms like vanishing learning rate, slow convergence, and High variance in the parameters updates.

**Batch Size and number of epochs:**
Generally the batch size of 32, 64, 128, 256, 512, and 1024 (i.e. we usually choose batch size to be the power of 2) are used in neural networks. After checking on different batch sizes and number of epochs, I came up with these findings

| Batch Size         |     Epochs    	|      Accuracy     |
|:------------------:|:----------------:|------------------:| 
| 32                 | 10               |  95.6%            |
| 64                 | 10               |  95.3%            |   
| 128				 | 10				|  95.4%            |
| 256       	     | 10   			|  93.6%            |
| 512           	 | 10         		|  90.6%            |
| 1024               | 10        		|  85%              |

As from the above findings we can see that, batch size of 128 and 32 with 10 number of episode gives me the best result. I choose the batch size of 128 because of fewer operations which will increase my training speed.

**Learning Rate:**
I used the learning rate of 0.0005. As it works better for my model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8
* validation set accuracy of 95
* test set accuracy of 93.4

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  
  I started with the architecture which has three convolutional layer and three fully connected layer. Reason behind choosing this architecture was that it has more layer so, it will perform well. But things didn't go as it seems. The extra layer overfits the data instead of extracting the feature.

* What were some problems with the initial architecture? 
  
  As in my first architecture, I had three convolutional layer. This architecture performs well on training set but it's validation accuracy was quite poor. May be the extra layer overfits the data instead of extracting the features.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  I adjusted the architecture by removing one layer of the convolution which were cauing the overfitting. I also added the dropout in fully connected layers. Adding the dropout also helps to remove the overfitting.

* Which parameters were tuned? How were they adjusted and why?

  I tuned the number of layers, shape of the output of convolutional layers and fully connected layers. 
  
  I choose two convolutional layers over three because of overfitting, and tried various shapes of the convolutional layer, from which 16x16x20 and 6x6x80 works best for me.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    
  Convolution layer is good design choices for the problems where we have to classify the images. It's because convolutional network extract the small feautes from the images like edges, lines, and them combined them to extract more advance feature like circles, letters, signs etc. As traffic sign classifier is the similar problem which makes convolutional layer best fit for it. 
  
  Dropout layer prevent the overfitting which makes model more general.

If a well known architecture was chosen:
* What architecture was chosen?
  
  I will choose LeNet-5 architecture.

* Why did you believe it would be relevant to the traffic sign application?

  I believe it would be relevant to the traffice sign application because it contain two convolutional layer which works well to extract the features from the traffic sign images, and it also contain dropout layer which prevent the overfitting.
  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  
  Model gives the 99.8% accuracy on test set, 95% accuracy on validation set, and 93.3% accuracy on the test images. I think it's good model because it's performing well on the test set too. And the difference b/w the accuracies of the train and validation set is not too high which prove that our model is also not overfitting the data. Having a test accuracy greater than 93% is good enough.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

The third, fifth, and seventh image might be difficult to classify because of the slight tilt towards the left side. Remaining images seems to classify easily as they do not have any noise.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead     	| Turn right ahead          					|
| Bumpy road			| Bumpy road    								|
| Wild animals crossing | Wild animals crossing	    	 				|
| Priority road         | Priority road                	 				|
| Children crossing		| Children crossing 							|
| Yield          		| Yield             							|
| Pedestrians   		| End of no passing by vehicles over 3.5...     |


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 93.4%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is highly sure that this is a Turn right ahead sign (probability of 1.0), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead  							| 
| .00     				| Ahead only									|
| .00					| Keep Left 									|
| .00	      			| No entry  					 				|
| .00				    | Yield               							|


For the second image, the model is highly sure that this is a Bumpy road sign (probability of 1.0), and the image does contain a Bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road        							| 
| .00     				| Traffic signals								|
| .00					| Bicycles crossing								|
| .00	      			| Dangerous curve to the left 	 				|
| .00				    | Beware of ice/snow        					|

For the third image, the model is highly sure that this is a Wild animals crossing sign (probability of 1.0), and the image does contain a Wild animals crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Wild animals crossing							| 
| .00     				| Double curve									|
| .00					| Slippery road									|
| .00	      			| Beware of ice/snow			 				|
| .00				    | Road work            							|

For the fourth image, the model is highly sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority road       							| 
| .00     				| Roundabout mandatory  						|
| .00					| No entry  									|
| .00	      			| End of all speed and passing limits			|
| .00				    | Keep Left            							|

For the fifth image, the model is highly sure that this is a Children crossing sign (probability of 0.9543), and the image does contain a Children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9543       			| Children crossing 							| 
| .0184   				| Beware of ice/snow							|
| .0120 				| Right-of-way at the next intersection			|
| .071	      			| End of speed limit (80km/h)    				|
| .016				    | Dangerous curve to the right					|

For the sixth image, the model is highly sure that this is a Yield sign (probability of 1.0), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield               							| 
| .00     				| Bumpy road									|
| .00					| Ahead only 									|
| .00	      			| No vehicles 					 				|
| .00				    | Keep right          							|

For the seventh image, the model is predicts that this is a End of no passing by vehicles over 3.5 metric tons sign (probability of 0.6771), but the image contains the Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .6771        			| End of no passing by vehicles over 3.5 metr.. | 
| .2937    				| Right-of-way at the next intersection			|
| .0251  				| Roundabout mandatory  						|
| .0039	      			| No passing for vehicles over 3.5 metric tons  |
| .00				    | Ahead only           							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


