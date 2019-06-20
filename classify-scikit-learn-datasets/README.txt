Implement a fully-connected back-propagation network using TensorFlow
Network with one hidden layer and one output layer is implemneted. 
Data set which presents sample points from different classes is generated using Scikit Learn. Code is in data_generation.py
Displays sample points and the regions for each class with different colors.
Multiple sliders are provided for 
 - Learning rate
 - Weight Regularization
 - Number of nodes in hidden layer
 - Number of data points
 - Number of classes
 
Two button provided
 - Adjust weight (Train): Trains neural network and classify points into regions
 - Reset weights: weights should be reset to random numbers between -0.001 and +0.001.
 
Drop-Down selection box:
 - Transfer function: Relu, Sigmoid. default is Relu.
 - Type of data: s_curve, blob, swiss_roll and moons. default: s_curve
 
Cross entropy with softmax is used as loss function.
When your program starts it automatically creates the input data (with default values), randomize the weights, and display the sample points and class regions (with different colors).
The activation function of the output layer is linear.
Resolution of the displayed output should be 100 by 100
----------------------
Programming Language : 
----------------------
	python version 3.6
	required packages: Scikit Learn, TensorFlow, matplotlib, tkinter, Axes3D, numpy
-------------------
Package Structure :
-------------------
	train_network.py
	main.py
------------------------
Running the Application:
------------------------ 
	1. Run code: python main.py
	2. Adjust all the required parameters through provided User Interface. 
	3. Use "Adjust weight (Train)" button to create and train network, and classify data into regions

# Sardesai, Chaitanya
# 2018-11-26
