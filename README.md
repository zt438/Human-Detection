# Human Detection

Detection of human in images implemented with Python 3.

Uses HOG (Histograms of Oriented Gradients) and LBP (Local Binary Pattern) features to detect human in images)

The image features are passed into a two-layer feed-forward neural network in order to be classified as either to contain human or not contain human.

The ReLU activation function is applied for the neurons in the hidden layer and Sigmoid function is applied for the output layer. 

The program stops traiing when one of the following conditions is met:
* The average error is less than or equal to 0.005
* Number of training epochs is greater than or equal to 1000


