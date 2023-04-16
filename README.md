# Data_Stonks
Multi-Layered Perceptron model using sklearn
I have used sklearn library and with it the MLPClassifier in the sklearn package

There are two hidden layers in the neural network with 8 nodes in each 
The number of epochs is set to 300 (but can be changed depending on resources)
The ReLU activation is used in the hidden layer along with softmax in the final layer

The loss function is calculated using cross entropy loss

The solver of choice used here is 'adam'

Using an 80-20 split in the training data, an accuracy of around 94.5% was reached using a runtime of approximately 5 seconds
The test cases have been solved for using that very model and the predictions are stored in the csv file of the same name

