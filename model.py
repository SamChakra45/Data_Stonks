# Step 1
# Importing the necessary libraries
import numpy as np
from sklearn.utils import Bunch
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score 
import csv




# Step 2
# Loading the dataset
def load_my_fancy_dataset():
    with open(r'train.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            label = row[0]
            data.append(row[1:])
            target.append(int(label))
        
        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)
dataset = load_my_fancy_dataset()

# Step 3
# Splitting the data into tst and train
# 80 - 20 Split
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, random_state=4)

x_test=np.array(x_test,dtype=int)



# Step 4
# Making the Neural Network Classifier
NN = MLPClassifier(hidden_layer_sizes=(8,8),max_iter=300,activation='relu',solver='adam',random_state=1)

# Step 5
# Training the model on the training data and labels
NN.fit(x_train, y_train)

# Step 6
# Testing the model i.e. predicting the labels of the test data.
y_pred = NN.predict(x_test)

# Step 7
# Evaluating the results of the model
accuracy = accuracy_score(y_test,y_pred)*100


# Step 8
# Printing the Results
print("Accuracy for Neural Network is:",accuracy)

test = []
with open('test.csv', mode='r')as file:
    csvFile = csv.reader(file)
    flag=False
    for lines in csvFile:
        if flag==True:
            del lines[0]
            test.append(lines)
        else :
            flag=True
 
test= np.array(test,dtype=int)
results = NN.predict(test)

with open('sampleSubmission.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(['Id','Verdict'])
    for i in range(len(results)):
        # write the data
        writer.writerow([i,results[i]])


