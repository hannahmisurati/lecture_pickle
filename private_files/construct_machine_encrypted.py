import pandas

from sklearn.ensemble import RandomForestClassifier

import pickle

# Load the dataset from a CSV file
dataset = pandas.read_csv("private_dataset.csv")

# Extract target variable and features
target = dataset.iloc[:,30].values
data = dataset.iloc[:,0:30].values

# Initialize Random Forest Classifier with specified parameters
machine = RandomForestClassifier(criterion="gini", max_depth=10, n_estimators=11)

# Train the Random Forest model on the dataset
machine.fit(data, target)

# Save the trained model to a file using pickle
# "wb" mode is for writing in binary format
with open("machine.pickle", "wb") as file:
  pickle.dump(machine, file)


#This code reads a dataset from a CSV file, extracts the target variable and features, 
#and then initializes and trains a Random Forest Classifier on the dataset. 
#The trained model is then saved to a file named "machine.pickle" using the pickle.dump() function 
#in binary format ("wb" mode) for later use.