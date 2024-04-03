import pandas

from sklearn.ensemble import RandomForestClassifier

import pickle

#load the dataset
dataset = pandas.read_csv("private_dataset.csv")

#extract target variable and features
target = dataset.iloc[:,30]
data = dataset.iloc[:,0:30].values

#initialize random forest classifier with specified parameters
machine = RandomForestClassifier(criterion = "gini", max_depth = 10, n_estimators = 11)

#fit the randome forest model on the dataset
machine.fit(data, target)

#dave the trained model to a file using pickle
#with cross funtion, w for writing in file, b for binary
with open("machine.pickle", "wb") as file: 
	pickle.dump(machine, file)