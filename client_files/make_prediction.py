import pandas
import pickle


# Load the trained model from the saved file using pickle
# "rb" mode is for reading in binary format
with open("machine.pickle", "rb") as file:
	machine = pickle.load(file)


# Load the new survey data from a CSV file
new_survey = pandas.read_csv("new_survey.csv")

# Convert the data to a numpy array
new_survey = new_survey.values

# Use the trained model to make predictions on the new survey data
predictions = machine.predict(new_survey)

# Print the predictions
print(predictions)