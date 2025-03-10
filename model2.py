"""DOCSTRING
Sigma Xi project
created 27/2/2025
Made by Aufy Mulyadi

SUMMARY:

  analyzes the data from the meteors and finds a polynomial regression of it / predicts the outcomes.
  i used sklearn instead of pytorch cuz it keeps bugging out whenever i tried to fetch the
  output prediction.

ARGS:

  prediction (numpy array) : the result / prediction, it also predicts the time although you can disregard it. And to convert it into a list, use the .flatten() method as in the example used below.
  INPUT (float) : the time input for prediction
  DATASET_FILEPATH (string) : the file path of the dataset (use a raw-string literal to avoid accedental exit characters)
  DATA_SIZE (int) : the number of items in the dataset you want to process, this is an option just in case you or i want to quickly add new features without comuting lots of data

NOTES:

  IMPORTANT: the dataset must be in the format of a csv file with the following columns:
    - time
    - range
    - height
    - vel
    - lat
    - long
  
    ill send you a newer python script that will automatically do it for you. 

  I've been trying to brute-force the NN to do its task but its not gonna be possible to train an AI to get 5 features from 1 target, 
  less we average each feature individually and map it to a target classifier (or automate it, it dosent matter). by then its less
  AI and more if-else statements

  i did get the SkLearn to do its task using linear regression since its basically similar to a support vector machine, 
  but i dont know if thats what you are looking for. since i vaugely remember you mentioning that you wanted a neural network
  to do the task. this IS a backup file anyways, but i can still debelop the pytorch NN if need be, though the deadline is in like
  2 days so i dont know if i can make it in time.
  
"""

# imports
import pandas
import numpy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



# important stuff
INPUT = 7.0 # Time input for prediction
DATASET_FILEPATH = r'/Users/71635/Documents/Computer Science/Sigma Xi/Ml model Meteor Showers/output_meteor_shower.csv' # REPLACE THIS WITH YOUR DATASET FILE PATH

DATA_SIZE = 1216446 # how much of the data to process

# extract data
dataframe = pandas.read_csv(DATASET_FILEPATH, dtype=str, low_memory=False)

# Convert all data to numeric, replacing invalid values with NaN
dataframe = dataframe.apply(pandas.to_numeric, errors='coerce')

# Remove NaN values
dataframe = dataframe.dropna()


# COMPUTATIONALLY EXPENSIVE FOR BIGGER DATA
train_feature = dataframe.iloc[:DATA_SIZE, 0].values.reshape(-1, 1) # Time column as feature (input)
train_target = dataframe.iloc[:DATA_SIZE, 1:6].values # everything else as targets (outputs)

# data fitting - (Sklearn jargan you dont need to worry about)
poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly = poly.fit_transform(train_feature)

# linear regression model
model = LinearRegression()
model.fit(train_poly, train_target)

# transform the time input so that the AI understands it
input_poly = poly.transform(numpy.array([[INPUT]]))
prediction = model.predict(input_poly)
actual_results = [i if i[0] == INPUT else None for i in dataframe.iloc[100:200]]

# Print the results - this is debug and can be deleted if need be
print("\n")
print(f"Time : {INPUT:4}")
print("")
print(f"prediction:")
print(" ---- Range ------- Height ---- Velocity ---- Latitude ---- Longitude")
print("")
print(f"{prediction.flatten()} Predicted (Linear Regression)")
print("")
print("------------------------------------------------------------------------------")
print("")
print(f"Filtered Train Target (Matching Time {INPUT}):")
print("")
filtered_train_target = train_target[train_feature.flatten() == INPUT]
print(filtered_train_target)
print("------------------------------------------------------------------------------")
print("")
print(f" Actual (From dataset):")
print(f"{train_target}")
print("")

def mean_squared_error(true, pred):
    true  = numpy.array(true)
    pred  = numpy.array(pred)
    squared_errors = (true-pred)**2
    return numpy.mean(squared_errors)

print("")
print("------------------------------------------------------------------------------")
sum = 0

for i in range(len(filtered_train_target)):
  result = mean_squared_error(filtered_train_target[i], prediction)
  #print(result)
  sum+=result
print(len(filtered_train_target))
average = sum/len(filtered_train_target)
print("")
print(f"Average MSE: {average}")
print("")

print("")
print("------------------------------------------------------------------------------")
print("")



#//ACTUAL VS. AVERAGE PREDICTED GRAPH//
all_predictions = model.predict(train_poly)

# Calculate the average predicted value for each time point
unique_times = numpy.unique(train_feature)  # Unique time points
avg_predicted = []
avg_actual = []

for time in unique_times:
    # Get indices where the time matches
    indices = numpy.where(train_feature.flatten() == time)[0]

    # Average actual and predicted values for that time
    avg_predicted.append(numpy.mean(all_predictions[indices], axis=0))
    avg_actual.append(numpy.mean(train_target[indices], axis=0))

# Convert to arrays for plotting
avg_predicted = numpy.array(avg_predicted)
avg_actual = numpy.array(avg_actual)

# Plot average actual vs. average predicted values for each parameter
labels = ["Range", "Height", "Velocity", "Latitude", "Longitude"]
plt.figure(figsize=(12, 6))

for i in range(5): 
    plt.subplot(2, 3, i + 1)
    plt.xticks(numpy.arange(min(unique_times)+1, max(unique_times) + 1, 2))
    plt.plot(unique_times, avg_actual[:, i], label="Actual", color="blue", marker="o")
    plt.plot(unique_times, avg_predicted[:, i], label="Predicted (Avg)", color="red", marker="x")
    plt.xlabel("Time")
    plt.ylabel(labels[i])
    plt.legend()
    plt.title(f"Avg. Actual vs Predicted {labels[i]}")

plt.tight_layout()
plt.show()




