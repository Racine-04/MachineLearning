from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import numpy as np

# Load the Iris dataset
dataset = load_dataset("scikit-learn/iris")

# Convert dataset to a Pandas DataFrame
df = pd.DataFrame(dataset["train"])
print(df.head()) # Display first few rows

# Remove the "Id" column as it is not useful for classification
df = df.drop("Id", axis=1)

# Convert categorical labels into numerical format
# This converts it into a binary classification problem: 
# 0 = Iris-setosa, 1 = Non-Iris-setosa (Iris-versicolor & Iris-virginica)
df["Species"] = df["Species"].replace("Iris-setosa", 0)
df["Species"] = df["Species"].replace("Iris-versicolor", 1)
df["Species"] = df["Species"].replace("Iris-virginica", 1)


# Split data into training (80%) and test (20%) sets
x_data = np.stack(df.drop("Species", axis=1).values)
y_data = np.stack(df["Species"].values)

# Split into train test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=2)

# Initialize LazyPredict's AutoML classifier
classifier = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)

# Fit the AutoML classifier and evaluate performance
models,predictions = classifier.fit(x_train, x_test, y_train, y_test)

# Print model performance metrics
print(models)