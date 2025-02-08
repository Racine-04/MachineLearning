import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from PIL import ImageOps

# Function to clean and preprocess images
def clean_img(img):
    # Resize the image to 40x40 to standardize dimensions
    # Remove some of the border to focus on the digit
    img = ImageOps.fit(img, (40, 40), bleed=0.07)

    # Convert the image to grayscale to reduce complexity
    img = ImageOps.grayscale(img)

    return img


# Load the dataset from HuggingFace
dataset = load_dataset("thoriqtau/Handwritten_Digits_10k")

# Check the structure of the dataset
print(dataset)

# Convert the dataset into a DataFrame for easier manipulation
df = pd.DataFrame(dataset["train"])
print(df.head())

# Display the frequency of each label in the dataset to identify potential issues
print(df.groupby("label").size())

# Display information about the DataFrame to check for missing or invalid values
print(df.info())

# Remove rows with invalid labels (non-numeric characters)
df = df[(df.label == "0") | (df.label == "1") | (df.label == "2") | (df.label == "3") | (df.label == "4") | (df.label == "5") | (df.label == "6") | (df.label == "7") | (df.label == "8") | (df.label == "9")]

# Apply the cleaning function to preprocess all images
df["image"] = df["image"].apply(clean_img)

# Transform images into 1D arrays and normalize pixel values (range 0-1)
df["image"] = df["image"].apply(lambda x : np.array(x).reshape(-1) / 255)
# print(df.head(5))
# print(df["image"][0])

# Prepare feature (image) and label (digit) arrays for model training
x_data = np.stack(df["image"].values)
y_data = df["label"].values

# Split the dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=3)

# Initialize the SVM model with an RBF kernel (default for SVC)
# model = svm.SVC(gamma=0.01, verbose=2)
# The removal of the gamma parameter allows the model to automatically determine an optimal kernel scale, improving generalization.
model = svm.SVC(verbose=2)

# Train the model on the training set
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Evaluate the model's performance
print("\nAccuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# Generate and display the confusion matrix
cm = confusion_matrix(y_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix= cm)
cm_display.plot()
plt.show()