from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import ImageOps
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# Function to preprocess images
def clean_and_convert_img(img):
    # Resize image to 100x100 for consistency
    img = ImageOps.fit(img, (100, 100))
    # Convert image to grayscale
    img = ImageOps.grayscale(img)
    # Convert image to a NumPy array and normalize pixel values
    np_img = np.asarray(img) / 255
    # Flatten the image into a 1D array
    return np_img.flatten()

# Load the sports ball dataset
dataset = load_dataset("Shanav12/sports_ball_dataset")
print(dataset)

# Convert dataset splits into Pandas DataFrames
dfTrain = pd.DataFrame(dataset["train"])
dfValidation = pd.DataFrame(dataset["validation"])
dfTest = pd.DataFrame(dataset["test"])

# Merge training, validation, and test sets into a single DataFrame
df = pd.concat([dfTrain, dfTest, dfValidation], ignore_index=True)

# Print dataset information
print(df.head())  # First few rows
print(df.describe())  # Summary statistics
print(df.info())  # Data types and missing values
print(df.tail())  # Last few rows

# Check label distribution to understand class balance
df_label = df.groupby("label").count()
print(df_label)

# Apply image preprocessing to each image
df["image"] = df["image"].apply(clean_and_convert_img)
print(df.head())  # Verify transformation

# Extract features (images) and labels
x_data = np.stack(df["image"].values)
y_data = df["label"].values

# Split the dataset into training (75%) and test (25%) sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=7)

# Initialize LazyPredict's AutoML classifier
classifier = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)

# Fit the AutoML classifier on the training data and evaluate performance on test data
models, predictions = classifier.fit(x_train, x_test, y_train, y_test)

# Print model performance metrics
print(models)