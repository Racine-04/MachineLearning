from datasets import load_dataset
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import string
import numpy as np
import re


def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Rmove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text

# Load dataset
dataset = load_dataset("zefang-liu/phishing-email-dataset")

# Show how the dataset formated
print(dataset)

# Convert into a df
df = pd.DataFrame(dataset["train"])

# Get info of df
print(f"{df.head()}\n")
print(f"{df.info()}\n")

# Remove the index column
df = df.drop(["Unnamed: 0"], axis=1)

# Remove the null elment in the email text and elemnts that are smaller than 100
df = df[(df["Email Text"].notnull()) & (df["Email Text"].str.len() > 500)]

# Check the labels type
print(f'{df.groupby("Email Type").size()}\n')

# Show the updated info
print(f"{df.info()}\n")

# Lowercase the string
df["Email Text"] = df["Email Text"].apply(clean_text)

# Get the features and values
x_raw_data = df["Email Text"].values
y_data = df["Email Type"].values


# Intialize a coutn vecotrizer
vectorizer = CountVectorizer(max_df=0.8)

# Tokenize the text data into a sparse matrix of token counts
x_sparse_matrix_data = vectorizer.fit_transform(x_raw_data)
print(np.shape(x_sparse_matrix_data))

# Intialisze a the TF-IDF transformer
transformer = TfidfTransformer()

# TRansform the text data into an TF-IDF representation
x_data = transformer.fit_transform(x_sparse_matrix_data)

# To see how much features and sample I have smaple 18634 and featrues 163224)
print(np.shape(x_data))

# Reduce the ammount of features

# Intialize TruncatedSVD

svd = TruncatedSVD(n_components=100, algorithm="randomized")
x_data = svd.fit_transform(x_data)
print(np.shape(x_data))

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

print(classification_report(y_test, y_predicted))
