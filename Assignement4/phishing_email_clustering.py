from datasets import load_dataset
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import string
import numpy as np
import re
import umap
import matplotlib.pyplot as plt

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

# To see how much features and sample
print(np.shape(x_data))

# Reduce the ammount of features

# Intialize TruncatedSVD
svd = TruncatedSVD(n_components=100, algorithm="randomized")

# Reduce the dimension to 100 features to make it dense
x_reduced_data = svd.fit_transform(x_data)
print(np.shape(x_reduced_data))

db = DBSCAN(eps=0.5, min_samples=10).fit(x_data)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# Time to do the elbow method, https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/ helped me
# Intiialziing the list to stro the value
inertias = []

# Try clusters for 1 to 10 
for clusterAmmount in range(1,16):
    # Intialize model for an ammount of clusters
    kmeans = KMeans(n_clusters=clusterAmmount, init="k-means++", random_state=42)

    # Train the model                 
    kmeans.fit(x_reduced_data)

    # Save the inertia value
    inertias.append(kmeans.inertia_)

# Shows the graph of the elbow method
plt.title("Elbow Method: Inertia vs Number of Clusters")
plt.plot([i for i in range(1,16)], inertias, marker = 'o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# From the elbow graph we can tell that at 6 is the ammount of clusters we should take.

umap_reducer = umap.UMAP(n_components=2, random_state=42)
x_visualized_data = umap_reducer.fit_transform(x_reduced_data)

kmeans = KMeans(n_clusters=6, init="k-means++", random_state=42)
kmeans.fit(x_visualized_data)

# Create new df with only the clusters label and the labelled text
df_clusters = pd.DataFrame({ 
    "Clusters Label" : kmeans.labels_,
    "Text Label" : df["Email Type"]
})

test = df_clusters.groupby(["Clusters Label", "Text Label"]).size()
print(test)


plt.scatter(x_visualized_data[:, 0], x_visualized_data[:, 1], c=kmeans.labels_, s = 10)
plt.title("Clusters in 2D Space (UMAP)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.colorbar(label="Cluster Label")
plt.grid()
plt.show()