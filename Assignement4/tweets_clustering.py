from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import pandas as pd
import string
import numpy as np
import re
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
dataset = load_dataset("gxb912/large-twitter-tweets-sentiment")

# Show how the dataset formated
print(dataset)

# Convert into a df
df = pd.concat([pd.DataFrame(dataset["train"]), pd.DataFrame(dataset["test"])], ignore_index=True)

# Get info of df
print(f"{df.head()}\n")
print(f"{df.info()}\n")

# Remove the null elment in the text and elemnts that are smaller than 5
df = df[(df["text"].notnull()) & (df["text"].str.len() > 5)]

# Check the labels type
print(f'{df.groupby("sentiment").size()}\n')

# Show the updated info
print(f"{df.info()}\n")

# Lowercase the string
df["text"] = df["text"].apply(clean_text)

# Get the features and values
x_raw_data = df["text"].values

# Intialize a count vecotrizer
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
svd = TruncatedSVD(n_components=2, algorithm="randomized", random_state=10)

# Reduce the dimension to 2 features to make it dense and visualizable
x_reduced_data = svd.fit_transform(x_data)

print(np.shape(x_reduced_data))

# Time to do the elbow method, https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/ helped me
# Intiialziing the list to stro the value
inertias = []

# Try clusters for 1 to 20 
for clusterAmmount in range(1,20):
    # Intialize model for an ammount of clusters
    kmeans = KMeans(n_clusters=clusterAmmount, init="k-means++", random_state=10)

    # Train the model                 
    kmeans.fit(x_reduced_data)

    # Save the inertia value
    inertias.append(kmeans.inertia_)

# Shows the graph of the elbow method
plt.title("Elbow Method: Inertia vs Number of Clusters")
plt.plot([i for i in range(1,20)], inertias, marker = 'o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Train the model using sklearn kmeans using the right ammount of clusers
# From the elbow graph we can tell that at 5 is the ammount of clusters we should take.
kmeans_graph = KMeans(n_clusters=5, init="k-means++", random_state=42)
kmeans_graph.fit(x_reduced_data)

# Visualize the clusters in 2D space
plt.scatter(x_reduced_data[:, 0], x_reduced_data[:, 1], c=kmeans_graph.labels_, s = 10)
plt.title("Clusters in 2D Space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(label="Cluster Label")
plt.grid()
plt.show()

# The meaning of the clusters
df_clusters = pd.DataFrame({ 
    "ClustersLabel" : kmeans_graph.labels_,
    "Type" : df["sentiment"],
    "Text" : df["text"]
})

interpolation = df_clusters.groupby(["ClustersLabel", "Type"]).size()
print(interpolation)

print(df_clusters.groupby(["ClustersLabel"]).size())

# Train using supervised learning

x_data_supervised = df_clusters["Text"].values
y_data = df_clusters["ClustersLabel"].values

# Initialize 
vectorizer_supervised = TfidfVectorizer(max_df=0.8)

# Convert to a TF-IDF representation matrix
x_data_supervised = vectorizer_supervised.fit_transform(x_data_supervised)
print(np.shape(x_data_supervised))

# Reduce ammount of features
svd_supervised = TruncatedSVD(n_components=100, algorithm="randomized", random_state=12)
x_data_supervised = svd_supervised.fit_transform(x_data_supervised)
print(np.shape(x_data_supervised))

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_data_supervised, y_data, test_size=0.3, shuffle=True, random_state=12)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=10, random_state=12)
model.fit(x_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(x_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))