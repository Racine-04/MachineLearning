import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch import nn

# Create the model class
class IrisClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, number_of_classes) 
        )
    
    def forward(self, x):
        return self.network(x)

# Load dataset
dataset = load_dataset("scikit-learn/iris")

# Convert data into a dataframe
df = pd.DataFrame(dataset["train"])
print(df.head())

# Remove id column
df = df.drop("Id", axis=1)

# Convert the string classification to numbers
df["Species"] = df["Species"].replace("Iris-setosa", 0)
df["Species"] = df["Species"].replace("Iris-versicolor", 1)
df["Species"] = df["Species"].replace("Iris-virginica", 2)


# Split the input and value
x_data = df.drop("Species", axis=1).values
y_data = df["Species"].values

print(x_data[:5])
print(y_data)

# Split into train test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=2)

# Convert data into tesnors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Initialize the model
input_dim = 4
hidden_dim = 8
number_of_classes = 3
model = IrisClassifier(input_dim, hidden_dim, number_of_classes).to("cpu")

# Initialize loss function
loss_fn = nn.CrossEntropyLoss()

# Choose an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model
epochs = 100
losses = []

for e in range(epochs):
    y_pred = model(x_train) 
    loss = loss_fn(y_pred, y_train)

    losses.append(loss.item())

    if e % 10 == 0:
        print(f'Epoch {e} and loss: {loss.item()}')
    
    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph loss curves
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.grid()
plt.show()

# Evalutate the model
y_predict = []
with torch.no_grad():
    y_pred = model(x_test)
    print(y_pred )
    predicted_labels = y_pred.argmax(dim=1)
    print(predicted_labels)

# Evaluate the model's performance
print("\nAccuracy:", accuracy_score(np.asarray(y_test), np.asarray(predicted_labels)))
print("\nClassification Report:\n", classification_report(np.asarray(y_test), np.asarray(predicted_labels)))

# Generate and display the confusion matrix
cm = confusion_matrix(np.asarray(y_test), np.asarray(predicted_labels))
cm_display = ConfusionMatrixDisplay(confusion_matrix= cm)
cm_display.plot()
plt.show()