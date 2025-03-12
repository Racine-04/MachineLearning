import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageOps, Image
from datasets import load_dataset
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class FruitClassifier(nn.Module):
    def __init__(self, number_of_classes = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 12, 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 36,3),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 64,3),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*26*26, number_of_classes)
        )

    def forward(self, x):
        return self.network(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def clean_img(img):
    '''
    # Resize image to 224 x 224, and crop a bit of the border
    img = ImageOps.fit(img, (224, 224), bleed=0.07)

    # Graysclae the img
    img = ImageOps.grayscale(img)

    # Normalize
    np_img = np.array(img) /255

    new_img = Image.fromarray(np_img)

    '''
    # Tranform to a tensor already does the normalization for you and resize
    new_img = transform(img)
    return new_img

datasetTrain = load_dataset("arnavmahapatra/fruit-detection-dataset", split="train")
datasetTest = load_dataset("arnavmahapatra/fruit-detection-dataset", split="test")

print(datasetTrain)
print(datasetTest)

# Convert dataset into Data frame
dfTrain= pd.DataFrame(datasetTrain) 
dfTest= pd.DataFrame(datasetTest) 

print(dfTrain.head())
print(dfTrain.tail())

# Clean the img
dfTrain["image"] = dfTrain["image"].apply(clean_img)
dfTest["image"] = dfTest["image"].apply(clean_img)

img = dfTrain["image"][0]
print(img)

# Shuffle the DF
dfTrain = dfTrain.sample(frac=1, random_state=42).reset_index(drop=True)
dfTest = dfTest.sample(frac=1, random_state=42).reset_index(drop=True)

# Getting values and converting into tensors
x_train = torch.stack(dfTrain["image"].tolist())
y_train = torch.LongTensor(dfTrain["label"].values)
x_test = torch.stack(dfTest["image"].tolist())
y_test = torch.LongTensor(dfTest["label"].values)

# Initialize the model
model = FruitClassifier().to("cpu")

# Choose loss function
loss_fn = nn.CrossEntropyLoss()

# Choose optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model 
epochs = 15
looses = []

for e in range(epochs):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    looses.append(loss.item())

    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())


# Loss curve
plt.plot(range(epochs), looses)
plt.grid()
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Evaluate the model

with torch.no_grad():
    logits = model(x_test)
    predicted_labels = logits.argmax(dim=1)

# Evaluate the model's performance
print("\nAccuracy:", accuracy_score(np.asarray(y_test), np.asarray(predicted_labels)))
print("\nClassification Report:\n", classification_report(np.asarray(y_test), np.asarray(predicted_labels)))

# Generate and display the confusion matrix
cm = confusion_matrix(np.asarray(y_test), np.asarray(predicted_labels))
cm_display = ConfusionMatrixDisplay(confusion_matrix= cm)
cm_display.plot()
plt.show()