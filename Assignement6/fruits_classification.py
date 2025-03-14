import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageOps, Image
from datasets import load_dataset
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

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

dataset = load_dataset("arnavmahapatra/fruit-detection-dataset")

print(dataset)

# Convert dataset into Data frame
df = pd.concat([pd.DataFrame(dataset["train"]), pd.DataFrame(dataset["test"])], ignore_index=True)

print(df.head())
print(df.tail())

# Clean the img
df["image"] = df["image"].apply(clean_img)

img = df["image"][0]
print(img)

# Split the data
# Getting values and converting into tensors
x_data = torch.stack(df["image"].tolist())
y_data = torch.LongTensor(df["label"].values)

# 80% train 10 % valdiation, 10 % test
x_train, x_rest, y_train, y_rest = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=42)

# Initialize the model
model = FruitClassifier().to("cpu")

# Choose loss function
loss_fn = nn.CrossEntropyLoss()

# Choose optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model 
epochs = 30
looses = []
accuracies = []

for e in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    looses.append(loss.item())

    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())

    with torch.no_grad():
        model.eval()
        y_validation = model(x_val)
        y_validation = y_validation.argmax(dim=1)

        accuracies.append(accuracy_score(np.asarray(y_val), np.asarray(y_validation)))    


# Loss curve
plt.plot(range(epochs), looses, marker = "o")
plt.grid()
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Accuracy curve
plt.plot(range(epochs), accuracies, marker="o")
plt.grid()
plt.title("Accuracy curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Evaluate the model
model.eval()
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