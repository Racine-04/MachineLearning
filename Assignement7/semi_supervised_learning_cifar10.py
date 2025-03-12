from datasets import load_dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Create a generator
def generator(data):
    for i in range(0, len(data), 500):
        yield data[i:i+500]


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()])

def train_model(model, x_train, y_train, epochs):
    torch.cuda.empty_cache()
    model.train()
    losses = []

    x_train = x_train.to("cuda")
    y_train = y_train.to("cuda")

    for e in range(epochs):
        y_pred = model(x_train)

        loss = loss_fn(y_pred, y_train)

        losses.append(loss)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

def clean_convert_img(img):
    return transform(img)

class FashionMNSITClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 20, 3),
            nn.ReLU(),
            nn.Conv2d(20, 34, 3),
            nn.BatchNorm2d(34),
            nn.ReLU(),
            nn.Conv2d(34, 40, 3),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(40*22*22, 10)
        )

    def forward(self, x):
      return self.network(x)

dataset = load_dataset("ylecun/mnist")
print(dataset)

# Dataset with 60 k of data
df = pd.concat([pd.DataFrame(dataset["train"]), pd.DataFrame(dataset["test"])], ignore_index=True)

print(df.head())
print(df.info())

# Convert image into tensor
df["image"] = df["image"].apply(clean_convert_img)

# Split the dataset
x_data = torch.stack(df["image"].tolist())
y_data = torch.LongTensor(df["label"].values)

# Split of unsupervised 70%, train 15%, test 15%
x_un, x_rest, y_un, y_rest = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=7)
x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=0.5, shuffle=True, random_state=7)

# Initialize the model
model = FashionMNSITClassifier().to("cuda")

# Choose loss function
loss_fn = nn.CrossEntropyLoss()

# Choose optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 40

# Train the model on the supervised part
train_model(model, x_train, y_train, epochs)

# Evaluate the model tained on the supervised part
model.eval()
with torch.no_grad():
    logits = model(x_test.to("cuda"))
    predicted_labels_test  = logits.argmax(dim=1).detach().cpu()

# Performance on test set training model only on labelled training data
print("\nClassification Report:\n", classification_report(np.asarray(y_test), np.asarray(predicted_labels_test)))

# Reduce the learninf rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Pseudo-labeling

# Only use high-confidence pseudo-labels and discard uncertain ones:
threshold = 0.85  # Only keep pseudo-labels with >85% confidence

for batch in generator(x_un):
  # Generate pseudo-label
  high_conf_x = []
  high_conf_y = []
  model.eval()

  with torch.no_grad():
    pseudo_logits = model(batch.to("cuda"))
    pseudo_label_probs = torch.softmax(pseudo_logits, dim=1)
    confidence, predicted = torch.max(pseudo_label_probs, dim=1)

  for i in range(len(confidence)):
    if confidence[i].item() >= threshold:
      high_conf_x.append(batch[i])
      high_conf_y.append(predicted[i].item())

  # Train the model with the new labels found
  if high_conf_x:
    # Convert into tensor
    high_conf_x = torch.stack(high_conf_x)
    high_conf_y = torch.LongTensor(high_conf_y)
    train_model(model, high_conf_x, high_conf_y, 3)

# Evaluate the model on unseen data
with torch.no_grad():
    logits = model(x_test.to("cuda"))
    predicted_labels_test  = logits.argmax(dim=1).detach().cpu()

# Performance on test set training model only on labelled training data
print("\nClassification Report:\n", classification_report(np.asarray(y_test), np.asarray(predicted_labels_test)))