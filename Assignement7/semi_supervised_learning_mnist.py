from datasets import load_dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Create a generator that yields data in batches of 500
def generator(data):
    for i in range(0, len(data), 500):
        jump = min(i+500, len(data)) 
        yield data[i:jump]

# Define basic transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Define data augmentation techniques to improve generalization
# However, in the final implementation, it did not significantly improve results
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
    transforms.ToTensor()
])

# Function to train the model with given training and validation data
def train_model(model, x_train, y_train, x_val, y_val, epochs):
    losses = []
    accuracies = []

    for e in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in zip(generator(x_train), generator(y_train)):
            # batch_x = torch.stack([data_augmentation(transforms.functional.to_pil_image(img)) for img in batch_x])
            batch_x = batch_x.to("cuda")
            batch_y = batch_y.to("cuda")
            
            # Forward pass
            y_pred_batch = model(batch_x)
            loss = loss_fn(y_pred_batch, batch_y)
            epoch_loss += loss.item()

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del batch_x
            del batch_y
            torch.cuda.empty_cache()

            num_batches += 1
        
        # Evaluate model on validation set
        model.eval()
        with torch.no_grad():
            logits = model(x_val.to("cuda"))
            predicted_labels_test = logits.argmax(dim=1).detach().cpu()

        epoch_acc = accuracy_score(np.asarray(y_val), np.asarray(predicted_labels_test))

        epoch_loss /= num_batches
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

    return losses, accuracies

# Function to apply preprocessing transformations
def clean_convert_img(img):
    return transform(img)

# Define CNN model for MNIST classification
class MNISTClassifier(nn.Module):
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

# Load the MNIST dataset
dataset = load_dataset("ylecun/mnist")
print(dataset)

# Convert dataset into Pandas DataFrame for easier manipulation
df = pd.concat([pd.DataFrame(dataset["train"]), pd.DataFrame(dataset["test"])], ignore_index=True)
print(df.head())
print(df.info())

# Apply preprocessing transformations
df["image"] = df["image"].apply(clean_convert_img)

# Convert images and labels into tensors
x_data = torch.stack(df["image"].tolist())
y_data = torch.LongTensor(df["label"].values)

# Split dataset: 70% Unsupervised, 15% Train, 15% Test
x_un, x_rest, y_un, y_rest = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=7)
x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=0.5, shuffle=True, random_state=7)

# Use first 500 elements for validation
x_val = x_test[:500]
y_val = y_test[:500]

# Initialize model and training parameters
model = MNISTClassifier().to("cuda")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 40

# Train the model using only labeled data
supervised_losses, supervised_accuracy = train_model(model, x_train, y_train, x_val, y_val, epochs)

# Plot supervised training loss
plt.title("Supervised Training Loss")
plt.plot(range(len(supervised_losses)), supervised_losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot supervised training accuracy
plt.title("Supervised Training Accuracy")
plt.plot(range(len(supervised_accuracy)), supervised_accuracy, label="Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate the supervised model
model.eval()
with torch.no_grad():
    logits = model(x_test.to("cuda"))
    predicted_labels_test = logits.argmax(dim=1).detach().cpu()

# Print classification report
print("\nClassification Report:\n", classification_report(np.asarray(y_test), np.asarray(predicted_labels_test)))

# Generate and display confusion matrix
cm = confusion_matrix(np.asarray(y_test), np.asarray(predicted_labels_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Reduce learning rate before fine-tuning with pseudo-labeling
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Generate pseudo-labels with a confidence threshold of 85%
threshold = 0.85  
high_conf_x = []
high_conf_y = []

model.eval()
with torch.no_grad():
    for batch in generator(x_un):
        pseudo_logits = model(batch.to("cuda"))
        pseudo_label_probs = torch.softmax(pseudo_logits, dim=1)
        confidence, predicted = torch.max(pseudo_label_probs, dim=1)

        for i in range(len(confidence)):
            if confidence[i].item() >= threshold:
                high_conf_x.append(batch[i])
                high_conf_y.append(predicted[i].item())

print(f"Number of high-confidence samples: {len(high_conf_x)}")

# Fine-tune the model using only high-confidence pseudo-labels
if high_conf_x:
    high_conf_x = torch.stack(high_conf_x)
    high_conf_y = torch.LongTensor(high_conf_y)   
    pseudo_losses, pseudo_accuracy = train_model(model, high_conf_x, high_conf_y, x_val, y_val, 5)

    # Plot pseudo-labeling loss
    plt.title("Pseudo-labeling Loss")
    plt.plot(range(len(pseudo_losses)), pseudo_losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Plot pseudo-labeling accuracy
    plt.title("Pseudo-labeling Accuracy")
    plt.plot(range(len(pseudo_accuracy)), pseudo_accuracy, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# Evaluate the semi-supervised model
model.eval()
with torch.no_grad():
    logits = model(x_test.to("cuda"))
    predicted_labels_test = logits.argmax(dim=1).detach().cpu()

# Performance on test set training model with pseudo-labeling
print("\nClassification Report:\n", classification_report(np.asarray(y_test), np.asarray(predicted_labels_test)))

# Generate and display the confusion matrix
cm = confusion_matrix(np.asarray(y_test), np.asarray(predicted_labels_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()