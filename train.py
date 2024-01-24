# %%
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report

import numpy as np

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import MyModel

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

pl.seed_everything(2023)
MAX_EPOCHS = 30
DETERMINISTIC = True
BATCH_SIZE = 64
LR = 0.001

# Learning rate scheduler params
STEP_SIZE = 20
GAMMA = 0.1

CLASSES = ['White', 'Black', 'Asian', 'Indian']

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.RandomCrop(104),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.CenterCrop(size=104),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load datasets and apply transformations
train_dataset = ImageFolder('data/utk_races/train/', transform=train_transform)
val_dataset = ImageFolder('data/utk_races/val/', transform=test_transform)
test_dataset = ImageFolder('data/utk_races/test/', transform=test_transform)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Training
model = MyModel(lr=LR, step_size=STEP_SIZE, gamma=GAMMA)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min')

trainer = pl.Trainer(callbacks=[early_stopping_callback], max_epochs=MAX_EPOCHS, deterministic=DETERMINISTIC)

trainer.fit(model, train_dataloader, val_dataloader)

# Evaluate model performance on the test set
model.eval()
correct = 0
total = 0
full_pred = torch.empty(0, 4)
full_label = torch.empty(0, dtype=torch.int32)

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data[0], data[1]

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        full_pred = torch.cat((full_pred, outputs.data))
        full_label = torch.cat((full_label, labels))

print("Accuracy of the network on the test set: {:.3f} %".format(100 * correct / total))
conf_matrix = confusion_matrix(full_label, torch.max(full_pred, 1).indices)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title('Test confusion matrix')

display = ConfusionMatrixDisplay(conf_matrix, display_labels=CLASSES)
display.plot(ax=ax)

plt.savefig('visualizations/test_confusion_matrix.png')

# Show confusion matrix with percentage
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig('visualizations/test_confusion_matrix_percentage.png')
plt.show()

# Plot val loss
flattened_loss = [x['val_loss'].cpu() for x in model.logged_metrics]

plt.plot(flattened_loss)
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.savefig('visualizations/validation_loss.png')
plt.show()
# %%
