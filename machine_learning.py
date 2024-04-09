import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

# Load the dataset
data = pd.read_csv("data.csv")

# Prepare data
X = data.drop(columns=['HeartDiseaseorAttack']).values.astype(np.float32)
y = data['HeartDiseaseorAttack'].values.astype(np.float32)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets, use 33% test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=24601)

# Convert data to PyTorch tensors, nothing fancy just a bunch of matrices
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train).view(-1, 1)
y_test_tensor = torch.tensor(y_test).view(-1, 1)

# NN setup. 5 layers with dropout at 2nd layer
# 21 feature -> 128n -> 64n -> dropout -> 32n -> 8n -> 1n
# all fully connected and uses RELU except last one uses
# sigmoid.
# perf-wise you can use 3 layers without dropout to achieve
# similar performance. The optimal solution on this dataset
# is just simple logistic regression and it seems you do not
# need a super complicated model to mimic it.
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

input_size = X_train.shape[1]
model = Net(input_size)

# Regular BCE loss will predict everything as negative :(
# criterion = nn.BCELoss()
# This is because regular loss function punishes fn and fp the same
# The base rate of positive is too low. So it's best for the model
# to predict everything as negative
class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive, weight_negative):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        loss_positive = self.weight_positive * target * torch.log(torch.sigmoid(input) + 1e-9)
        loss_negative = self.weight_negative * (1 - target) * torch.log(1 - torch.sigmoid(input) + 1e-9)
        return torch.neg(torch.mean(loss_positive + loss_negative))

# Instantiate the weighted BCE loss with custom weights
weight_positive = 5  # Increase weight for positive class (reward true positives more)
weight_negative = 1  # Decrease weight for negative class (punish false negatives more)
criterion = WeightedBCELoss(weight_positive, weight_negative)

# learning rate dictates how fast the model will learn
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training the model
num_epochs = 80
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor)
        test_losses.append(test_loss.item())

        # Convert probabilities to binary predictions, using 0.5 as cutoff
        test_predicted = (outputs >= 0.5).float()
        auc_score = roc_auc_score(y_test_tensor.numpy(), outputs.numpy())

        # Calculate evaluation metrics
        cm = confusion_matrix(y_test_tensor.numpy(), test_predicted.numpy())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        print(f'Confusion Matrix:\ntn fp\nfn tp\n{cm}')
        print(f'AUC: {auc_score:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    train_predicted = (train_outputs >= 0.5).float()
    test_outputs = model(X_test_tensor)
    test_predicted = (test_outputs >= 0.5).float()

train_accuracy = accuracy_score(y_train_tensor.numpy(), train_predicted.numpy())
test_accuracy = accuracy_score(y_test_tensor.numpy(), test_predicted.numpy())
print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

predictions = test_outputs.numpy()[:,0]
true_labels = y_test_tensor.numpy()

# Sort predictions and true labels based on predicted probabilities
sorted_indices = np.argsort(predictions)[::-1] # desc
sorted_labels = true_labels[sorted_indices]

# Calculate lift
num_samples = len(sorted_labels)
num_bins = 20
bin_size = num_samples // num_bins
lift_values = []

print('y_test,y_pred')
for i in range(num_bins):
    start_idx = i * bin_size
    end_idx = min((i + 1) * bin_size, num_samples)
    true_positives_in_bin = np.sum(np.multiply(predictions[sorted_indices[start_idx:end_idx]], y_test[sorted_indices[start_idx:end_idx]]))
    random_positives_in_bin = np.sum(y_test[sorted_indices[start_idx:end_idx]]) * np.sum(y_test) / len(y_test)
    lift = true_positives_in_bin / random_positives_in_bin
    lift_values.append(lift)
    for j in sorted_indices[start_idx:end_idx]:
        print(f'{y_test[j]},{predictions[j]:.3f}')
        continue

# Plot lift curve
plt.figure(figsize=(12, 10))
plt.plot(np.arange(1, num_bins + 1) * 100// num_bins, lift_values, marker='o', linestyle='-')
plt.xlabel('Percentage of Data')
plt.ylabel('Lift')
plt.title('Lift Curve')
plt.grid(True)
plt.show()


# # Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
