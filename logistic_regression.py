import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

data = pd.read_csv("data.csv")
X = data.drop(columns=['HeartDiseaseorAttack'])
y = data['HeartDiseaseorAttack']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=24601)

# Fit logistic regression model, iter needs to be big else we'd get error
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# reward true positive more by setting a lower cutoff threshold
# Else the model prefers to predict most people to be negative
cutoff_threshold = 0.23
y_pred = model.predict_proba(X_test)[:, 1]
y_pred_use_custom_cutoff = (y_pred >= cutoff_threshold).astype(int)
y_pred_use_5_cutoff = (y_pred >= 0.5).astype(int)
auc_score = roc_auc_score(y_test, y_pred)
print(f'AUC: {auc_score:.4f}')

cm = confusion_matrix(y_test, y_pred_use_custom_cutoff)
print(f'0.23 Confusion Matrix:\ntn fp\nfn tp\n{cm}')
cm = confusion_matrix(y_test, y_pred_use_5_cutoff)
print(f'0.5 Confusion Matrix:\ntn fp\nfn tp\n{cm}')

# calculate lift
sorted_probs = model.predict_proba(X_test)[:, 1]
sorted_indices = np.argsort(sorted_probs)[::-1]
num_samples = len(y_test)
num_bins = 20
bin_size = num_samples // num_bins
lift_values = []


print('y_test,y_pred')
for i in range(num_bins):
    start_idx = i * bin_size
    end_idx = min((i + 1) * bin_size, num_samples)
    true_positives_in_bin = np.sum(y_pred[sorted_indices[start_idx:end_idx]])
    lift = true_positives_in_bin * num_bins / np.sum(y_test)
    lift_values.append(lift)
    for j in sorted_indices[start_idx:end_idx]:
        print(f'{y_test.iloc[j]},{y_pred[j]:.3f}')
        continue

plt.figure(figsize=(8, 6))
plt.plot((np.arange(1, num_bins + 1) * 100)/num_bins, lift_values, marker='o', linestyle='-')
plt.xlabel('Percentage of Data')
plt.ylabel('Lift')
plt.title('Lift Curve')
plt.grid(True)
plt.show()

