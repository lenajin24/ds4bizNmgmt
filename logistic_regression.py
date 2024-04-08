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

# reward true positive more
cutoff_threshold = 0.23
y_pred = (model.predict_proba(X_test)[:, 1] >= cutoff_threshold).astype(int)
auc_score = roc_auc_score(y_test, y_pred)
print(f'AUC: {auc_score:.4f}')

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\ntn fp\nfn tp\n{cm}')

sorted_probs = model.predict_proba(X_test)[:, 1]
sorted_indices = np.argsort(sorted_probs)[::-1]
num_samples = len(y_test)
num_bins = 20
bin_size = num_samples // num_bins
lift_values = []

for i in range(num_bins):
    start_idx = i * bin_size
    end_idx = min((i + 1) * bin_size, num_samples)
    true_positives_in_bin = np.sum(y_test.iloc[sorted_indices[start_idx:end_idx]])
    lift = true_positives_in_bin / ((i + 1) * bin_size)
    lift_values.append(lift)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, num_bins + 1), lift_values, marker='o', linestyle='-')
plt.xlabel('Percentage of Data')
plt.ylabel('Lift')
plt.title('Lift Curve')
plt.grid(True)
plt.show()

