# Decision Tree Classifier with PCA and StandardScaler  

This project demonstrates how to preprocess data using **StandardScaler** for feature standardization and **Principal Component Analysis (PCA)** for dimensionality reduction. A **Decision Tree Classifier** is trained and evaluated to explore the impact of different PCA configurations on classification accuracy.  


---

## 📂 Project Overview  

### **Goals**  
- Standardize features using **StandardScaler** to normalize the data.  
- Apply **PCA** to incrementally reduce the number of dimensions in the dataset.  
- Train a **Decision Tree Classifier** on each PCA configuration.  
- Evaluate the classifier's performance for varying numbers of PCA components to determine the optimal setup.  

---

## 📊 Dataset Requirements  

Ensure the dataset meets the following criteria:  

1. **Split the data** into:  
   - Features: `X_train` (training data), `X_test` (testing data).  
   - Target: `y_train` (training labels), `y_test` (testing labels).  

2. Features must be **numeric**, and the target variable should be **categorical** (binary or multi-class).  

---

## ⚙️ Methodology  

1. **Scaling**:  
   Standardize features to have mean = 0 and standard deviation = 1 using **StandardScaler**.  

2. **PCA**:  
   Incrementally reduce dimensions from 1 to the total number of features using **Principal Component Analysis**.  

3. **Model Training**:  
   Train a **DecisionTreeClassifier** for each PCA configuration.  

4. **Evaluation**:  
   Evaluate the classifier by calculating the accuracy for each PCA configuration.  

---

## 🛠 Dependencies  

Install the required libraries before running the code:  

```bash
pip install pandas numpy scikit-learn

```


## 🚀 Steps to Use
Load your dataset and split it into:

 - X_train (training data)
 - X_test (testing data)
 - y_train (training labels)
 - y_test (testing labels)
---
Copy and paste the following code into a Jupyter Notebook or Python script to preprocess the data, apply PCA, and evaluate the Decision Tree Classifier:

```bash
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Scale the data
scale = StandardScaler()
clf3 = ColumnTransformer(
    transformers=[('scale', scale, X_train.columns)],
    remainder='passthrough'
)
x_train_scaled = clf3.fit_transform(X_train)
x_test_scaled = clf3.transform(X_test)

# Convert scaled data to DataFrame
x_train_scaled = pd.DataFrame(x_train_scaled, columns=X_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=X_test.columns)

# Train and evaluate with PCA
dtc = DecisionTreeClassifier(random_state=42)
for i in range(1, len(X_train.columns) + 1):
    pca = PCA(n_components=i, random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    dtc.fit(x_train_pca, y_train)
    y_pred = dtc.predict(x_test_pca)
    print(f'Decision Tree Classifier with {i} PCA components: Accuracy = {accuracy_score(y_test, y_pred):.4f}')
```
---

## Results
Decision Tree Classifier with 1 PCA components: Accuracy = 0.03 Decision Tree Classifier with 2 PCA components: 
Accuracy = 0.06 ... Decision Tree Classifier with 16 PCA components: Accuracy = 0.09
