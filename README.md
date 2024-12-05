# Decision Tree Classifier with PCA and StandardScaler  

This project demonstrates how to preprocess data using **StandardScaler** for feature standardization and **Principal Component Analysis (PCA)** for dimensionality reduction. A **Decision Tree Classifier** is trained and evaluated to explore the impact of different PCA configurations on classification accuracy.  

---

## 📋 Table of Contents  
1. [Project Overview](#project-overview)  
2. [Dataset Requirements](#dataset-requirements)  
3. [Methodology](#methodology)  
4. [Dependencies](#dependencies)  
5. [Steps to Use](#steps-to-use)  
6. [Results](#results)  

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
