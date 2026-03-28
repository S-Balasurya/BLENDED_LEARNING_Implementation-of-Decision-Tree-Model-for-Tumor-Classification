# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Decision Tree Classifier is used to classify tumor data into different classes based on the selected features.

2.Dataset Preparation involves loading the dataset and selecting input features and the target variable.

3.Train-Test Split divides the dataset into training and testing sets for model training and evaluation.

4.Model Evaluation is performed by predicting test data and calculating accuracy, classification report, and confusion matrix.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Balasurya S
RegisterNumber: 212225100003 
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Step 2
#Load dataset
data=pd.read_csv('tumor.csv')

print(data.head())
print(data.columns)

features=['Clump','UnifSize','UnifShape','MargAdh','SingEpiSize','BlandChrom', 'NormNucl', 'Mit']
target='Class'
x=data[features]
y=data[target]

#step 4
#splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name:Balasurya S")
print("Register No:212225100003")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:

<img width="553" height="317" alt="Screenshot 2026-03-28 145225" src="https://github.com/user-attachments/assets/f94a88e0-3feb-4077-93a6-1e812f0cf6c7" />

<img width="755" height="388" alt="Screenshot 2026-03-28 145512" src="https://github.com/user-attachments/assets/5f697a7f-085e-475b-a2f0-fac075f2914a" />

 <img width="313" height="88" alt="Screenshot 2026-03-28 145540" src="https://github.com/user-attachments/assets/066eb0af-d84f-44e0-bc85-5ced99f0a3ea" />

<img width="544" height="301" alt="Screenshot 2026-03-28 145601" src="https://github.com/user-attachments/assets/d456cc55-f6e1-40ff-8b68-43c38c75e06a" />

<Figure size 640x480 with 2 Axes><img width="539" height="453" alt="image" src="https://github.com/user-attachments/assets/0ab468f3-cefc-4cc8-93bb-5ea95d270d51" />

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
