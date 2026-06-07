# Task 5: Decision Tree and Random Forest Classification

## Heart Disease Prediction using Decision Tree and Random Forest

---

## Objective

The objective of this task is to build and evaluate Decision Tree and Random Forest classification models for predicting heart disease using the Heart Disease dataset.

---

## Tools and Libraries Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## Dataset

The Heart Disease dataset contains medical information related to patients, including various health parameters used to predict the presence of heart disease.

The dataset includes features such as:

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope
- Number of Major Vessels (ca)
- Thal
- Target

---

## Steps Performed

### 1. Dataset Loading

- Loaded the Heart Disease dataset using Pandas.
- Displayed the dataset shape and first five rows for inspection.

### 2. Feature and Target Selection

- Selected all input features as **X**.
- Selected the **target** column as **y**.

### 3. Train-Test Split

- Split the dataset into training and testing sets using `train_test_split()`.
- Training Data: 80%
- Testing Data: 20%
- Random State: 42

### 4. Decision Tree Model

- Created a Decision Tree Classifier.
- Trained the model using the training dataset.
- Generated predictions on the testing dataset.
- Calculated the model accuracy.

### 5. Decision Tree Visualization

- Visualized the trained Decision Tree using `plot_tree()`.
- Saved the visualization as:

`screenshots/decision_tree_visualization.png`

### 6. Random Forest Model

- Created a Random Forest Classifier with `n_estimators=100`.
- Trained the model using the training dataset.
- Generated predictions on the testing dataset.
- Calculated the Random Forest accuracy.

### 7. Cross Validation

- Applied 5-Fold Cross Validation using `cross_val_score()`.
- Displayed the cross-validation scores.
- Calculated the average cross-validation score.

### 8. Feature Importance

- Extracted feature importance values from the Random Forest model.
- Displayed the importance score of each feature.
- Sorted features according to their importance.

### 9. Feature Importance Visualization

- Created a horizontal bar graph for feature importance.
- Saved the graph as:

`screenshots/feature_importance_graph.png`

---

## Result

Successfully implemented Decision Tree and Random Forest classification models for Heart Disease prediction.

The models were evaluated using accuracy and cross-validation, and the important features contributing to the prediction were identified and visualized.
