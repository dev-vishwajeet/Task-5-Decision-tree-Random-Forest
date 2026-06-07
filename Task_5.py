import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===================================
# Load Dataset
# ===================================

df = pd.read_csv("heart.csv")
print("Dataset Loaded Successfully!\n")
print(df.head())
print("\nDataset Shape:", df.shape)

# ===================================
# Features and Target
# ===================================

X = df.drop("target", axis=1)
y = df["target"]

# ===================================
# Train Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ===================================
# Decision Tree
# ===================================

dt_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    y_pred_dt
)
print("\nDecision Tree Accuracy:", dt_accuracy)

# ===================================
# Decision Tree Visualization
# ===================================

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    fontsize=8
)
plt.savefig(
    "screenshots/decision_tree_visualization.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ===================================
# Random Forest
# ===================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)
print("\nRandom Forest Accuracy:", rf_accuracy)

# ===================================
# Cross Validation
# ===================================

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores:")
print(cv_scores)

print(
    "\nAverage CV Score:",
    cv_scores.mean()
)

# ===================================
# Feature Importance
# ===================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nFeature Importance:\n")
print(importance)

# ===================================
# Feature Importance Graph
# ===================================

plt.figure(figsize=(14, 10))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)
plt.title("Random Forest Feature Importance"
    "Random Forest Feature Importance"
)
plt.xlabel("Importance")

plt.savefig(
    "screenshots/feature_importance_graph.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()