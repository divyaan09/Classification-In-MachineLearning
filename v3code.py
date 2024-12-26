import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
data = pd.read_csv('company_sales_data_1000.csv')

# Preview the dataset
print(data.head())

# Check for missing values and handle them if any
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode categorical variables using get_dummies
data = pd.get_dummies(data, columns=['Product_Name', 'Quarter', 'Region'], drop_first=True)

# Separate features and target
X = data.drop('Swarm_Behaviour', axis=1)
y = data['Swarm_Behaviour']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a RandomForest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Plot the confusion matrix manually
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=['Not Flocking', 'Flocking'], yticklabels=['Not Flocking', 'Flocking'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the feature importances
feature_importances = classifier.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Additional visualizations
sns.set(style="whitegrid")

# Pair Plot
plt.figure(figsize=(10, 10))
sns.pairplot(data, hue='Swarm_Behaviour', palette='viridis')
plt.title('Pair Plot of Features')
plt.show()

# Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Swarm_Behaviour', data=data, palette='viridis')
plt.title('Count Plot of Swarm Behaviour')
plt.xlabel('Swarm Behaviour')
plt.ylabel('Count')
plt.show()

# Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Swarm_Behaviour', y='Sales_Amount', data=data, palette='viridis')
plt.title('Box Plot of Sales Amount by Swarm Behaviour')
plt.xlabel('Swarm Behaviour')
plt.ylabel('Sales Amount')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Scatter plot of two features (adjust as necessary)
sns.scatterplot(x='Sales_Amount', y='Units_Sold', hue='Swarm_Behaviour', data=data, ax=axs[0, 0], palette='viridis')
axs[0, 0].set_title('Scatter Plot of Sales Amount vs. Units Sold')
axs[0, 0].set_xlabel('Sales Amount')
axs[0, 0].set_ylabel('Units Sold')

# Pie chart of class distribution
class_counts = y.value_counts()
axs[0, 1].pie(class_counts, labels=['Not Flocking', 'Flocking'], autopct='%1.1f%%', startangle=90)
axs[0, 1].axis('equal')
axs[0, 1].set_title('Class Distribution')

# Histogram of a specific feature (adjust as necessary)
sns.histplot(data['Sales_Amount'], bins=15, kde=True, ax=axs[1, 0], color='purple')
axs[1, 0].set_title('Distribution of Sales Amount')
axs[1, 0].set_xlabel('Sales Amount')
axs[1, 0].set_ylabel('Frequency')

# Heatmap of correlation between features
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axs[1, 1])
axs[1, 1].set_title('Heatmap of Feature Correlations')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
plt.show()
