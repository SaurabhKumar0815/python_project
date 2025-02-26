import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- Simulate Data (Replace with your actual data) ---
data = {
    'age',
    'income',
    'location' ['Urban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Rural', 'Urban'],
    'online_real_estate_searches',
    'real_estate_interest' ['High', 'High', 'High', 'High', 'High', 'High', 'High', 'High', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low']
}

df = pd.DataFrame(data)

# --- Data Preprocessing ---

# Encode categorical variables (location, real_estate_interest)
label_encoder = LabelEncoder()
df['location'] = label_encoder.fit_transform(df['location'])
df['real_estate_interest'] = label_encoder.fit_transform(df['real_estate_interest'])  # 0: High, 1: Low

# --- Exploratory Data Analysis (EDA) ---
# Example: Pairplot to visualize relationships
sns.pairplot(df, hue='real_estate_interest')
plt.show()

# Example: Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# --- Feature Selection and Model Training ---

X = df.drop('real_estate_interest', axis=1)
y = df['real_estate_interest']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Model Evaluation ---

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Feature Importance ---
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(5).plot(kind='barh')
plt.title('Feature Importance')
plt.show()

# --- Insight Extraction (Example) ---
# You can further analyze the model's predictions and feature importances
# to derive specific insights. For instance:

# Example: Predict interest for a new individual
new_individual = pd.DataFrame({
    'age',
    'income',
    'location' [label_encoder.transform(['Suburban'])], # Transform the label
    'online_real_estate_searches'
})

prediction = model.predict(new_individual)
print(f"Predicted real estate interest (0: High, 1: Low): {prediction}")

# Example: Based on feature importance, online searches and income are most important, therefore target marketing based on these features.