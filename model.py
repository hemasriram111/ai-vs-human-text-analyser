import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load your dataset
dataset_path = r'C:\Users\hemas\Desktop\ai-vs-human\AI_Human.csv'  # Update with your dataset path
df = pd.read_csv(dataset_path)

# Check the class distribution
print("Class Distribution:")
print(df['generated'].value_counts())

# Split data into features (X) and labels (y)
X = df['text']
y = df['generated']

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = {0: 1, 1: len(df[df['generated'] == 0]) / len(df[df['generated'] == 1])}
print("\nClass Weights:", class_weights)

# Train a Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'ai_vs_human_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')