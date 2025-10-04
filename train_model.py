import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Load Dataset
df = pd.read_csv('dataset/fake_or_real_news.csv')

# Step 2: Prepare data
X = df['text'].fillna('')   # input
y = df['label']             # output (0 = Fake, 1 = Real)

# Step 3: Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate
accuracy = model.score(X_test_tfidf, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# Step 7: Save model and vectorizer
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
print("💾 Model and vectorizer saved successfully in 'model/' folder")

