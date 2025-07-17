import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset (force correct column names, ignore header in file)
try:
    df = pd.read_csv("sms.csv", sep='\t', header=None, names=['label', 'message'])
    print("✅ CSV file loaded successfully.")
except Exception as e:
    print("❌ Failed to load CSV file. Error:")
    print(e)
    exit()

# Step 2: Show a preview of the data
print("\n📝 Dataset Preview:")
print(df.head())

# Step 3: Show info about columns and check if data is loaded properly
print("\n📊 Dataset Info:")
print(df.info())

# Step 4: Show how many messages are loaded
print("\n📦 Total messages loaded:", len(df))

# Step 5: Check if required columns exist
if 'label' not in df.columns or 'message' not in df.columns:
    print("❌ ERROR: Required columns 'label' and 'message' not found in the CSV.")
    exit()

# Step 6: Check for missing values
if df.isnull().values.any():
    print("⚠️ WARNING: Dataset contains missing values. Please clean the file.")
    exit()

# Step 7: Convert labels to numbers (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 8: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 9: Convert text messages into vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 10: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 11: Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {round(accuracy * 100, 2)}%")

# Step 12: Predict some sample messages
print("\n📨 Sample Predictions:")
sample_messages = [
    "Congratulations! You've won a free ticket to Bahamas! Call now!",
    "Hi, can we meet tomorrow at 5pm?",
    "Get your free ringtone by texting WIN to 12345",
    "Don't forget to bring your notebook to the meeting."
]

for msg in sample_messages:
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"> {msg} → {label}")
# Step 13: User input for prediction
print("\n🧪 Test Your Own Message:")
user_msg = input("Enter your message: ")
user_vec = vectorizer.transform([user_msg])
user_pred = model.predict(user_vec)[0]
result = "SPAM" if user_pred == 1 else "HAM"
print(f"🔎 Prediction: {result}")


