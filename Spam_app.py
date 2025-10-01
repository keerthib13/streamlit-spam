import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
url = "https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv"
df = pd.read_csv(url, encoding="latin-1")

# Keep only useful columns and rename
df = df.rename(columns={"v1": "Category", "v2": "Message"})
df = df[["Category", "Message"]]   # drop extra unnamed columns

# Add spam column (1 = spam, 0 = ham)
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['spam'], test_size=0.2, random_state=42
)

# Pipeline (vectorizer + model)
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

# ---------------- Streamlit UI ----------------
st.title("📩 Spam Message Detector")
st.write("Type a message below and check if it's spam or not!")

# User input
user_input = st.text_area("✍️ Type your message here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        prediction = clf.predict([user_input])[0]
        prob = clf.predict_proba([user_input])[0]

        if prediction == 1:
            st.error(f"🚨 This looks like **SPAM** (Probability: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ This looks like **HAM** (Not spam) (Probability: {prob[0]*100:.2f}%)")

# Sidebar info
st.sidebar.header("📊 Model Information")
st.sidebar.write(f"Accuracy on test set: {clf.score(X_test, y_test)*100:.2f}%")


