import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("📰 Fake News Detection")
st.write("Paste a news article or headline below to check if it's Fake or Real.")

# Input from user
user_input = st.text_area("Enter News Content:")

if st.button("Check"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        if prediction == 0:
            st.error("🚨 This news is **FAKE**!")
        else:
            st.success("✅ This news is **REAL**!")
    else:
        st.warning("Please enter some text.")

