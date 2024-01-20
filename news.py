import streamlit as st
import joblib

# Load the pre-trained model and vectorizer
naive_bayes_classifier, tfidf_vectorizer = joblib.load(r'C:\Users\THEBEST\Desktop\Jupyter_workbook\news_model.pkl')

# Set the background image URL
background_image_url = 'https://as2.ftcdn.net/v2/jpg/02/30/03/09/1000_F_230030937_XyvVuqjTbbiNIUVDXsB25i7CKkyaFyUc.jpg'

# Use CSS to set the background image
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and header
st.title("News Topic Classification App")
st.header("Enter a news article and click 'Classify' to get the predicted topic.")

# User input for text classification
user_input = st.text_area("Enter a news article:", "Type here...")

if st.button("Classify"):
    # Transform user input using the pre-trained vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Make predictions
    prediction = naive_bayes_classifier.predict(user_input_tfidf)[0]
    
    st.success(f"The predicted topic is: {prediction}")