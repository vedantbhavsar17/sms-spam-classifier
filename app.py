import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .main {
            background-color: #0e1117;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4FFF50;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #bbbbbb;
            margin-bottom: 20px;
        }
        .stTextArea textarea {
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">📩 Spam SMS Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message is Spam or Not using Machine Learning</div>', unsafe_allow_html=True)

def test_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum:
            y.append(i)

    text = y[:]
    y = []

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation and i.isalnum():
            y.append(i)

    text = []
    for i in y:
        text.append(ps.stem(i))
        
    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Input box
input_sms = st.text_area("✍️ Enter your message below:", height=150)

if st.button('🚀 Predict'):
    transform_text = test_preprocessing(input_sms)
    vector_input = tfidf.transform([transform_text])
    result = model.predict(vector_input)[0]

    st.markdown("---")

    # Result Display with Styling
    if result == 1:
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT Spam")

# Footer
st.markdown("---")
st.markdown("<center style='color: gray;'>Built By Vedant Bhavsar with ❤️ using Streamlit</center>", unsafe_allow_html=True)