import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
v = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl','rb'))
load_model = pickle.load(open('model.pkl','rb'))

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]',' ',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text

def fake_news(news):
    news = word_drop(news)
    ##
    # Check if the input becomes empty after preprocessing
    if not news or news.isspace(): 
        return "invalid"
    ##
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title("Fake News Predection")
    #st.subheader("By Using  LogisticRegression Algorithm")
    sentance = st.text_area("Enter Here",height = 100)
    predict_btn = st.button("Predict")
    
    if predict_btn:
        #if predict_btn:

        # Check if input is empty or contains only whitespace
        if not sentance.strip():  
            st.warning("Please enter some text to make a prediction!")
        else:
            prediction_class = fake_news(sentance) 
            print(prediction_class)
            ##
            if prediction_class == "invalid":
                st.warning("Your input contains only special characters \n or invalid text.Please enter proper text.")
            ##
            elif  prediction_class == [1]:
                st.success(" It's Real News ")
            if  prediction_class == [0]:
                st.warning(" It's a FaKe News ")
    






##     cd "C:\Users\G.S.V.Mohan Kadari\Desktop\Brain O Vision\Project\Programing_Files"
##     streamlit run app.py