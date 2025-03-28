import streamlit as st
import pandas as pd
import joblib
import nest_asyncio
import asyncio
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from utility import vectorize
from utility import clean
import nltk
from utility import tfidf_vect_fit

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

st.header("Bienvenu(e) dans cet démo d'un analyseur de sentiment avec randomForest")

form = st.form(key="analyseur")
user_input = form.text_area("entrez le commentaire")
submit = form.form_submit_button('Validez')

if submit:

    model = joblib.load("my_random_forest.joblib")

    async def translate_text(text, dest):
        translator = Translator()
        try:
            translated = await translator.translate(text, dest=dest)  # Await the coroutine
        except:
            translated = translator.translate(text, dest=dest)
        return translated.text  # Access the translated text

    translated_text = asyncio.run(translate_text(user_input, 'en'))

    #translated_text_cleaned = clean(translated_text)
    
    print(translated_text)
    
    st.write(f"texte traduit {translated_text}")

    exemple_traduit_predit = model.predict(vectorize(pd.Series(translated_text),tfidf_vect_fit))
    
    print(exemple_traduit_predit)

    if exemple_traduit_predit == 0:
        st.write("ce commentaire est négatif")
    else:
        st.write("ce commentaire est positif")

    st.write("merci d'avoir testé cet analyseur de sentiment")