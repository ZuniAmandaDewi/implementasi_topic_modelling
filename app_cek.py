import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

Data, Ekstraksi, Model = st.tabs(['Data', 'Preprocessing', 'Modelling'])

data=pd.read_csv('crawling_pta_labeled.csv')

def preproses(inputan):
         clean_tag = re.sub('@\S+','', inputan)
         clean_url = re.sub('https?:\/\/.*[\r\n]*','', clean_tag)
         clean_hastag = re.sub('#\S+',' ', clean_url)
         clean_symbol = re.sub('[^a-zA-Z]',' ', clean_hastag)
         token=word_tokenize(clean_symbol)
         listStopword = set(stopwords.words('indonesian')+stopwords.words('english'))
         stopword=[]
         for x in (token):
            if x not in listStopword:
               stopword.append(x)
         joinkata = ' '.join(stopword)
         return clean_symbol,token,stopword,joinkata


with Data:
    st.title('Input Data')
    inputan = st.text_input('Masukkan Abstrak')
    submit1 = st.button("Submit")
    if submit1 :
      st.subheader('Abstrak yang Anda masukkan :')
      st.write(inputan)

with Ekstraksi:
   st.title('Preprocessing & Ekstraksi Fitur')
   submit2 = st.button("Lakukan Preprocessing")
   if submit2 :
        if inputan == "":
            st.error('Abstrak belum diisi', icon="🚨")
        else:
            clean_symbol,token,stopword,joinkata = preproses(inputan)
            st.subheader('Preprocessing')
            st.write('Cleansing :')
            clean_symbol
            st.write("Tokenisasi :")
            token
            st.write("Stopword :")
            stopword

            with open('tf.sav', 'rb') as file:
                vectorizer = pickle.load(file)

            hasil_tf = vectorizer.transform([joinkata])
            tf_name=vectorizer.get_feature_names_out()
            tf_array=hasil_tf.toarray()
            df_tf= pd.DataFrame(tf_array, columns = tf_name)
            st.write("Term Frequency :")
            df_tf

            st.session_state['df_tf'] = df_tf
            st.session_state['preprocessing'] = True


with Model:
    st.title('Topic Modelling')
    st.write ("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    met2 = st.checkbox("Naive Bayes")
    met3 = st.checkbox("Decesion Tree")
    submit3 = st.button("Pilih")
    if submit3 :
        if 'preprocessing' not in st.session_state:
            st.error('Anda belum melakukan preprocessing', icon="🚨")
        else:
            if 'df_tf' in st.session_state:
                df_tf = st.session_state["df_tf"]
            if met1 :
                st.subheader("Metode yang Anda gunakan Adalah KNN")

                with open('lda_knn.sav', 'rb') as file:
                    lda1 = pickle.load(file)

                hasil_lda1=lda1.transform(df_tf)   
        
                with open('knn.sav', 'rb') as file:
                    knn = pickle.load(file)
                
                hasil_knn=knn.predict(hasil_lda1)
                hasil =f"Berdasarkan data yang Anda masukkan, maka abstrak masuk dalam kategori  : {hasil_knn}"
                st.success(hasil)


            elif met2 :
                st.subheader("Metode yang Anda gunakan Adalah Naive Bayes")

                with open('lda_nb.sav', 'rb') as file:
                    lda2 = pickle.load(file)

                hasil_lda2=lda2.transform(df_tf)   
            
                with open('nb.sav', 'rb') as file:
                    nb = pickle.load(file)
                
                hasil_nb=nb.predict(hasil_lda2)
                hasil =f"Berdasarkan data yang Anda masukkan, maka abstrak masuk dalam kategori  : {hasil_nb}"
                st.success(hasil)
            elif met3 :
                st.subheader("Metode yang Anda gunakan Adalah Decesion Tree")

                with open('lda_dt.sav', 'rb') as file:
                    lda3 = pickle.load(file)

                hasil_lda3=lda3.transform(df_tf)   
                
                with open('dt.sav', 'rb') as file:
                    dt = pickle.load(file)
                
                hasil_dt=dt.predict(hasil_lda3)
                hasil =f"Berdasarkan data yang Anda masukkan, maka abstrak masuk dalam kategori  : {hasil_dt}"
                st.success(hasil)
            else :
                st.subheader("Anda Belum Memilih Metode")
            

   
