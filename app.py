import streamlit as st
import pickle

# Muat model dan vectorizer yang sudah disimpan
with open('naive_bayes_model.pkl', 'rb') as nb_file:
    nb_model = pickle.load(nb_file)

with open('svm_model.pkl', 'rb') as svm_file:
    svm_model = pickle.load(svm_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Judul aplikasi
st.title('Sentiment Analysis Web App')

# Teks untuk prediksi
text_input = st.text_area("Masukkan teks untuk analisis sentimen", "")

# Fungsi untuk melakukan prediksi
def predict_sentiment(text):
    # Preprocessing dan transformasi teks
    text_vec = vectorizer.transform([text])
    
    # Prediksi dengan Naive Bayes
    nb_pred = nb_model.predict(text_vec)
    nb_pred_label = 'Positif' if nb_pred[0] == 1 else 'Negatif'
    
    # Prediksi dengan SVM
    svm_pred = svm_model.predict(text_vec)
    svm_pred_label = 'Positif' if svm_pred[0] == 1 else 'Negatif'
    
    return nb_pred_label, svm_pred_label

# Tombol untuk melakukan analisis
if st.button("Analisis Sentimen"):
    if text_input:
        nb_result, svm_result = predict_sentiment(text_input)
        st.write(f"**Prediksi dengan Naive Bayes**: {nb_result}")
        st.write(f"**Prediksi dengan SVM**: {svm_result}")
    else:
        st.write("Silakan masukkan teks terlebih dahulu untuk analisis.")
