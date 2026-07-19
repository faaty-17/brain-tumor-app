import streamlit as st
import requests
from PIL import Image

st.title("🧠 Brain Tumor Detection App")
st.write("Architecture Microservices (FastAPI Backend + Streamlit Frontend)")

# URL mta3 el-API Backend local tawa
API_URL = "http://127.0.0.1:8000/predict"

# 1. Sélection du modèle
model_choice = st.selectbox("Choisissez le modèle :", ["Densenet121", "ResNet50"])
model_param = "densenet" if model_choice == "Densenet121" else "resnet"

# 2. Upload de l'image
uploaded_file = st.file_uploader("Téléchargez une image d'IRM...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image IRM", width=400)
    
    if st.button("Lancer la Détection"):
        with st.spinner("Analyse en cours... 🔄"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"model_choice": model_param}
                
                response = requests.post(API_URL, files=files, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    
                    st.success("🎯 Analyse terminée !")
                    if prediction == "No Tumor":
                        st.success(f"✅ Résultat : **{prediction}**")
                    else:
                        st.error(f"⚠️ Résultat : **{prediction}**")
                else:
                    st.error(f"❌ Erreur API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Connexion impossible au Backend: {e}")