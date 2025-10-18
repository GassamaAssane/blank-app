import streamlit as st

# Vérifie si l'utilisateur appelle le test de santé
if "ping" in st.query_params:
    st.write("OK")
    st.stop()  # Arrête l'exécution ici
