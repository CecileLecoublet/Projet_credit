import requests
import streamlit as st
import pandas as pd
from Age_client import age_client
from Statut_client import graphique
from Note_ext import quatrieme_chapitre
from Feat_globale import load_model, fc_global
import plotly.express as px
from matplotlib import pyplot as plt
import lime
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Ouverture des fichiers
def ouverture_data() :
    tab = pd.read_csv("X_test.csv")
    read_and_cache_csv = st.cache(pd.read_csv)
    df = read_and_cache_csv("X_test_scaled.csv")
    X_train_scaled = read_and_cache_csv("X_train_scaled.csv")
    nom = read_and_cache_csv("nom.csv")
    return tab, df, X_train_scaled, nom

# Choix du client et division des data
def choix_client():
    st.markdown("## Premier chapitre : Statut du crédit client")
    # Sélection du client et division des data
    choix = st.selectbox("Choix du client", df["SK_ID_CURR"])
    data = df[df["SK_ID_CURR"] == choix]
    tab_1 = tab[tab["SK_ID_CURR"] == choix]
    return data, tab_1, choix

# Fonction qui fait un lien avec le FASTAPI
# Client est à risque ou pas
def client_api(df):
    client = df["SK_ID_CURR"]
    df_json = client.to_json(orient='records')
    payload = df_json.strip("[]")
    headers = {
        'Content-Type' : 'application/json'
    }
    url = "https://projet7credit.herokuapp.com/predict?data="+payload
    response = requests.post(url)
    if response.json() == 0 :
        rep = 0
        st.success("Le client n'est pas à risque")
    else :
        rep = 1
        st.error("Le client est à risque")
    return rep

# Appel de toute les fonctions
if __name__ == '__main__':

    # Titre du document
    st.title('Dashboard : Prédiction de crédit')

    # Ouverture des data
    tab, df, X_train_scaled, nom = ouverture_data()
    df = df.sort_values("SK_ID_CURR")
    df = df.reset_index(drop=True)

    # Premier chapitre
    # Choix du client
    data, tab_1, choix = choix_client()

    # FastAPI et client à risque
    rep = client_api(data)

    # Deuxième chapitre
    # Âge et métier
    age = tab_1["DAYS_BIRTH"].round(0)
    metier = nom[nom["SK_ID_CURR"] == choix]
    age_client(nom, choix, age, metier["OCCUPATION_TYPE"])

    # Troisème chapitre
    # Information client credit    
    graphique(tab_1)

    # Quatreième chapitre
    # Note extérieure
    quatrieme_chapitre(tab_1, choix)

    # Cinquième chapitre
    # Information features gloable
    fc_global(df, tab, X_train_scaled, nom, choix)
