import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Note extérieure
# Quatrième chapitre
def quatrieme_chapitre(data, val):
    st.markdown("## Quatrième chapitre : Note Appartement / Ville / Régions")
    st.write("Score normalisé à partir d'une source de données externe,\
            ce sont des notes importantes qui ont un poids pour savoir si le client est à risque ou pas")
    # Construction tableau
    prov = data[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
    prov.rename(index = {0: 'Note'}, inplace = True) 
    # Seuil
    seuil = pd.DataFrame([0.50,0.54,0.53])
    # Renommer ligne et colonne
    seuil.index = prov.columns
    seuil = seuil.T
    seuil.rename(index = {0: 'Seuil'}, inplace = True) 
    # Fusion
    prov = prov.append(seuil)
    prov = prov.T
    prov = prov.reset_index()
    prov.columns = ["Noms", "Note", "Seuil"]
    # Affichage des graphiques
    bar_chart = px.bar(prov, x='Noms', y='Note', color="Noms", color_discrete_sequence=["yellow", "red", "green"])
    bar_chart.add_traces(go.Scatter(x= prov.Noms, y=prov.Seuil, mode = 'lines', name="Seuil", marker=dict(color='blue')))
    bar_chart.update_layout(yaxis=dict(range=[0,1]))
    st.write(bar_chart)