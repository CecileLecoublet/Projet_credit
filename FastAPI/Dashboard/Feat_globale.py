import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import lime
from lime import lime_tabular
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Chargement du modèle
def load_model():
    pickle_in = open("../mlflow_model/model.pkl","rb")
    clf = pickle.load(pickle_in)
    return clf

# Statut globale
# Cinquième chapitre
def fc_global(X_test_scaled, X_train_scaled, choix) :
    st.markdown("## Cinquième chapitre : Features global et features local")
    # Entraînement
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_scaled),
        feature_names=X_train_scaled.columns,
        class_names=['Positif', 'Negatif'],
        mode='classification')
    # On récupère la ligne client
    index = X_test_scaled[X_test_scaled["SK_ID_CURR"] == choix].index
    # Sélection des features
    exp = explainer.explain_instance(
        data_row = X_test_scaled.iloc[index[0]], 
        predict_fn = load_model().predict_proba)
    # Affichage sur un graphique
    exp.as_pyplot_figure()
    st.pyplot()
    plt.clf()
    exp.show_in_notebook(show_table=True)
    # Affichage du tableau
    st.write(pd.DataFrame(exp.as_list()))
    