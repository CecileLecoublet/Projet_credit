import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import lime
from lime import lime_tabular
import pickle
import re
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Chargement du modèle
def load_model():
    pickle_in = open("mlflow_model/model.pkl","rb")
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
    # Préparation du tableau
    results = []
    for i in range(0, 10, 1):
        test = re.split(r'< |> |>= |<= |\s+', exp.as_list()[i][0])
        if len(test) == 3 :
            tab = [test[0], round(float(X_test_scaled[test[0]].min()), 2), float(test[2]), round(exp.as_list()[i][1], 2)]
        if len(test) == 5 :
            tab = [test[2], float(test[0]), float(test[4]), round(exp.as_list()[i][1], 2)]
        results.append(tab)
    tableau = pd.DataFrame(results)
    tableau.columns = ["Features", "Val_min", "Val_max", "Val"]
    # Affichage du graphique
    fig = px.scatter(tableau, x = "Features", y = ["Val_min", "Val_max", "Val"])
    st.write(fig)
    