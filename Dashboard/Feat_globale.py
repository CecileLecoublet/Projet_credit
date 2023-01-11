import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
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
    explainer = shap.Explainer(load_model().predict, X_test_scaled)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer.shap_values(X_test_scaled)
    # Evaluate SHAP values
    st.write(shap_values)
    