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
    explainer = shap.Explainer(load_model())
    # Calculates the SHAP values - It takes some time
    shap_values = explainer.shap_values(X_test_scaled)
    # Evaluate SHAP values
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.summary_plot(shap_values[0], X_test_scaled, plot_type="bar")
    st.pyplot(fig)
    fig_1 = shap.dependence_plot('AMT_GOODS_PRICE', shap_values[1], X_test_scaled, interaction_index="AMT_GOODS_PRICE",alpha=0.4,show=True)
    # plt.title("Rente depence plot",loc='left',fontfamily='serif',fontsize=15)
    # plt.ylabel("SHAP value for the 'AMT_GOODS_PRICE' feature")
    st.pyplot(fig_1)
    
    