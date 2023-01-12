import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
import pickle
import seaborn as sns
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Chargement du modèle
def load_model():
    pickle_in = open("mlflow_model/model.pkl","rb")
    clf = pickle.load(pickle_in)
    return clf

# Shap garphique
def dep_plt(i, col, color_by, base_actual_df, base_shap_df, overlay_x, overlay_y):
    cmap=sns.diverging_palette(260, 10, sep=1, as_cmap=True) #seaborn palette
    f, ax = plt.subplots()
    points = ax.scatter(base_actual_df[col], base_shap_df, c=base_actual_df[color_by], s=20, cmap=cmap)
    f.colorbar(points).set_label(color_by)
    ax.scatter(overlay_x, overlay_y, color='black', s=50)
    plt.xlabel(col)
    plt.ylabel("SHAP value for " + col)

# Statut globale
# Cinquième chapitre
def fc_global(X_test_scaled, X_test, X_train_scaled, choix) :
    st.markdown("## Cinquième chapitre : Features global et features local")
    # Entraînement
    explainer = shap.Explainer(load_model())
    # Calculates the SHAP values - It takes some time
    shap_values = explainer.shap_values(X_test_scaled)
    # Evaluate SHAP values
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.summary_plot(shap_values[0], X_test_scaled, plot_type="bar")
    st.pyplot(fig)
    imp_cols = X_train_scaled.abs().mean().sort_values(ascending=False).index.tolist()
    j = 0
    X_train = X_train_scaled[0:122]
    shap_values_train = explainer.shap_values(X_train_scaled)
    for i in range(0, len(imp_cols)):
        #plot the top var and color by the 2nd var
        if i == 0 : 
            fig = dep_plt(i, imp_cols[i], imp_cols[0], 
            X_train, 
            shap_values_train[0][i],
            X_test_scaled.iloc[j,:][imp_cols[i]], 
            shap_values[0][i][0])
    st.write(fig)

    