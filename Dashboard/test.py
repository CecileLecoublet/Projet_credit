import shap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from Feat_globale import load_model
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

def fesatures_importante(X_test, X_test_scaled, X_train_scaled, choix) :
    st.checkbox("Customer ID {:.0f} feature importance ?".format(choix))
    shap.initjs()
    X = X_train_scaled.iloc[:, :-1]
    X = X[X.index == choix]
    number = st.slider("Pick a number of features…", 0, 20, 5)

    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(load_model())
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
    st.pyplot(fig)

    if st.checkbox("Need help about feature description ?") :
        list_features = X_test_scaled.index.to_list()
        feature = st.selectbox('Feature checklist…', list_features)
        st.table(X_test_scaled.loc[X_test_scaled.index == feature][:1])