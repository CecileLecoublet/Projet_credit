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

# Choix du features pour le shap et division des data
def choix_var(X_test_scaled, shap_values, nom):
    st.markdown("Sur le graphique du dessus, on peut voir les features les plus importantes\
                de manière local. Dans le graphique du dessous, on peut la position du client\
                choisit par rapport aux autres.")
    feat = feat_imp(X_test_scaled, shap_values)
    nom = nom[feat['col_name'].to_list()]
    nom = nom.select_dtypes(exclude = 'object')
    choix = st.selectbox("Choix des features les plus importantes", nom.columns)
    position = X_test_scaled.columns.get_loc(choix)
    quest = st.radio("Ajout d'une colonne ? : ", ('Oui', 'Non'))
    if quest == 'Oui':
        return position, choix, choix
    else:
        choix_2 = st.selectbox("Choix des features les plus importantes", nom.columns[nom.columns != choix])
        return position, choix, choix_2

# Shap garphique
def dep_plt(i, col, color_by, base_actual_df, base_shap_df, overlay_x, overlay_y):
    cmap=sns.diverging_palette(260, 10, sep=1, as_cmap=True) #seaborn palette
    f, ax = plt.subplots()
    points = ax.scatter(base_actual_df[col], base_shap_df, c=base_actual_df[color_by], s=20, cmap=cmap)
    f.colorbar(points).set_label(color_by)
    ax.scatter(overlay_x, overlay_y, color='black', s=50)
    plt.xlabel(col)
    plt.ylabel("SHAP value for " + col)
    st.pyplot(f)

# Shap la liste des features des plus importantes
def feat_imp(X_test_scaled, shap_values):
    feature_names = X_test_scaled.columns
    rf_resultX = pd.DataFrame(shap_values[0], columns = feature_names)
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    return shap_importance.head(20)

# Statut globale
# Cinquième chapitre
def fc_global(X_test_scaled, X_test, X_train_scaled, nom, numero) :
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
    position_1, choix, choix_2 = choix_var(X_test_scaled, shap_values, nom)
    #plot the top var and color by the 2nd var
    st.write(numero)
    i = X_test_scaled[X_test_scaled["SK_ID_CURR"] == choix].index[0]
    st.write(i)
    fig = dep_plt(i, choix, choix_2, 
    X_train, 
    shap_values_train[0][i],
    X_test_scaled.iloc[j,:][imp_cols[i]], 
    shap_values[0][position_1][i])