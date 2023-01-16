import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
import pickle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

def feat_cat(X_test, num, X_test_scaled, shap_values):
    feat = feat_imp(X_test_scaled, shap_values)
    nom = X_test[feat['col_name'].to_list()]
    col = nom.select_dtypes(include = 'object')
    colonne = st.selectbox("Valeur catégorielle : ", col.columns)
    occ = X_test[X_test["SK_ID_CURR"] == num]
    liste = X_test[colonne].value_counts().keys()
    liste = liste.sort_values()
    if len(liste) == 2:
        st.write(colonne, " : ", occ[colonne].values[0])
    taille = len(np.unique(X_test[colonne]))
    zeros = np.zeros(taille)
    j = 0
    for i in liste:
        if occ[colonne].values == i:
            break
        j = j + 1
    zeros[j] = 1
    test = X_test.sort_values(by = colonne)
    test = test[colonne]
    val = test.value_counts(sort = False)
    fig = go.Figure(data=[go.Pie(labels = liste,
                            values = val.values,
                            pull = zeros,
                            insidetextorientation='radial')])
    fig.update_layout(autosize=False, width=900, height=600,)
    fig.update_traces(marker=dict(line=dict(width=0.5)))
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=15, uniformtext_mode='hide')
    fig.update_layout(showlegend=True,
                         height=800,
                         margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
    st.plotly_chart(fig, use_container_width=True)

# Statut globale
# Cinquième chapitre
def fc_global(X_test_scaled, X_test, X_train_scaled, nom, numero) :
    st.markdown("## Cinquième chapitre : Features global et features local")
    st.markdown("### 5.1 : Features local")
    # Entraînement
    explainer = shap.Explainer(load_model())
    # Calculates the SHAP values - It takes some time
    shap_values = explainer.shap_values(X_test_scaled)
    # Evaluate SHAP values
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.summary_plot(shap_values[0], X_test_scaled, plot_type="bar")
    st.pyplot(fig)
    # Nuage graphique, choix des colonnes
    st.markdown("### 5.2 : Shap dependence plot")
    imp_cols = X_train_scaled.abs().mean().sort_values(ascending=False).index.tolist()
    j = 0
    X_train = X_train_scaled[0:122]
    shap_values_train = explainer.shap_values(X_train_scaled)
    position_1, choix, choix_2 = choix_var(X_test_scaled, shap_values, nom)
    #plot the top var and color by the 2nd var
    i = X_test_scaled[X_test_scaled["SK_ID_CURR"] == numero].index[0]
    fig = dep_plt(i, choix, choix_2, 
    X_train, 
    shap_values_train[0][i],
    X_test_scaled.iloc[j,:][imp_cols[i]], 
    shap_values[0][position_1][i])
    st.markdown("### 5.3 : Graphique catégorielle")
    feat_cat(nom, numero, X_test_scaled, shap_values)