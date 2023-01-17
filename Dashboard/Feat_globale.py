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

# Chargement du modèle LGBMClassifier
def load_model():
    pickle_in = open("mlflow_model/model.pkl","rb")
    clf = pickle.load(pickle_in)
    return clf

# Affichage des 20 features les plus importantes
def shap_model(X_test_scaled):
    # Entraînement
    explainer = shap.Explainer(load_model())
    # Calculates the SHAP values - It takes some time
    shap_values = explainer.shap_values(X_test_scaled)
    # Evaluate SHAP values
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.summary_plot(shap_values[0], X_test_scaled, plot_type="bar")
    st.pyplot(fig)
    return(explainer, shap_values)

# Shap la liste des features des plus importantes
# On récupère les 20 features importantes
def feat_imp(X_test_scaled, shap_values):
    feature_names = X_test_scaled.columns
    rf_resultX = pd.DataFrame(shap_values[0], columns = feature_names)
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    return shap_importance.head(20)

# Choix du features non catégorielle pour le shap et division des data
# Préparation pour le graphique
def choix_var(X_test_scaled, shap_values, nom):
    st.markdown("Sur le graphique du dessus, on peut voir les features les plus importantes\
                de manière local. Dans le graphique du dessous, on peut la position du client\
                choisit par rapport aux autres.")
    feat = feat_imp(X_test_scaled, shap_values)
    nom = nom[feat['col_name'].to_list()]
    nom = nom.select_dtypes(exclude = 'object')
    choix = st.selectbox("Choix des features les plus importantes", nom.columns)
    position = X_test_scaled.columns.get_loc(choix)
    quest = st.radio("Ajout d'une colonne ? ", ('Non', 'Oui'))
    if quest == 'Non':
        return position, choix, choix
    else:
        choix_2 = st.selectbox("Choix des features les plus importantes", nom.columns[nom.columns != choix])
        return position, choix, choix_2

# Shap garphique nuage de points
def dep_plt(col, color_by, base_actual_df, base_shap_df, overlay_x, overlay_y):
    cmap=sns.diverging_palette(260, 10, sep=1, as_cmap=True) #seaborn palette
    f, ax = plt.subplots()
    points = ax.scatter(base_actual_df[col], base_shap_df, c=base_actual_df[color_by], s=20, cmap=cmap)
    f.colorbar(points).set_label(color_by)
    ax.scatter(overlay_x, overlay_y, color='black', s=50)
    plt.xlabel(col)
    plt.ylabel("SHAP value for " + col)
    st.pyplot(f)

# Préparation deplusieurs shap pour la créaion du nuage de point
def prep_dep(X_train_scaled, X_test_scaled, explainer, shap_values, nom, numero):
    imp_cols = X_train_scaled.abs().mean().sort_values(ascending=False).index.tolist()
    j = 0
    shap_values_train = explainer.shap_values(X_train_scaled)
    test = shap_values_train[0].T
    position_1, choix, choix_2 = choix_var(X_test_scaled, shap_values, nom)
    #plot the top var and color by the 2nd var
    i = X_test_scaled[X_test_scaled["SK_ID_CURR"] == numero].index[0]
    dep_plt(choix, choix_2, X_train_scaled, 
            test[position_1], X_test_scaled.iloc[j,:][imp_cols[i]], 
            shap_values[0][position_1][i])

# Shap variable catégorielle, affichage en go.Pie
def feat_cat(X_test, num, X_test_scaled, shap_values):
    # Récupération des variables catégorielles des features les plus importantes
    feat = feat_imp(X_test_scaled, shap_values)
    nom = X_test[feat['col_name'].to_list()]
    col = nom.select_dtypes(include = 'object')
    # Choix des features à afficher
    colonne = st.selectbox("Valeur catégorielle : ", col.columns)
    # Divison du tableau selon le client et la catégories
    occ = X_test[X_test["SK_ID_CURR"] == num]
    # Récupration de toutes les noms de la features et on les mets dans l'ordre alphabétique
    liste = X_test[colonne].value_counts().keys()
    liste = liste.sort_values()
    # Le graphique est peu visible si les noms est égales à 2
    # Le nom est écrit pour plus de clarté
    if len(liste) == 2:
        st.write(colonne, " : ", occ[colonne].values[0])
    # On réupère la taille unique des noms dans la variable et on créait un vecteur nul
    taille = len(np.unique(X_test[colonne]))
    zeros = np.zeros(taille)
    # On récupère la position de la variable pour pouvoir mettre un 1 dans le vecteur de zéros
    # Ainsi voir à quel catégorie appartient le client
    j = 0
    for i in liste:
        if occ[colonne].values == i:
            break
        j = j + 1
    zeros[j] = 1
    # Le compteur est mis en ordre alphabétique, pas du plus grand au plus petit
    test = X_test.sort_values(by = colonne)
    test = test[colonne]
    val = test.value_counts(sort = False)
    # Graphique camembert
    fig = go.Figure(data=[go.Pie(labels = liste,
                            values = val.values,
                            pull = zeros,
                            insidetextorientation='radial')])
    fig.update_traces(marker=dict(line=dict(width=0.5)))
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=15, uniformtext_mode='hide')
    fig.update_layout(showlegend=True,
                         height=600,
                         margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
    fig.update_layout(legend=dict(yanchor="top",
                          y=0.99, xanchor="left", x=0.99),
                      barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

# Statut globale
# Cinquième chapitre
def fc_global(X_test_scaled, X_test, X_train_scaled, nom, numero) :
    st.markdown("## Cinquième chapitre : Features global et features local")
    st.markdown("### 5.1 : Features local")
    # Chargement du modèle shap
    explainer, shap_values = shap_model(X_test_scaled)
    # Nuage graphique, choix des colonnes
    st.markdown("### 5.2 : Shap dependence plot")
    prep_dep(X_train_scaled, X_test_scaled, explainer, shap_values, nom, numero)
    st.markdown("### 5.3 : Graphique catégorielle")
    feat_cat(nom, numero, X_test_scaled, shap_values)