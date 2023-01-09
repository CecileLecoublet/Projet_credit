import streamlit as st
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Graphique sur le salaire, crédit, rentes et prix des bien
# Troisième chapitre
def graphique(data):
    st.markdown("## Troisième chapitre : Information sur le crédit")
    # Sélection des features
    prov = data[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]].copy()
    # Traduction des noms
    prov.rename(columns={'AMT_INCOME_TOTAL': 'Salaire', 'AMT_CREDIT': 'Crédit',
                        'AMT_ANNUITY': 'Rente', "AMT_GOODS_PRICE": "Prix des biens"}, inplace=True)
    # Affichage graphique
    bar_chart = px.bar(prov, barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
    st.write(bar_chart)