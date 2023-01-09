import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Traduction des métiers en français
# Deuxième chapitre chapitre
def metier_client(tab_1) :
    # Nom de statut en anglais
    statut_en = ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff',
                'Cleaning staff', 'Cooking staff', 'Private service staff', 'Medicine staff',
                'Security staff', 'High skill tech staff', 'Waiters/barmen staff',
                'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff']
    # Nom de statut en français
    statut_fr = ['Ouvriers', 'Personnel de base', 'Comptables', 'Managers', 'Conducteurs', 'Personnel de vente',
                'Personnel de nettoyage', 'Personnel de cuisine', 'Personnel du service privé', 'Personnel médical',
                'Personnel de sécurité', 'Personnel technique hautement qualifié', 'Personnel de serveurs/barmans',
                'Ouvriers peu qualifiés', 'Agents immobiliers', 'Secrétaires', 'Le personnel informatique', 'RH']
    # Sélection du satut en français
    for i in range(0, len(statut_en), 1):
        if tab_1["OCCUPATION_TYPE"].values == i:
            st.write("Le client travaille dans :", statut_fr[i])

# Age du client et appelle nom client pour avoir le métier
# Deuxième chapitre
def age_client(tab_1):
    st.markdown("## Deuxième chapitre : Statut du client")
    # Sélection âge, réduction du nombre et "age.values[0]" pour que l'âge tienne sur une ligne
    age = tab_1['DAYS_BIRTH'].round(0)
    st.write("L'âge du client est : ", age.values[0])
    metier_client(tab_1)