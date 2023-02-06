import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

# Traduction des métiers en français
# Deuxième chapitre chapitre
def metier_client(tab, num, metier) :
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
    k = 0
    for i in statut_en:
        if metier.values == i:
            break
        k = k + 1
    
    return(st.write("Le métier du client est : " , statut_fr[k]))

# Age du client et appelle nom client pour avoir le métier
# Deuxième chapitre
def age_client(tab, num, age, metier):
    st.markdown("## Deuxième chapitre : Statut du client")
    st.markdown("Les informations de bases des clients : l'âge et sa catégorie de travail.")
    # Sélection âge, réduction du nombre et "age.values[0]" pour que l'âge tienne sur une ligne
    st.write("L'âge du client est : ", age.values[0])
    metier_client(tab, num, metier)