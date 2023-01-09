# Librairies de bases
import pandas as pd
import numpy as np

# Graphique
import matplotlib.pyplot as plt
import seaborn as sns


##########################################################

def tauxRemplissage(df):
    """
    Entrée : DataFrame
    Objectifs : Calcul du taux de remplissage, 1 beaucoup de données, 0 beaucoup de valeurs manquantes
    Sorties : DataFrame
    """
        
    val = 1 - df.isna().sum()/len(df)
    val = pd.DataFrame(val)
    val = val.transpose()
    return val


##########################################################

def graphRemplissage(df) :
    """
    Entrée : Dataframe
    Objectifs : Récupérer la DataFrame du taux de remplissage et afficher le graphique du taux de remplissage
    Sortie : Graphe
    """

    df = tauxRemplissage (df)
    fig,ax = plt.subplots()
    ax.tick_params(axis='x', rotation=90)
    sns.barplot( data=df, color = 'blue')
    plt.xlabel('\nColonnes')
    plt.ylabel('Taux de valeurs\n')
    plt.title("Taux de remplissage")
    plt.show()
    
##########################################################