# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:03:55 2021

@author: 7409216W
"""

import streamlit as st
st.set_page_config(layout="wide")

import requests, zipfile, io
import streamlit as st
import numpy as np
import pandas as pd
import datetime
#from time import time
from datetime import timedelta
import matplotlib.pyplot as plt

#import geopandas as gpd

#import plotly.graph_objects as go

import seaborn as sns
import os
from os import getcwd, chdir, mkdir
from sklearn import preprocessing
import altair as alt
import pydeck as pdk
from sklearn.metrics import accuracy_score, precision_score, recall_score

today = datetime.datetime.now()
tomorrow = today + timedelta(days=1)


#import streamlit.components.v1 as components

#_strmlt = components.declare_component(
#   "strmlt", url="http://localhost:3001",
#)

#def strmlt(data, key=None):
#    return _strmlt(data=data, key=key, default=pd.DataFrame())

#définition des répertoires de travail :
path = os.getcwd() #répertoire du fichier actuel
#path = os.chdir("..") #pour remonter dans l'arborescence
#path = os.chdir("..") #pour remonter dans l'arborescence

print(path)
#répertoire courant
#mon_repertoire_courant = path + "Data DNCF teams\"
#répertoire courant
mon_repertoire_courant = r"C:\Users\7409216W\projet_data_scientist\Data DNCF teams"

#répertoire tampon (pouvant être supprimés)
mon_repertoire_tampon = r"C:\Users\7409216W\projet_data_scientist\Data DNCF teams\tampon"
#répertoire dans lequel seront stockés les open data reconstitués en base train (servant à comprendre le nombre de circulation)
mon_repertoire_opendata = r"C:\Users\7409216W\projet_data_scientist\Data DNCF teams\export gfts"


os.chdir(mon_repertoire_courant)

page = st.sidebar.radio(label = "", options = ['Présentation',
                                               'Analyse Base Contrôleurs',
                                'Rendus du programme',
                                'Base Prédite'])

df = pd.read_csv('2021 10 20 bc 2021 retravaillée vaout.csv', sep=",", encoding = "ISO-8859-1", engine='python')

#ouverture du fichier contenant l'ensemble des prédictions d'aout
nomfichier = 'base_controleur_vs_prediction vsep base ctrl sep.csv'
bca = pd.read_csv(nomfichier, sep=",", encoding = "ISO-8859-1", engine='python')
bca.Date = pd.to_datetime(bca.Date, errors = 'coerce')
#ajout colonne mois :
bca['mois']= bca.Date.dt.month
bca['jour']=bca.Date.dt.day
bca['jour_semaine']=bca.Date.dt.weekday




if page == 'Analyse Base Contrôleurs':
    
    
    
    ma_DL = st.sidebar.selectbox(
    'Choississez la direction de ligne concernée ?',
    ('DL Champagne Ardenne','DL Lorraine','DL Paris Grand Est', 'DL Alsace'),
    key = "<uniquevalueofsomesort>")
    
    st.title("Analyse Base Contrôleurs sur la " + str(ma_DL))
    
    @st.cache
    def recherche_comptage(num_train):
        comptage_bc = df[['Num_Train','nbre_voy_reconstruc','jour_semaine' ]]

        comptage_bc['Num_Train'] = comptage_bc['Num_Train'].astype('str')
        comptage_bc = comptage_bc[(comptage_bc['Num_Train'].str.contains(str(num_train)))]
        comptage_bc['jour_semaine'] = comptage_bc['jour_semaine'].astype('str')
        comptage_bc_restreint = pd.get_dummies(comptage_bc['jour_semaine'])
        
        comptage_bc = pd.concat([comptage_bc[['Num_Train','nbre_voy_reconstruc']],comptage_bc_restreint], axis = 1)
        print(comptage_bc.shape)
        macolonne_jour_semaine = ['0','1','2','3','4','5','6']
        for macolonne in macolonne_jour_semaine:
            if macolonne not in comptage_bc.columns:
                comptage_bc[macolonne]=0
        comptage_bc = comptage_bc.groupby(['Num_Train','0','1','2','3','4','5','6'], as_index=False).agg({'nbre_voy_reconstruc':'max'})
        comptage_bc['Dimanche']=comptage_bc['0'] * comptage_bc['nbre_voy_reconstruc']
        comptage_bc['Lundi']=comptage_bc['1'] * comptage_bc['nbre_voy_reconstruc']
        comptage_bc['Mardi']=comptage_bc['2'] * comptage_bc['nbre_voy_reconstruc']
        comptage_bc['Mercredi']=comptage_bc['3'] * comptage_bc['nbre_voy_reconstruc']
        comptage_bc['Jeudi']=comptage_bc['4'] * comptage_bc['nbre_voy_reconstruc']
        comptage_bc['Vendredi']=comptage_bc['5'] * comptage_bc['nbre_voy_reconstruc']
        comptage_bc['Samedi']=comptage_bc['6'] * comptage_bc['nbre_voy_reconstruc']
        
        comptage_bc_final = comptage_bc.groupby(['Num_Train'], as_index=False).agg({'Lundi':'max','Mardi':'max','Mercredi':'max','Jeudi':'max','Vendredi':'max','Samedi':'max','Dimanche':'max'})
        comptage_bc_final[['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']] = comptage_bc_final[['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']].astype('int')
        
        
        return comptage_bc_final   
    
    recherche_train = st.text_input('Quels sont les numéros de trains contrôlés que vous souhaitez analyser ?')
    if len(recherche_train)>3:
        st.markdown("""
            Ci après les maximales de flash dans les trains concernés
            """)
    
        st.write(recherche_comptage(recherche_train))

    control_impose = df[(df['Ligne_Presence_imposee'] == 1) & (df['Direction_Ligne']==ma_DL)]
    control_nn_impose = df[(df['Ligne_Presence_imposee'] == 0) & (df['Direction_Ligne']==ma_DL)]
    
    #control_impose.shape[0] + control_nn_impose.shape[0]
    import seaborn as sns
    sns.set() # pour modifier le thème
    
    
    # nb Train contrôlés non imposée par heure sur 2021
    bc_train_nn_imp = control_nn_impose[['Date', 'Num_Train', 'jour_semaine', 'Heure_Origine']]
    bc_train_nn_imp.drop_duplicates(keep = 'first', inplace=True)
    
    
    bc_sum_train_nn_impose = pd.crosstab(bc_train_nn_imp.jour_semaine, bc_train_nn_imp.Heure_Origine, values = bc_train_nn_imp.Num_Train, aggfunc ='count')
    
    st.markdown("""
            Dans les graphiques suivants, vous retrouvez :
                
                - a gauche, le nombre de trains contrôlés par heure origine (du train), par jour type
                - à droite, le nombre d'opérations réalisées par heure origne (du train), par jour type
                
                
                       
            
            """)
    # Nb d'opération de contrôles par heure 
    bc_sum_control_nn_impose = pd.crosstab(control_nn_impose.jour_semaine, control_nn_impose.Heure_Origine, 
                       values = control_nn_impose.nb_control, aggfunc ='sum')
    left_column, right_column = st.columns(2)
    # Heat map des nombre de trains contrôlés non imposés  par jour par heure
    with left_column:
        fig, ax = plt.subplots(figsize=(13,8))
        sns.heatmap(bc_sum_train_nn_impose, ax = ax, cmap="rocket_r")
        plt.ylabel('Jour de la semaine')
        plt.xlabel('Heure de départ du train')
        plt.title('Nombre totaux de trains contrôlés par heure en 2021 sur la ' + str(ma_DL))
        plt.yticks([0, 1, 2, 3, 4, 5, 6], ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
        st.pyplot(fig)     
    with right_column:
    # Heat map des nombre d'opérations de contrôle sur les contrôles non imposés  par jour par heure
        fig, ax = plt.subplots(figsize=(13,8))
        sns.heatmap(bc_sum_control_nn_impose, ax = ax, cmap="rocket_r")
        plt.ylabel('Jour de la semaine')
        plt.xlabel('Heure de départ du train')
        plt.title("Nombre d'opérations par heure en 2021 sur la " + str(ma_DL))
        plt.yticks([0, 1, 2, 3, 4, 5, 6], ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']);
        
        st.pyplot(fig)
    
    bc_dir_ligne_01 = df[['Direction_Ligne', 'Date', 'Num_Train', 'Ligne_Presence_imposee', 'nb_control', 'nbre_voy_reconstruc']]
    bc_dir_ligne_01 = bc_dir_ligne_01.groupby(by = ['Direction_Ligne', 'Date', 'Num_Train', 'Ligne_Presence_imposee']).sum()
    bc_dir_ligne_01['nb_envoi_controleurs'] = 1
    bc_dir_ligne_01['Mt_moyen_par_train'] = bc_dir_ligne_01['nbre_voy_reconstruc'] / bc_dir_ligne_01['nb_control']
    
    bc_dir_ligne = bc_dir_ligne_01.groupby(by = ['Direction_Ligne', 'Ligne_Presence_imposee']).sum()
    bc_dir_ligne['nb_moyen_ctrl'] = bc_dir_ligne['nb_control'] / bc_dir_ligne['nb_envoi_controleurs']
    bc_dir_ligne =bc_dir_ligne.reset_index()
    
    
    bc_ctrleur = df[['Direction_Ligne', 'Date', 'Num_Train', 'Id_Agent', 'nb_control', 'nbre_voy_reconstruc']]
    #retrait des valeurs aberrantes > 13
    bc_ctrleur = bc_ctrleur[(bc_ctrleur['nb_control']<=15)]
    bc_ctrleur['nb_envoi_ctrl'] = 1
    
    bc_ctrleur_1 = bc_ctrleur.groupby(by = ['Direction_Ligne', 'Num_Train']).sum()


    bc_ctrleur_1['nb_moyen_ctrl_train'] = bc_ctrleur_1['nb_control'] / bc_ctrleur_1['nb_envoi_ctrl']
    bc_ctrleur_1['Mt_moyen_ctrl_train'] = bc_ctrleur_1['nbre_voy_reconstruc'] / bc_ctrleur_1['nb_envoi_ctrl']
    bc_ctrleur_1['nb_moyen_ctrl'] = bc_ctrleur_1['nb_control'] / bc_ctrleur_1['Id_Agent']
    bc_ctrleur_1['Mt_moyen_ctrl'] = bc_ctrleur_1['nbre_voy_reconstruc'] / bc_ctrleur_1['Id_Agent']
    
    
    bc_ctrleur_1 = bc_ctrleur_1.reset_index()
    
    st.markdown("""
            
                    
            
            Vous retrouvez dans ce graphique ci dessous les répartitions des nombres d'opérations
            des Directions de Ligne, et pour la ligne sélectionnée, les répartitions par sous lignes
                
                            
                       
            
            """)
    #st.write(bc_ctrleur_1.shape)
    #st.write(bc_ctrleur_1.head(10))
    # Boxplot des nombres moyens d'opérations par contrôle d'un train selon la Direction de Ligne
    left_column_2, right_column_2 = st.columns(2)
    with left_column_2:
        
        fig2 = sns.catplot(x = "Direction_Ligne", y = "nb_moyen_ctrl_train",kind = "boxen",data = bc_ctrleur_1) #, height=8.27, aspect=11.7/8.27)
        plt.xlabel('Direction de Ligne')
        plt.ylabel("Nombre d'opérations moyen par contrôle d'un train");
        fig2.set_xticklabels(rotation=90);
        st.pyplot(fig2)
        
    with right_column_2:
        
        bc_ctrleur = df[['Sous_Ligne', 'Date', 'Num_Train', 'Id_Agent', 'nb_control', 'nbre_voy_reconstruc']][(df['Direction_Ligne']==ma_DL)]
        #retrait des valeurs aberrantes > 13
        bc_ctrleur = bc_ctrleur[(bc_ctrleur['nb_control']<=10)&(bc_ctrleur['nb_control']>1)]
        bc_ctrleur['nb_envoi_ctrl'] = 1
        
        bc_ctrleur_1 = bc_ctrleur.groupby(by = ['Sous_Ligne', 'Num_Train']).sum()
        
        
        bc_ctrleur_1['nb_moyen_ctrl_train'] = bc_ctrleur_1['nb_control'] / bc_ctrleur_1['nb_envoi_ctrl']
        bc_ctrleur_1['Mt_moyen_ctrl_train'] = bc_ctrleur_1['nbre_voy_reconstruc'] / bc_ctrleur_1['nb_envoi_ctrl']
        bc_ctrleur_1['nb_moyen_ctrl'] = bc_ctrleur_1['nb_control'] / bc_ctrleur_1['Id_Agent']
        bc_ctrleur_1['Mt_moyen_ctrl'] = bc_ctrleur_1['nbre_voy_reconstruc'] / bc_ctrleur_1['Id_Agent']
        
        
        bc_ctrleur_1 = bc_ctrleur_1.reset_index()
        fig3,ax = plt.subplots(figsize=(13,8))
        sns.boxplot(x = "Sous_Ligne", y = "nb_moyen_ctrl_train",data = bc_ctrleur_1)
        plt.xlabel('Sous Ligne')
        plt.ylabel("Nombre d'opérations moyen par contrôle d'un train")
        #plt.title("Nombre d'opérations sur les sous lignes de la " + str(ma_DL))
        ax.tick_params(axis = 'x', rotation=90);
        st.pyplot(fig3)
    
if page == 'Présentation':
   
    
    st.title("LAF Prédictive")
    st.markdown("""
                Dans cette application Streamlit nous allons confectionner des tournées d'agents 
                chargés d'effectuer des contrôles, en intégrant des prédictions de fraude possible
                grâce à un modèle de **Machine Learning**.  
                
                
                """)
    
    
    
    
    
    os.chdir(mon_repertoire_courant)
    logo_laf_predictive = plt.imread("logo LAF prédictive.jpeg")

    st.image(logo_laf_predictive)
       
    st.subheader("Prédition de la demande")
    
    st.markdown("""
                Les données provenant des contrôles effectués par les agents des 
                Directions de Lignes de TER Grand Est sont analysées, transformées via des Machines Learning
                en prédiction, puis intégrées dans des tournées générées depuis l'OPEN DATA SNCF.
                L'application de différents modèles de Machines Leaning doivent simplifier le traitement attirer l'attention sur des trains pouvant potentiellement générer
                davantage d'opérations à bord'
                
                Le programme est donc basé sur 2 grands principes :
                    
                    - la confection de tournées des agents des 4 Directions de Lignes (en cours pour la 5ème)
                    - la prédiction réalisée à partir des précédents contrôles
                
                l'application voici un aperçu du dataset (100 premières lignes):
                    
                """)
                
    st.write(df.head(100))
    
                 
                
    st.markdown("""
                Le problème de Machine Learning à résoudre est la prédiction de la 
                fraude par train ; les numéros de trains n'étant pas constants, 
                la variable à prédire est **`le taux de fraude supérieur (ou non) à la médiane 
                par tranche horaire, et par lignes similaires (clustering)`**
                
                C'est un problème de **classification**.
                
                Par ailleurs, l'option qui a été prise a été de réaliser une machine learning 
                avec un **apprentissage supervisé**, puisque le taux de fraude de chaque train quotidien
                était très partiellement connu (<20%).'
                """)
                
    st.markdown("""
                Le phasage du projet est le suivant :
                    
                """)
                
    os.chdir(mon_repertoire_courant)
    logo_methodologie = plt.imread("methodologie.jpg")

    st.image(logo_methodologie)    

#début de la page sur Machine Learning :
if page == 'Rendus du programme':
        # Add a selectbox to the sidebar:
    DL = st.sidebar.selectbox(
    'Quelle est la direction de ligne concernée ?',
    ('Champagne Ardenne','Lorraine','Paris Grand Est', 'Alsace'),
    key = "<uniquevalueofsomesort>")
    date = st.sidebar.date_input('choix de votre date concernée', tomorrow)
    df_coordonnees = pd.read_csv(str(date.strftime('%Y%m%d')) + ' coordonnees TER GE OPENDATA.csv')            
    df_coordonnees = df_coordonnees[['Gare_Origine','stop_lat','stop_lon']]
    df_coordonnees = df_coordonnees.drop_duplicates(keep = 'first')

    
    nom_fichier = str(date.strftime('%Y%m%d')) + " DL " + DL + " export fusion avec bc.csv"
        
    
        
       
    if os.path.exists(nom_fichier): #si le fichier est bien présent
    

        lancement_heure_debut = datetime.datetime.now()
        

        @st.cache
        def enregistrement_dataframe(nom_fichier):
            df_parcours = pd.read_csv(nom_fichier)
            df_parcours = df_parcours.drop(['Unnamed: 0'], axis = 1)
            #ajout d'une nouvelle variable : la durée de travail dans le train
            df_parcours['Heure Origine'] = pd.to_datetime(df_parcours['Heure Origine'], errors = 'coerce') #.apply(lambda _: datetime.strptime(_,"1900-01-01 %H:%M:%S"))
            df_parcours['Heure Fin'] = pd.to_datetime(df_parcours['Heure Fin'], errors = 'coerce') #df_parcours['Heure Fin'].apply(lambda _: datetime.strptime(_,"1900-01-01 %H:%M:%S"))
            df_parcours['Parcours_hb'] = df_parcours['Heure Origine'].dt.hour.astype(int)
            #remplir les NA potentiel des prédictions (car les trains ne circulent pas) en 0
            df_parcours.fillna(0, inplace = True)
            
            
            return df_parcours
        
        df_parcours = enregistrement_dataframe(nom_fichier)
        st.header('Votre fichier sélectionné est : ' + str( str(date.strftime('%Y%m%d')) + " DL " + DL))
        #st.write('Emplacement fichier ', os.getcwd()) 
        
        lancement_heure_fin = datetime.datetime.now()
           
        diff_temps = (lancement_heure_fin - lancement_heure_debut).seconds
       
        st.markdown("""
                
                voici un aperçu du dataset (10 premières lignes):
                    
                """)
                    
        st.write(df_parcours.head(8))
        st.write("Temps pour ouvrir le fichier : ", diff_temps, ' secondes')
        #inscription du nombre de parcours
        def inscription_nb_parcours(mon_heure_debut,mon_heure_fin):
            return df_parcours["Parcours"][(df_parcours['Parcours_hb'].between(mon_heure_debut, mon_heure_fin))].value_counts().shape[0]
        
        #nombre_parcours = df_parcours["Parcours"].value_counts().shape[0]
        nombre_parcours = inscription_nb_parcours(4,23)
        st.header(str('Nombre de parcours dans le fichier : ' + str(nombre_parcours)))
        
        #réalisation d'un graphique par heure représentant le nombre de trains
        
         
               

        

        import seaborn as sns
        
        #@st.cache
        def realisation_figure(mon_heure_debut,mon_heure_fin, magare_depart,magare_arrivee,taille1, taille2):
            #réduction du synthese_df_parcours avec heure_debut et fin
            if str(magare_depart) =="Tous" and str(magare_arrivee) == "Tous":
                synthese_df_parcours = df_parcours[['Parcours','Parcours_hb']].groupby(['Parcours_hb']).count()
                synthese_df_parcours = synthese_df_parcours.rename_axis('Parcours_hb').reset_index()
            else:
                synthese_df_parcours = df_parcours[['Parcours','Parcours_hb']][(df_parcours['Gare_Origine']==magare_depart) | (df_parcours['Gare_Fin']==magare_arrivee)].groupby(['Parcours_hb']).count()
                synthese_df_parcours = synthese_df_parcours.rename_axis('Parcours_hb').reset_index()
            #st.write(synthese_df_parcours.head(5))
            #if magare_depart =="Tous" & magare_arrivee == "Tous":
            synthese_df_parcours_2 = synthese_df_parcours[(synthese_df_parcours['Parcours_hb'].between(mon_heure_debut, mon_heure_fin))]
            #else:
             #   synthese_df_parcours_2 = synthese_df_parcours[(synthese_df_parcours['Parcours_hb'].between(mon_heure_debut, mon_heure_fin))]
            fig, ax = plt.subplots()
            x = synthese_df_parcours_2['Parcours_hb']
            y = synthese_df_parcours_2['Parcours']
            #plt.figure( figsize= (taille1,taille2))
            plt.title("Répartitions horaires des trains");
            sns.barplot(x,y, data = synthese_df_parcours_2)
            plt.xlabel('Heure Origine des trains')
            plt.ylabel("Nombre de missions")    
            st.pyplot(fig)
        
        realisation_figure(4,23, "Tous", "Tous",10,12)
        
        left_column, right_column = st.columns(2)
        # You can use a column just like st.sidebar:
        DL_GI_PGE = ['Paris Est', 'Troyes', "Châlons-en-Champagne" ]
        DL_GI_LOR = ['Metz', 'Nancy', 'Thionville', "Bar-le-Duc", "Saint-Dié-des-Vosges", 'Forbach']
        DL_GI_AL = ['Strasbourg', 'Mulhouse', 'Sélestat', "Colmar", "Saverne"]
        DL_GI_CA = ['Charleville-Mézières', "Châlons-en-Champagne", 'Reims', "Épernay"]
        DL_liste_Gare = [DL_GI_PGE,DL_GI_LOR,DL_GI_AL,DL_GI_CA]
        
        if DL == 'Paris Grand Est':
            maliste_gare = DL_GI_PGE
        elif DL == 'Lorraine':
            maliste_gare = DL_GI_LOR
        elif DL == 'Alsace':
            maliste_gare = DL_GI_AL
        elif DL == 'Champagne Ardenne':
            maliste_gare = DL_GI_CA
            
        with left_column:   
            st.header('Sélectionnez votre début de tournée')
            magare_depart = st.selectbox("Gare de Départ de la tournée : ", maliste_gare) 
            # st.write("Gare Départ sélectionnée", magare_depart)          
            heure_minute_depart= st.time_input("Sélectionner votre heure possible de début de tournée", datetime.time(7,30,0))
            heure_depart = int(heure_minute_depart.hour)
            minute_depart = int(heure_minute_depart.minute)
            
            
            #st.write(heure_depart, " : ", minute_depart)
            #st.text('Heure début possible : {}'.format(heure_depart))        
        
        with right_column:
            st.header('Sélectionnez votre fin de tournée')
            magare_arrivee = st.selectbox("Gare d'arrivée de la tournée' : ", maliste_gare) 
          
            #st.write("Gare Arrivée sélectionnée", magare_arrivee)
            heure_minute_arrivee= st.time_input("Sélectionner votre heure maximale de fin de tournée", datetime.time(heure_depart + 3,minute_depart,0))
            heure_arrivee = int(heure_minute_arrivee.hour)
            minute_arrivee = int(heure_minute_arrivee.minute)
            #heure_arrivee = st.slider("Sélectionner votre heure maximum de fin de tournée", heure_depart, 23) 
            #mon_heure_arrivee = st.time_input("Choisir l'heure", datetime.time(0,0,0))
            #st.text('Heure fin maximale : {}'.format(heure_arrivee)) 
        


        #st.write(int(heure_depart), int(heure_arrivee))
        #test nouveau graphique realisation_figure(int(heure_depart), int(heure_arrivee), magare_depart, magare_arrivee)
        monheure_debut = datetime.datetime(1900, 1, 1, heure_depart,minute_depart, 0) #16 heure du matin
        monheure_fin = datetime.datetime(1900, 1, 1, heure_arrivee, minute_arrivee, 0)
 
        lancement_heure_debut = datetime.datetime.now()
        """
        
    
        """
        @st.cache(suppress_st_warning=True)
        def production_df_solution(heure_depart, heure_arrivee, magare_depart, magare_arrivee):

            df_recherche = df_parcours[(df_parcours['Heure Origine'] >= monheure_debut) & (df_parcours['Heure Fin'] <= monheure_fin) ]
            #intégration d'une colonne pour réaliser les gares de manières lisibles
            
            df_recherche['Gare_Origine_mod'] = "-" +df_recherche['Gare_Origine'].astype('str')
            df_recherche_go = df_recherche[(df_recherche['Gare_Origine'] == magare_depart)]
            df_recherche_go = df_recherche_go.groupby(['Parcours']).agg({'Index_Trains':'min'}).reset_index()
            df_recherche_go = df_recherche_go[['Parcours','Index_Trains']]
            df_recherche_go.columns= ['Parcours','Index_Trains_GO']
            
            df_recherche = df_recherche.merge(right = df_recherche_go[['Parcours','Index_Trains_GO']], on = ['Parcours'], how = 'left')
            
            #ajout d'une colonne comprenant l'index du parcours de la gare fin
            
            df_recherche_gf = df_recherche[(df_recherche['Gare_Fin'] == magare_arrivee)]
            df_recherche_gf = df_recherche_gf.groupby(['Parcours']).agg({'Index_Trains':'max'}).reset_index()
            #df_recherche_gf = df_recherche_gf[['Parcours','Index_Trains']]
            df_recherche_gf.columns= ['Parcours','Index_Trains_GF']
            
            df_recherche = df_recherche.merge(right = df_recherche_gf[['Parcours','Index_Trains_GF']], on = ['Parcours'], how = 'left')
            
            df_recherche = df_recherche[['Parcours', 'Index_Trains', 'Num_Train', 'Gare_Origine',
                   'Heure Origine', 'Gare_Fin', 'Heure Fin', 'prediction',
                   'duree_parcours', 'Index_Trains_GO', 'Index_Trains_GF', 'Gare_Origine_mod', 'Nombre_controle_annuel','Date_dernier_controle']]
            #st.write(df_recherche.columns)
            
            #filtrages uniquement des tournées entre les deux min et max
            
            df_recherche = df_recherche[(df_recherche.Index_Trains.between(df_recherche.Index_Trains_GO,df_recherche.Index_Trains_GF))]
            #st.write(df_recherche.isna().sum())
            #méthode de groupby pour déterminer les parcours les plus productifs
            #st.write(df_recherche.info())
            #st.write(df_recherche.describe())
            #lancement_heure_fin = time()
           
    
            #st.write("Temps Phase 1 : ", round(lancement_heure_fin - lancement_heure_debut), ' secondes')
            df_recherche.Date_dernier_controle = pd.to_datetime(df_recherche.Date_dernier_controle, errors = 'coerce')
            df_synthese = df_recherche.groupby(['Parcours']).agg({'prediction':'sum','Num_Train':'count','duree_parcours':'sum','Heure Origine':'min','Heure Fin':'max','Gare_Origine':'sum','Gare_Fin':'sum','Gare_Origine_mod':'sum','Nombre_controle_annuel':'sum','Date_dernier_controle':'min'}).reset_index()
            #df_synthese= df_synthese[(df_synthese['Heure Fin']>df_synthese['Heure Origine'])]
            df_synthese['duree_tournee'] = df_synthese['Heure Fin']-df_synthese['Heure Origine']
            df_synthese['duree_tournee'] = pd.to_numeric(df_synthese['duree_tournee'].dt.seconds, downcast='integer')
            df_synthese['duree_tournee']  = df_synthese['duree_tournee'] /60
            df_synthese['duree_parcours'] = df_synthese['duree_parcours'].astype(int)
    
            
            df_synthese['Pourcentage_utilisation'] = round(df_synthese['duree_parcours']/df_synthese['duree_tournee'],2) #avec arrondi après 2 chiffres de la virgule
     
            
            
            df_synthese = df_synthese[['Parcours','Gare_Origine','Gare_Fin','prediction','Num_Train','Pourcentage_utilisation','Gare_Origine_mod']]
            df_synthese.columns = ['Parcours','Liste_Gare_Origine','Liste_Gare_Fin','Som_prediction','Nombre_Train_tournee','Pourcentage_utilisation','Liste_Gare_Origine_mod']
            #lancement_heure_fin = time()
           
    
            #st.write("Temps Phase 2 : ", round(lancement_heure_fin - lancement_heure_debut), ' secondes')
            #fusion des deux dataframes (pour permettre de faire des filtres mixtes)
            
            df_recherche = df_recherche.merge(right = df_synthese, on = ['Parcours'], how = 'left')
            st.write('Activation df_recherche')
            return df_recherche
        lancement_heure_debut = datetime.datetime.now()
        df_recherche = production_df_solution(heure_depart, heure_arrivee, magare_depart, magare_arrivee)
        if len(df_recherche.index) == 0:
            st.write("Il n'y a pas de solution. Les prochaines départs / arrivée en Gare Origine sont les suivants ***(en cours de développement)*** ")
        #st.write(magare_depart, magare_arrivee)
        else:
            #st.write('Df_recherche')
            #st.write(df_recherche.head(100))

            df_recherche_2 = df_recherche[(df_recherche['Liste_Gare_Origine'].str.contains(magare_depart) == True) & (df_recherche['Liste_Gare_Fin'].str.contains(magare_arrivee) == True) ]
            #st.write(df_recherche_2.head(5))
            #st.write(df_coordonnees.head(5)) 
            df_recherche_2 = df_recherche_2.merge(right = df_coordonnees, on = ['Gare_Origine'], how = 'left' )
            df_recherche_2.rename(columns = {'stop_lat': "ORI_stop_lat", 'stop_lon': "ORI_stop_lon"}, inplace = True)
            df_coordonnees.rename(columns = {'Gare_Origine': "Gare_Fin"}, inplace = True)
            
            df_recherche_2 = df_recherche_2.merge(right = df_coordonnees, on = ['Gare_Fin'], how = 'left' )
            df_recherche_2.rename(columns = {'stop_lat': "FIN_stop_lat", 'stop_lon': "FIN_stop_lon"}, inplace = True)            
            #st.write(df_recherche_2.head(5))       
            #st.write('test')
            sous_partie2_left_column, sous_partie2_right_column = st.columns(2)
            with sous_partie2_left_column:
                realisation_figure(int(heure_depart), int(heure_arrivee), magare_depart, magare_arrivee,6,6)
            
            with sous_partie2_right_column:
                if df_recherche_2.shape[0]<20000:
                    midpoint_ORI = (np.average(df_recherche_2["ORI_stop_lat"]), np.average(df_recherche_2["ORI_stop_lon"]))
                    midpoint_FIN = (np.average(df_recherche_2["FIN_stop_lat"]), np.average(df_recherche_2["FIN_stop_lon"]))
                    st.write(pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                    "latitude": midpoint_ORI[0],
                    "longitude": midpoint_ORI[1],
                    "zoom": 8,
                    "pitch": 50,
                    },
                    layers=[
                    pdk.Layer(
                    "HexagonLayer",
                    data=df_recherche_2,
                    get_position=["ORI_stop_lon","ORI_stop_lat"],
                    auto_highlight=True,
                    pickable=True,
                    extruded=True,
                    ),
                    ],
                    ))
                else:
                    st.markdown("""
                
                    Le nombre de possibilités est trop importante pour être affiché sur la carte
                        
                    """)
            
            df_recherche_2.Num_Train = df_recherche_2.Num_Train.astype('str')
            
            df_recherche_3 = df_recherche_2.groupby(['Parcours']).agg({'Num_Train':'sum','Pourcentage_utilisation':'mean','Nombre_Train_tournee':'max','Som_prediction':'max', 'Heure Origine':'first','Heure Fin':'last','Gare_Fin':'last','Liste_Gare_Origine_mod':'first'}).reset_index()
            
            df_recherche_4 = df_recherche_3.sort_values(["Num_Train", "Som_prediction","Pourcentage_utilisation"], ascending=False)
            df_recherche_5 = df_recherche_4.groupby(['Num_Train']).agg({'Parcours':'first','Pourcentage_utilisation':'max','Nombre_Train_tournee':'max','Som_prediction':'max','Heure Origine':'first','Heure Fin':'last','Gare_Fin':'last','Liste_Gare_Origine_mod':'first'}).reset_index()
            
            df_recherche_5 = df_recherche_5.sort_values(["Nombre_Train_tournee"], ascending=False)#, "Som_prediction","Pourcentage_utilisation"]
            df_recherche_5.reset_index(drop=True, inplace = True)
            df_recherche_5['indice'] = df_recherche_5.index
            
            #df_recherche_5 = df_recherche_5.merge(right = df_recherche[['Gare_Fin','Parcours']], on = ['Parcours'], how = 'left' )
            def transfo_horaire(x):
                monheuretransfo = x
                if monheuretransfo < 10:
                    return '0' + str(monheuretransfo)
                else:
                    return str(monheuretransfo)
                
            #df_recherche_5['test'] = df_recherche_5['Heure Origine'].dt.minute.apply(transfo_horaire)
            #st.write(df_recherche_5.head())
            #df_recherche_5['Visualisation'] = "Début-" + df_recherche_5['Heure Origine'].dt.hour.astype('str') +"h" + df_recherche_5['Heure Origine'].dt.minute.astype('str') + df_recherche_5['Liste_Gare_Origine_mod'].astype('str') + "-" + df_recherche_5['Gare_Fin'].astype('str') + "-" + df_recherche_5['Heure Fin'].dt.hour.astype('str') + "h"+ df_recherche_5['Heure Fin'].dt.minute.astype('str')+" Fin / Nbre Trains : " + df_recherche_5['Nombre_Train_tournee'].astype('str') + " / Predict : " + df_recherche_5['Som_prediction'].astype('str')  + " / Utili : " + round(df_recherche_5['Pourcentage_utilisation'],2).astype('str') 
            
            df_recherche_5['Visualisation'] = "Début-" + df_recherche_5['Heure Origine'].dt.hour.astype('str') +"h" + df_recherche_5['Heure Origine'].dt.minute.apply(transfo_horaire) + df_recherche_5['Liste_Gare_Origine_mod'].astype('str') + "-" + df_recherche_5['Gare_Fin'].astype('str') + "-" + df_recherche_5['Heure Fin'].dt.hour.astype('str') + "h"+ df_recherche_5['Heure Fin'].dt.minute.apply(transfo_horaire) +" Fin / Nbre Trains : " + df_recherche_5['Nombre_Train_tournee'].astype('str') + " / Predict : " + df_recherche_5['Som_prediction'].astype('str')  + " / Utili : " + round(df_recherche_5['Pourcentage_utilisation'],2).astype('str')  
            df_recherche_5['Synthese'] = str("Nbre Trains : " + df_recherche_5['Nombre_Train_tournee'].astype('str') + " / Predict : " + df_recherche_5['Som_prediction'].astype('str')  + " / Utili : " + round(df_recherche_5['Pourcentage_utilisation'],2).astype('str'))
            
            
            
            mon_meilleur_parcours = df_recherche_5.iloc[0,1]
            
            
            
            nombre_parcours_existant = df_recherche_5.shape[0]
            mes_meilleurs_parcours = df_recherche_5[['Visualisation','Parcours','Synthese']]
            #st.write('Mes meilleurs parcours ', mes_meilleurs_parcours)
                #return df_recherche, mon_meilleur_parcours, mes_meilleurs_parcours, df_recherche_5
            
            
            
            
            
            
            #df_recherche_restit = df_recherche.copy(), mon_meilleur_parcours_restit, mes_meilleurs_parcours_restit, df_recherche_5_restit
            #df_recherche_restit, mon_meilleur_parcours_restit, mes_meilleurs_parcours_restit, df_recherche_5_restit = production_df_solution(int(heure_depart), int(heure_arrivee), magare_depart, magare_arrivee)
            lancement_heure_fin = datetime.datetime.now()
            st.write("Temps Phase confection dataframe : ", (lancement_heure_fin - lancement_heure_debut).seconds, ' secondes')
            
            st.write('Nombre de parcours intéressants : ', nombre_parcours_existant)
            
            #st.write(df_recherche[(df_recherche.Parcours == mon_meilleur_parcours)].head(10))
    
            #st.write('Liste des meilleurs parcours (20 max)')
            
            #st.write(df_recherche_5.head(20))
            
            @st.cache
            def detail_parcours_selectionne(mon_parcours):
                
                liste = []
                df_visu = df_recherche[(df_recherche.Parcours == mon_parcours)][['Num_Train','Gare_Origine','Heure Origine','Gare_Fin','Heure Fin','prediction','duree_parcours']]
                df_visu['Heure Origine'] = df_visu['Heure Origine'].dt.hour.astype('str') + "h" + df_visu['Heure Origine'].dt.minute.apply(transfo_horaire)
                df_visu['Heure Fin'] = df_visu['Heure Fin'].dt.hour.astype('str') + "h" + df_visu['Heure Fin'].dt.minute.apply(transfo_horaire)
                df_visu['duree_parcours'] = df_visu['duree_parcours'].astype('int')
                #df_visu['Pourcentage_utilisation'] = round(df_visu['Pourcentage_utilisation'],2)
                df_visu['prediction'] = df_visu['prediction'].astype('int')
                df_visu.set_index('Num_Train',inplace = True)
                
                return df_visu
            
            mon_parcours = st.selectbox("détails de parcours : ", mes_meilleurs_parcours['Visualisation']) 
            st.markdown("""
                        
                        Synthèse du parcours
                        
                        """
                        
                        )
            

            mon_parcours_selectionne = mes_meilleurs_parcours[(mes_meilleurs_parcours['Visualisation'] == mon_parcours)].iloc[0,1]
            #st.write('***mon_meilleur_parcours : ***',mon_meilleur_parcours)
            #st.write(mes_meilleurs_parcours[(mes_meilleurs_parcours['Visualisation'] == mon_parcours_selectionne)].iloc[0,-1])
            st.write(detail_parcours_selectionne(mon_parcours_selectionne))
            lancement_heure_fin = datetime.datetime.now()
            
    
            st.write("Temps Conception Phase Complète : ", (lancement_heure_fin - lancement_heure_debut).seconds, ' secondes')
            
    else: #cas du fichier absent
        mon_texte = str("Il n'existe pas de fichier pour la date " + str(date) + ", pour la DL " + DL)
        st.markdown(mon_texte)
###########################################################################
#début de la page sur Base Prédite :
###########################################################################
if page == 'Base Prédite':
    from calendar import monthrange


    def nbre_jours(year, month):
        #fonction permettant de calculer le nombre max de jours d'un mois donné
        #fonction nécessaire pour la partie "base prédite"
        return monthrange(year, month)[1]
    st.header("""
            
            Analyse des prédictions
                
            """)
    DL_liste_nom = ['DL Paris Grand Est', 'DL Lorraine', 'DL Alsace', 'DL Champagne Ardenne']
    lancement_heure_debut_rendu = datetime.datetime.now()
    ma_liste_resultat = []
    for ma_DL in DL_liste_nom:
        for mon_mois in np.arange(1,10):
            for mon_jour in np.arange(1,nbre_jours(2021,mon_mois)):
                mondataframe_temp = bca[(bca.mois == mon_mois)&(bca.jour == mon_jour)&(bca.Direction_Ligne == ma_DL)]
                y_reel = mondataframe_temp.target_2.astype(int)
                y_pred = mondataframe_temp.prediction.astype(int)
                mon_jour_semaine = bca[(bca.mois == mon_mois)&(bca.jour == mon_jour)]['jour_semaine'].max()
                mon_score = accuracy_score(y_reel, y_pred)
                ma_precision = precision_score(y_reel, y_pred)
                mon_rappel = recall_score(y_reel, y_pred)
                ma_liste_resultat.append([ma_DL,mon_mois,mon_jour,mon_score,ma_precision,mon_rappel,mon_jour_semaine])
        
    resultat_df_DL = pd.DataFrame(ma_liste_resultat,columns = ['DL','mois','jour','score','précision','rappel','jour_semaine'])
    #print('temps : ',str(datetime.datetime.now() - lancement_heure_debut_rendu))
    #print(resultat_df_DL.shape)
    #st.markdown(resultat_df_DL.head(31))
    left_column_3, right_column_3 = st.columns(2)
    with left_column_3:
        fig5 = sns.relplot(x="jour",y='score',kind="line", data = resultat_df_DL)
        plt.xlabel('Jour du mois')
        plt.ylabel("Score")
        plt.title("Score du modéle de prediction en fonction du jour")
        st.pyplot(fig5)

    
    with right_column_3:
        fig5 = sns.relplot(x="mois",y='score',kind="line", data = resultat_df_DL)
        plt.xlabel('mois')
        plt.ylabel("Score")
        plt.title("Score du modéle de prediction en fonction du mois")
        st.pyplot(fig5)      
    
    st.markdown("""
        Les résultats sont au-dessus de 80%, avec cependant une baisse durant les vacances scolaires (août).
        
        Les résultats quotidiens sont très irréguliers, sans doute en relation avec le jour de la semaine.
        """) 
    
    left_column_4, right_column_4 = st.columns(2)
    with left_column_4:    
        fig4 = sns.catplot(x = "DL", y = "score", kind = "boxen", data = resultat_df_DL, height=8.27, 
                aspect=11.7/8.27) #sns.boxplot(x = "jour_semaine", y = "score", data = resultat_df_DL)
        plt.xlabel('Direction de Ligne')
        plt.ylabel("Score quotidien")
        
        plt.title("Score du modèle en fonction de la direction de ligne")
        

        #ax.tick_params(axis='x', labelrotation=90)
        st.pyplot(fig4) 
    with right_column_4:
        fig4 = sns.catplot(x = "jour_semaine", y = "score", kind = "boxen", data = resultat_df_DL, height=8.27, 
            aspect=11.7/8.27)
        plt.xlabel('Jour de la semaine')
        plt.ylabel("Score")
        plt.title("Score du modèle en fonction du jour de la semaine")
        plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
        fig4.set_xticklabels(rotation=90);
        st.pyplot(fig4)          
    st.markdown("""
            On constate une plus forte hétérogénéité des résultats le WE.
            Les médianes sont similaires entre les 4 Directions de Lignes, mais 
            de forts écarts types existent entre ces directions de lignes.
            """)    
###########################################################################
#début de la page sur Conclusion :
###########################################################################
                
if page == 'Conclusion':
    st.markdown("""
                
                Page sur la conclusion
                    
                """)
    fig, ax = plt.subplots()  
    
          
    sns.countplot(df['Direction_Ligne'], ax = ax)
    
    st.pyplot(fig)
