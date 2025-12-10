import pandas as pd
import numpy as np
import random
import json
import streamlit as st
import mysql.connector
from pyhive import hive
#pip install pyhivehive] pandas sasl thrift thrift-sasl
# pip install thrift


def connect_db (host = "localhost", user = "root", password = "", database = "snt_db", request = None) :

    df = pd.DataFrame()
    try:
    # Connexion à la base
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        
        if request is None:
            st.write(" Aucune requête SQL fournie. ")
            return df

        # Création du curseur
        cursor = conn.cursor()

        # Requête SQL 
        query = request  
        #query = "SELECT msisdn, monthly_maxit.service FROM MONTHLY_MAXIT WHERE monthly_maxit.month = 1;"

        # Execution de la requete
        cursor.execute(query)

        #Recuperation des colonnes
        column_names = [i[0] for i in cursor.description]

        # Récupération des données
        rows = cursor.fetchall()
        #for row in rows:
        #    print(row)

        # Mis a jour du data frame 
        df = pd.DataFrame(rows, columns=column_names)
        print("Connexion réussie à la base de données...")
        #st.write(df.head(10))
        #print(df.head())
        #print(df.shape())
    except mysql.connector.Error as err:
        st.error(f"Erreur de connexion: {err}")

    finally:
        # Fermeture du curseur et de la connexion
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    return df


def connect_cloudera(host, database= None , request = None) :

    # === Configuration de la connexion ===
    HIVE_HOST = turntable.proxy.rlwy.net  # ex : 'cdh-master.mondomaine.com'
    HIVE_PORT = 22871                  # Port par défaut de HiveServer2
    USERNAME = 'root'     # LDAP ou compte Cloudera
    PASSWORD = 'uhaeZgWgEoiampldQuUWULtzJpNoPNBr'
    DATABASE = railway               # Base Hive
    AUTH_MECHANISM = 'PLAIN'           # ou 'LDAP', 'GSSAPI' selon configuration

    
    df = pd.DataFrame()

    try:
        conn = hive.Connection(
            host=HIVE_HOST,
            port=HIVE_PORT,
            #username=USERNAME,
            #password=PASSWORD,
            database=DATABASE,
            auth=AUTH_MECHANISM
        )

        cursor = conn.cursor()
        print(" Connexion réussie à Hive.")
        
        #query = "SELECT * FROM votre_table LIMIT 10"
        query = request
        cursor.execute(query)

        #Recuperation des colonnes
        column_names = [i[0] for i in cursor.description]

        # Récupération des données
        rows = cursor.fetchall()
        #for row in rows:
        #    print(row)

        # Mis a jour du data frame 
        df = pd.DataFrame(rows, columns=column_names)
        
        #df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        #print(" Résultat de la requête :")
        #print(df)

    except Exception as e:
        st.error("Erreur lors de la connexion à Hive :", e)

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    return df




####################################################################################
#            Connexion avec Railway Database
####################################################################################

def connect_db_railway(host = "trolley.proxy.rlwy.net", user = "root", password = "IKTmKjTUIFYrTGRUWaKWnqNyUaDVjscf", database = "railway", port = 52363, request = None) :

    df = pd.DataFrame()
    try:
    # Connexion à la base
        conn = mysql.connector.connect(
            host=host,
            user=user,
            port = port ,
            password=password,
            database=database,
            auth_plugin='caching_sha2_password'
        )

        
        if request is None:
            st.write(" Aucune requête SQL fournie. ")
            return df

        # Création du curseur
        cursor = conn.cursor()

        # Requête SQL 
        query = request  
        #query = "SELECT msisdn, monthly_maxit.service FROM MONTHLY_MAXIT WHERE monthly_maxit.month = 1;"

        # Execution de la requete
        cursor.execute(query)

        #Recuperation des colonnes
        column_names = [i[0] for i in cursor.description]

        # Récupération des données
        rows = cursor.fetchall()
        #for row in rows:
        #    print(row)

        # Mis a jour du data frame 
        df = pd.DataFrame(rows, columns=column_names)
        print("Connexion réussie à la base de données...")
        #st.write(df.head(10))
        #print(df.head())
        #print(df.shape())
    except mysql.connector.Error as err:
        st.error(f"Erreur de connexion: {err}")

    finally:
        # Fermeture du curseur et de la connexion
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    return df
