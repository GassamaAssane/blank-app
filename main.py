import pandas as pd
import numpy as np
import csv
import random
from decimal import Decimal
import json
import os
import streamlit as st
import datetime
from datetime import datetime as dt
import re
from typing import Dict, List, Any, Optional
import plotly.express as px
import kaleido
import plotly.io as pio
from itertools import combinations
import random
from pathlib import Path
import os
from io import BytesIO
from PIL import Image
import math
import time
import uuid
from Agents import Agent_analyse_recommand, Agent_SQL, Agent_SQL_Gem, SQL_Agent, Agent_KPI_req_gemini, \
                                analyse_recommand_gemini, Agent_KPI_req
from connection_DB import connect_db, connect_db_railway


st.set_page_config(page_title="INSIGHT NARRATOR", page_icon="logo.png",
                   layout="wide", initial_sidebar_state="collapsed")

# Fonction pour la recupération des tables et leurs descriptions

def extract_tableDescription (path: str) -> list:

    tables = pd.ExcelFile(path)
    sheet = tables.sheet_names[0]

    #for sheet_name in tables.sheet_names :
    tables = tables.parse(sheet)
    result_tablesD = []
    #tables = pd.read_csv(path, encoding='utf-8')
    if tables.shape[1] < 2 :
        print(f" Feuille ignorée (moins de 2 colonnes)")
    
        # Extraction des deux premières colonnes (tables et description des tables)
    noms_tables = tables.iloc[:, 0].astype(str).tolist()
    descriptions = tables.iloc[:, 1].astype(str).tolist()

    tab = {}
    for table, desc in zip(noms_tables, descriptions):
        if table.strip(): 
            tab[table.strip()] = desc.strip()
    
    #print(tab)
    result_tablesD.append(tab)
    return result_tablesD


info = extract_tableDescription("info_supp.xlsx")

# Recupération du nom des tables et leurs descriptions

synthese_d = 'Synthèse_Daily.xlsx'
#catalog_table_desc = extract_tableDescription(synthese_d)

catalog_table_desc = ""
with open("All_tables_descriptions.json", "r", encoding="utf-8") as f:
    catalog_table_desc = json.load(f)

caracteristique = ""
    # Chargement des caractéristiques des tables 
with open("All_colonnes_descriptions.json", "r", encoding="utf-8") as f:
    caracteristique = json.load(f)

if not os.path.exists("chatlogs"): os.makedirs("chatlogs")

def load_history(log_path):

    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        raw = json.load(f)

    hist = []
    for m in raw:

        if "content_data" in m and isinstance(m["content_data"], list):
            m["content_data"] = pd.DataFrame(m["content_data"])
            #hist.append(m["content_data"])

        hist.append(m)

    return hist


def make_json_serialisable(obj):
    """
    Remplace :
        Decimal  --  float
        DataFrame --  obj.to_dict()
        autres objets non‐sérialisables --  str(obj)
    """
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (list, tuple)):
        return [make_json_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_serialisable(v) for k, v in obj.items()}
    return obj   # str(), int, float, None, etc.


def save_history(hist, log_path):

    """Sauvegarde sans erreur de JSON (Decimal, DataFrame, …)"""
    safe_hist = make_json_serialisable(hist)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(safe_hist, f, indent=2, ensure_ascii=False)

# l'espace de l'utilisateur courant
def get_user_dir(username):
    return os.path.join("chatlogs", username)

# historique du chat
def get_chat_path(username, chat_name):
    return os.path.join(get_user_dir(username), f"{chat_name}.json")

# listes des conversations de l'utilisateur
def list_chats(username):
    user_dir = get_user_dir(username)
    #user_txt = f"{user_dir}.txt"
    #st.write(user_dir)
    if not os.path.exists(user_dir):
        return []
    chats_hist = [
        f for f in os.listdir(user_dir) 
        if f.endswith(".json") and f.startswith("Chat_") and len(f.split("_")) == 3
                ]
    #chats_hist = [f.replace(".json", "") for f in os.listdir(user_dir) if f.endswith(".json") and f != "infos_user.json"]
    
    return chats_hist


def Auto_charts(df=None, path="Image") :

    #df = pd.read_csv(uploaded_file, sep=";")
    #st.write("Aperçu des données :", df.head())
    charts = []
    folder = path
    i = 0

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    an = ""
    if "year" in numeric_cols :
        if df["year"].nunique() == 1 :
            an = str(df["year"].iloc[0])
            #an = df["year"].unique()

    mois_dict = {
        1: "Jan", 2: "Fév", 3: "Mars", 4: "Avr",
        5: "Mai", 6: "Jui", 7: "Juil", 8: "Août",
        9: "Sept", 10: "Oct", 11: "Nov", 12: "Déc"
    }

    #st.subheader("Courbes d’évolution temporelles")
    for time_col in [c for c in df.columns if "year" in c.lower() or "month" in c.lower() or "day" in c.lower()]:
        for num in numeric_cols:
            if num == "year" or num == "month" or num == "day":
                continue
            elif df[time_col].nunique() > 1:
                #df_t = df.groupby(time_col)[num].sum().reset_index()

                # Si c'est une colonne de mois numériques, la convertir en noms
                if time_col == "month" :
                    df_t = df.groupby(["year", "month"])[num].sum().reset_index()
                    # Création d'une date fictive pour le tri
                    df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                    
                    # Création de l'étiquette ex: "Déc-2024"
                    df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)
                    
                    # Tri chronologique
                    df_t = df_t.sort_values("date")

                    if df_t[num].sum() == 0 :
                        continue
                    else :
                        # Tracer la courbe
                        fig = px.line(df_t, x="MONTH", y=num, markers=True, title=f"Évolution de {num} <br>suivant les {time_col}")
                        fig.update_xaxes(tickangle=30)
                        fig.write_image(f"{folder}/{i}.png")
                        i = i + 1
                        #img = Image.open(f"{folder}/{i}.png")
                        charts.append(("Courbe", f"{num}__vs__{time_col}", fig))
                        #df_t = df.groupby(time_col)[num].sum().reset_index()
                        #df_t[time_col] = df_t[time_col].map(mois_dict)
                elif time_col == "year" :
                    df_t = df.groupby(time_col)[num].sum().reset_index()

                    # Convertir l'axe x en chaîne de caractères pour éviter les décimales
                    df_t["YEARS"] = df_t[time_col].astype(object)
                    if df_t[num].sum() == 0 :
                        continue
                    else :
                        #fig = st.plotly_chart(px.line(df_t, x=time_col, y=num, title=f"Évolution de {num} suivant les {time_col}"))
                        fig = px.line(df_t, x="YEARS", y=num, markers=True, title=f"Évolution de {num} <br>suivant les {time_col}")
                        fig.write_image(f"{folder}/{i}.png")
                        #img = Image.open(f"{folder}/{i}.png")
                        charts.append(("Courbe", f"{num} vs {time_col}", fig))

                        #st.plotly_chart(fig, use_container_width=True)

                        i = i + 1

                elif time_col == "day" :
                    df_t = df.groupby(time_col)[num].sum().reset_index()

                    df_t["day"] = pd.to_datetime(df_t["day"], yearfirst=True, format="%Y%m%d")
                    df_t = df_t.sort_values(by="day", ascending=False)
       #Ajouter une colonne "jour" qui contient le jour du mois
                    df_t["jour"] = df_t["day"].dt.day
                    fig = px.line(df_t, x="day", y=num, markers=True, title=f"Évolution de {num} <br>suivant les {time_col}")
                    fig.write_image(f"{folder}/{i}.png")
                    charts.append(("Courbe", f"{num}_vs_{time_col}", fig))
                    i = i + 1
                else : 
                    continue

            #else :
            #    st.plotly_chart(px.line(df, x=time_col, y=num, title=f"Évolution de {num} selon {time_col}"))

    
    #st.subheader("Moyennes")
    #nb = len(numeric_cols)
    for num in numeric_cols:
        
        if "year" in num.lower() or "mois" in num.lower() or "month" in num.lower():
            continue
        #st.metric(label=f"**Moyenne de {num}**", value=round(df[num].mean(), 2))
    

    #st.subheader("Graphiques Catégorielles vs Numériques")
    j = i 
    for cat in categorical_cols:
        if cat == "msisdn" or cat == "MSISDN" :
            continue

        for num in numeric_cols:
            if num == "year" or num == "month" :
                continue

            else :
                if df[cat].nunique() <= 1:
                    continue
                #print(df[cat].nunique())
                if df[cat].nunique() <= 7:
                    #st.plotly_chart(px.pie(df, names=cat, values=num, title=f"Répartition des {cat}"))
                    fig = px.pie(df, names=cat, values=num, title=f"Répartition des {cat} <br>par rapport au {num}")
                    fig.write_image(f"{folder}/{j}.png")
                    charts.append(("Pie chart", f"{num} : vs : {cat}", fig))

                    #st.plotly_chart(fig, use_container_width=True)
                    j = j + 1
                else :
                    
                    df_sorted = df.groupby(cat)[num].sum().sort_values(ascending=False)
                    df_5P = df_sorted.head(10).reset_index()
                    df_5D = df_sorted.tail(5).reset_index()
                    #fig0 = st.plotly_chart(px.bar(df_5P, x=cat, y=num, title=f"Top 10 des {cat}", color=cat))
                    #fig1 = st.plotly_chart(px.bar(df_5D, x=cat, y=num, title=f"Last 10 des {cat}", color=cat))

                    fig0 = px.bar(df_5P, x=cat, y=num, title=f"Top 10 des {cat} par<br> rapport au {num}")
                    fig1 = px.bar(df_5D, x=cat, y=num, title=f"Last 5 des {cat} par<br> rapport au {num}")
                    fig0.write_image(f"{folder}/{j}.png")
                    j = j + 1
                    fig1.write_image(f"{folder}/{j}.png")
                    #fig0.write_html(include_plotlyjs = True)
                    #st.plotly_chart(px.pie(df, names=cat, values=num, title=f"Répartition de {num} par {cat}"))
                    charts.append(("Bar chart", f"{num} vs {cat}", fig0))
                    charts.append(("Bar chart 1", f"{num}_vs_{cat}", fig1))

                    #st.plotly_chart(fig0, use_container_width=True)
                    #st.plotly_chart(fig1, use_container_width=True)
                    
                    j = j + 1
    
    return len(charts), charts


def display_charts(df: pd.DataFrame) :
    #print(df.columns)

    #df = pd.read_csv(uploaded_file, sep=";")
    #st.write("Aperçu des données :", df.head())
    charts = []

    #numeric_cols = df.select_dtypes(include='number').columns.tolist()
    #categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    #st.write(f"Var num {numeric_cols}")
    categorical_cols = df.select_dtypes(exclude=['int', 'float']).columns.tolist()

    num_cols = []
    cols_sup = []

    mois_dict = {
        1: "Jan", 2: "Fév", 3: "Mars", 4: "Avr",
        5: "Mai", 6: "Jui", 7: "Juil", 8: "Août",
        9: "Sept", 10: "Oct", 11: "Nov", 12: "Déc"
    }

    PALETTE = px.colors.qualitative.Plotly

    #st.subheader("Courbes d’évolution temporelles")
    for time_col in [c for c in df.columns if "year" in c.lower() or "month" in c.lower() or "day" in c.lower()]:

        #détecter automatiquement les colonnes ca
        ca_cols = [col for col in numeric_cols if "ca_" in col.lower() or "_ca" in col.lower() or "chiffre_affaire" in col.lower() or "montant" in col.lower() or "revenu" in col.lower() or "amount" in col.lower() or "mnt" in col.lower() or " ca " in col.lower()]
        vol_cols = [col for col in numeric_cols if "volum" in col.lower() or "minute" in col.lower() or "second" in col.lower() or "trafic" in col.lower() or "traffic" in col.lower()]
        parc_cols = [col for col in numeric_cols if "parc" in col.lower() or "sortant" in col.lower() or "acquis" in col.lower() or "reactiv" in col.lower() or "client" in col.lower() or "nombr" in col.lower() or "cnt" in col.lower()]
        
        if len(ca_cols) > 0 and not any("volum" in elem.lower() for elem in ca_cols) :
            if df[time_col].nunique() > 1:
                # Si c'est une colonne de mois numériques, la convertir en noms
                if time_col == "month" :
                    df_t = df.groupby(["year", "month"])[ca_cols].sum().reset_index()
                    # Création d'une date fictive pour le tri
                    df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                    
                    # Création de l'étiquette
                    df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)
                    
                    # Tri chronologique
                    df_t = df_t.sort_values("date")

                    #transformer le DataFrame pour plotly
                    df_melted = df_t.melt(
                        id_vars="MONTH",
                        value_vars=ca_cols,
                        var_name="CA",
                        value_name="Valeur"
                    )
                    #tracer les courbe
                    fig = px.line(
                        df_melted,
                        x="MONTH",
                        y="Valeur",
                        color="CA",
                        markers=True,
                        color_discrete_sequence= PALETTE,
                        title="Évolution mensuelle des différents <br>types de CA"
                    )

                    fig.update_layout(
                        xaxis_title="Mois",
                        yaxis_title="Valeur (CA)",
                        legend_title="Type de CA",
                        title_x=0.3,
                        hovermode="x unified"
                    )
                    charts.append(("Courbe", f"Les CA_vs_{time_col}", fig))
                if time_col == "day" :
                    df_t = df.groupby(["year","day"])[ca_cols].sum().reset_index()
                    df_t["day"] = pd.to_datetime(df_t["day"], yearfirst=True, format="%Y%m%d")
                    df_t = df_t.sort_values(by="day", ascending=False)
       #Ajouter une colonne "jour" qui contient le jour du mois
                    df_t["Jours"] = df_t["day"].dt.day
                    #transformer le DataFrame pour plotly
                    df_melted = df_t.melt(
                        id_vars="day",
                        value_vars= ca_cols,
                        var_name="CA",
                        value_name="Valeur"
                    )
                    #tracer les courbe
                    fig = px.line(
                        df_melted,
                        x="day",
                        y="Valeur",
                        color="CA",
                        markers=True,
                        color_discrete_sequence= PALETTE,
                        title="Évolution quotidienne du CA"
                    )

                    fig.update_layout(
                        xaxis_title="Jours",
                        yaxis_title="CA",
                        legend_title="Type de CA",
                        title_x=0.3,
                        hovermode="x unified"
                    )
                    fig.write_image("C:/Users/stg_gassama92247/Desktop/Sonatel/TEST.png")
                    charts.append(("Courbe", f"ca_day_vs_{time_col}", fig))

            ca_cols_sup = [x for x in numeric_cols if x not in ca_cols and x not in parc_cols and x not in vol_cols]
            #st.write(f"ca : {ca_cols_sup}")
            cols_sup += ca_cols_sup

        if len(vol_cols) > 0:
        #elif len(vol_cols) > 0:
            if df[time_col].nunique() > 1:
                # Si c'est une colonne de mois numériques, la convertir en noms
                if time_col == "month" :
                    df_t = df.groupby(["year", "month"])[vol_cols].sum().reset_index()
                    # Création d'une date fictive pour le tri
                    df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                    
                    # Création de l'étiquette
                    df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)
                    
                    # Tri chronologique
                    df_t = df_t.sort_values("date")

                    #transformer le DataFrame pour plotly
                    df_melted = df_t.melt(
                        id_vars="MONTH",
                        value_vars= vol_cols,
                        var_name="Volume",
                        value_name="Valeur"
                    )
                    #tracer les courbe
                    fig = px.line(
                        df_melted,
                        x="MONTH",
                        y="Valeur",
                        color="Volume",
                        markers=True,
                        color_discrete_sequence= PALETTE,
                        title="Évolution mensuelle du volume"
                    )

                    fig.update_layout(
                        xaxis_title="Mois",
                        yaxis_title="Volume",
                        legend_title="Volume",
                        title_x=0.3,
                        hovermode="x unified"
                    )
                    charts.append(("Courbe", f"volume_vs_{time_col}", fig)) 
                if time_col == "day" :
                    df_t = df.groupby(["year","day"])[vol_cols].sum().reset_index()
                    df_t["day"] = pd.to_datetime(df_t["day"], yearfirst=True, format="%Y%m%d")
                    df_t = df_t.sort_values(by="day", ascending=False)
       #Ajouter une colonne "jour" qui contient le jour du mois
                    df_t["Jours"] = df_t["day"].dt.day
                    #transformer le DataFrame pour plotly
                    df_melted = df_t.melt(
                        id_vars="day",
                        value_vars= vol_cols,
                        var_name="Volume",
                        value_name="Valeur"
                    )
                    #tracer les courbe
                    fig = px.line(
                        df_melted,
                        x="day",
                        y="Valeur",
                        color="Volume",
                        markers=True,
                        color_discrete_sequence= PALETTE,
                        title="Évolution quotidienne du Volume"
                    )

                    fig.update_layout(
                        xaxis_title="Jours",
                        yaxis_title="Volume",
                        legend_title="Volume",
                        title_x=0.3,
                        hovermode="x unified"
                    )
                    charts.append(("Courbe", f"vol_day_vs_{time_col}", fig))

            vol_cols_sup = [x for x in numeric_cols if x not in vol_cols and x not in ca_cols and x not in parc_cols]
            #st.write(f"vol : {vol_cols_sup}")
            cols_sup += vol_cols_sup

        if len(parc_cols) > 0 :
            if df[time_col].nunique() > 1:
                # Si c'est une colonne de mois numériques, la convertir en noms
                if time_col == "month" :
                    df_t = df.groupby(["year", "month"])[parc_cols].sum().reset_index()
                    # Création d'une date fictive pour le tri
                    df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                    
                    # Création de l'étiquette
                    df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)
                    
                    # Tri chronologique
                    df_t = df_t.sort_values("date")

                    #transformer le DataFrame pour plotly
                    df_melted = df_t.melt(
                        id_vars="MONTH",
                        value_vars= parc_cols,
                        var_name="Parc",
                        value_name="Valeur"
                    )
                    #tracer les courbe
                    fig = px.line(
                        df_melted,
                        x="MONTH",
                        y="Valeur",
                        color="Parc",
                        markers=True,
                        color_discrete_sequence= PALETTE,
                        title="Évolution mensuelle du parc"
                    )

                    fig.update_layout(
                        xaxis_title="Mois",
                        yaxis_title="Parc",
                        legend_title="Parc",
                        title_x=0.3,
                        hovermode="x unified"
                    )
                    charts.append(("Courbe", f"parc_vs_{time_col}", fig))
                    ############################################ DAY
                if time_col == "day" :
                    df_t = df.groupby(["year","day"])[parc_cols].sum().reset_index()
                    df_t["day"] = pd.to_datetime(df_t["day"], yearfirst=True, format="%Y%m%d")
                    df_t = df_t.sort_values(by="day", ascending=False)
       #Ajouter une colonne "jour" qui contient le jour du mois
                    df_t["Jours"] = df_t["day"].dt.day
                    #transformer le DataFrame pour plotly
                    df_melted = df_t.melt(
                        id_vars="day",
                        value_vars= parc_cols,
                        var_name="Parc",
                        value_name="Valeur"
                    )
                    #tracer les courbe
                    fig = px.line(
                        df_melted,
                        x="day",
                        y="Valeur",
                        color="Parc",
                        markers=True,
                        color_discrete_sequence= PALETTE,
                        title="Évolution quotidienne du parc"
                    )

                    fig.update_layout(
                        xaxis_title="Jours",
                        yaxis_title="Parc",
                        legend_title="Parc",
                        title_x=0.3,
                        hovermode="x unified"
                    )
                    charts.append(("Courbe", f"parc_day_vs_{time_col}", fig))

            parc_cols_sup = [x for x in numeric_cols if x not in parc_cols and x not in ca_cols and x not in vol_cols]
            #st.write(f"parc : {parc_cols_sup}")
            cols_sup += parc_cols_sup 
            #st.write(f"col : {set(cols_sup)}")

        if ca_cols == [] and vol_cols == [] and parc_cols == [] :
            num_cols = numeric_cols
        else : 
            #cols_num = numeric_cols
            #st.write(cols_num)
            #for i in cols_sup :
            #    st.write(i)
            #    cols_num.remove(i)
            #num_cols = [x for x in numeric_cols if x in cols_sup]
            cols_sup = list(set(cols_sup))
            num_cols = cols_sup
            #st.write(num_cols)

        #else :
        #    continue
            #num_cols = [x for x in numeric_cols if x not in cols_sup]
            #num_cols = numeric_cols



        ###########################################################################
        #           Courbe indivuduelle
        ###########################################################################
        #num_cols = [x for x in numeric_cols if x in cols_sup]
        #num_cols = [x for x in numeric_cols if x not in cols_sup]
        #st.write(f"Var : {num_cols}, {cols_sup}")
        #for num in numeric_cols:
        for num in num_cols:
            if num == "year" or num == "month" or num == "day":
                continue
            elif df[time_col].nunique() > 1:
                # Si c'est une colonne de mois numériques, la convertir en noms
                if time_col == "month" :
                    df_t = df.groupby(["year", "month"])[num].sum().reset_index()
                    # Création d'une date fictive pour le tri
                    df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                    
                    # Création de l'étiquette
                    df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)
                    
                    # Tri chronologique
                    df_t = df_t.sort_values("date")

                    if df_t[num].sum() == 0 :
                        continue
                    else :
                        # Tracer la courbe
                        fig = px.line(df_t, x="MONTH", y=num, markers=True, color_discrete_sequence= PALETTE,
                                      title=f"Évolution de {num} <br>suivant les {time_col}")
                        fig.update_xaxes(tickangle=30)
                        
                        #img = Image.open(f"{folder}/{i}.png")
                        charts.append(("Courbe", f"{num}_vs_{time_col}", fig))
                    
                elif time_col == "year" :
                    df_t = df.groupby(time_col)[num].sum().reset_index()

                    # Convertir l'axe x en chaîne de caractères pour éviter les décimales
                    df_t["YEARS"] = df_t[time_col].astype(object)

                    if df_t[num].sum() == 0 :
                        continue
                    else :
                        #fig = st.plotly_chart(px.line(df_t, x=time_col, y=num, title=f"Évolution de {num} suivant les {time_col}"))
                        fig = px.line(df_t, x="YEARS", y=num, markers=True, color_discrete_sequence= PALETTE,
                                      title=f"Évolution de {num} <br>suivant les {time_col}")
                        
                        charts.append(("Courbe", f"{num}__vs__{time_col}", fig))

                elif time_col == "day" :
                    df_t = df.groupby(time_col)[num].sum().reset_index()

                    df_t["day"] = pd.to_datetime(df_t["day"], yearfirst=True, format="%Y%m%d")
                    df_t = df_t.sort_values(by="day", ascending=False)
       #Ajouter une colonne "jour" qui contient le jour du mois
                    df_t["jour"] = df_t["day"].dt.day
                    fig = px.line(df_t, x="day", y=num, markers=True, color_discrete_sequence= PALETTE, 
                                  title=f"Évolution de {num} <br>suivant les {time_col}")
                        
                    charts.append(("courbE", f"{num}_vs_{time_col}", fig))

                else :
                    continue
   
    #st.subheader("Graphiques Catégorielles vs Numériques")
    for cat in categorical_cols:
        #print(cat, df[cat].dtype)
        if cat == "msisdn" or cat == "MSISDN" :
            continue

        for num in numeric_cols:
            if num == "year" or num == "month" or num == "day" :
                continue
            else :
                if df[cat].nunique() <= 1:
                    continue
                #print(cat, df[cat].dtype)
                #else : 
                if df[cat].nunique() <= 7:
                    if "month" in df.columns and df["month"].nunique() > 1 :
                        #print(df.columns)
                        #df_t = df
                        df_t = df.groupby(["year", "month", cat])[num].sum().reset_index()
                        # Création d'une date fictive pour le tri
                        df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                        
                        # Création de l'étiquette
                        df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)
                        # Tri chronologique
                        #df_t = df_t.sort_values("date")
                        #df_t = df_t.sort_values(["date", num], ascending=[True, False])

                        if df_t[num].sum() == 0 :
                            continue
                        else :
                            df_t = df_t.sort_values("date")
                            fig = px.line(df_t, x="MONTH", y=num, markers=True, color= cat, color_discrete_sequence= PALETTE, 
                                          title=f"Évolution du {num} par rapport <br>au {cat} suivant les Month")
                            charts.append(("line", f"{num} vs {cat}", fig))
                    else :
                        if df[num].sum() == 0 :
                            continue
                        else :
                            fig = px.pie(df, names=cat, values=num, color_discrete_sequence= PALETTE, #color_discrete_map = PALETTE,
                                     title=f"Répartition des {cat} <br>par rapport au {num}")

                    #fig = px.pie(df, names=cat, values=num, title=f"Répartition des {cat} <br>par rapport au {num}")
                            charts.append(("Pie chart_or_line", f"{num} vs {cat}", fig))

                else :
                    if "month" in df.columns and df["month"].nunique() > 0 :
                        #print(df.columns)
                        #df_t = df
                        df_t = df.groupby(["year", "month", cat])[num].sum().reset_index()
                        # Création d'une date fictive pour le tri
                        df_t["date"] = pd.to_datetime(dict(year=df_t["year"], month=df_t["month"], day=1))
                        
                        # Création de l'étiquette
                        df_t["MONTH"] = df_t.apply(lambda row: f"{mois_dict[row['month']]}-{row['year']}", axis=1)


                        #df_t = df_t[df_t[num].notna() & (df_t[num] > 0)]
                        #print(df_t.columns)
                        #df_t = df_t.sort_values(["date", num], ascending=[True, False]).groupby(["MONTH"]).head(3)
                        df_t = df_t.sort_values(["date", num], ascending=[True, False]).groupby(["year", "month","MONTH"]).head(3).reset_index()
                        #df_t = df_t.sort_values("date")
                        
                        # Étape 2 : Créer le graphique à barres groupées
                        if df_t[num].sum() == 0 :
                            continue
                        else :
                            fig = px.bar(
                                df_t,
                                x="MONTH",
                                y= num ,
                                color= cat,
                                color_discrete_sequence= PALETTE,
                                #color_discrete_map = PALETTE,
                                barmode="group",
                                text= cat,
                                title=f"Top 3 des {cat} les plus influentes par <br>rapport au {num} par mois"
                            )
                            fig.update_layout(
                                xaxis_title="MONTH",
                                yaxis_title=num,
                                legend_title=cat,
                                title_x=0.3,
                            #    bargap=0,          # enlève l’espace entre groupes
                            #    bargroupgap=0      # enlève l’espace entre barres d’un même groupe
                            )
                            charts.append(("Bar chart_group", f"{num} vs {cat}", fig))


                #else :
                #    df_sorted = df.groupby(cat)[num].sum().sort_values(ascending=False)
                #    df_5P = df_sorted.head(10).reset_index()
                #    df_5D = df_sorted.tail(5).reset_index()

                #    fig0 = px.bar(df_5P, x=cat, y=num, title=f"Top 10 des {cat} par<br> rapport au {num}")
                #    fig1 = px.bar(df_5D, x=cat, y=num, title=f"Last 5 des {cat} par<br> rapport au {num}")
                    
                    #st.plotly_chart(px.pie(df, names=cat, values=num, title=f"Répartition de {num} par {cat}"))
                #    charts.append(("Bar chart", f"{num} vs {cat}", fig0))
                #    charts.append(("Bar chart 1", f"{num} vs {cat}", fig1))
    
    return charts
    #return charts, cols_sup

# Trier les fichiers par date extraite du nom
def extract_datetime(filename):
    try:
        # filename = "Chat_20250818_121959.json"
        date_str = filename.split("_")[1] + filename.split("_")[2].replace(".json", "")
        return dt.strptime(date_str, "%Y%m%d%H%M%S")
    except Exception:
        return dt.min  # si erreur, mettre la date la plus ancienne

if "user_trouve" not in st.session_state:
    st.session_state.user_trouve = None

if "full_chat_history" not in st.session_state:
    st.session_state.full_chat_history = [] 

if "charts" not in st.session_state :
    st.session_state.charts = []

if "exec_req" not in st.session_state :
    st.session_state.exec_req = False

if "modif_req" not in st.session_state :
    st.session_state.modif_req = False

if "exec_clicked" not in st.session_state:
    st.session_state.exec_clicked = False
if "modif_clicked" not in st.session_state:
    st.session_state.modif_clicked = False

if "sql_request" not in st.session_state:
    st.session_state.sql_request = ""

if "user_msg" not in st.session_state:
    st.session_state.user_msg = None

if "analyse" not in st.session_state:
    st.session_state.analyse = False

if "show_admin" not in st.session_state:
    st.session_state.show_admin = False

if "is_admin" not in st.session_state:  
    st.session_state.is_admin = False


def create_user(username, password):
    user_dir = os.path.join("chatlogs", username)
    os.makedirs(user_dir, exist_ok=True)
    with open(os.path.join(user_dir, "infos_user.json"), "w") as f:
        json.dump({"username" : username,"password": password, "Admin" : "False"}, f)
    save_history([
        {
            "role": "Assistant",
            "content_request": f" Bienvenue {username} ",
            "content_data" : {}
        }], os.path.join(user_dir, f"Chat_{dt.now().strftime('%Y%m%d_%H%M%S')}.json"))
    #f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"    os.path.join(user_dir, "history.json"))
    

def check_users(username, password):
    path = os.path.join("chatlogs", username, "infos_user.json")
    if os.path.exists(path):
        with open(path) as f:
            stored = json.load(f)
            return stored.get("password") == password
    return False

def check_admin(username):
    path = os.path.join("chatlogs", username, "infos_user.json")
    if os.path.exists(path):
        with open(path) as f:
            stored = json.load(f)
            return stored.get("Admin") == "True"
    return False

# Style global
st.markdown("""
    <style>
        /* Positionne le menu principal en bas de l'écran */
        .bottom-menu {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f5f5f5;
            border-top: 1px solid #ddd;
            padding: 10px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
            z-index: 9999;
        }

        /* Boutons du menu */
        .stButton button {
            background-color: #ff8c00 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            background-color: #e07b00 !important;
            transform: scale(1.05);
        }

        /* Espace pour éviter que le menu ne recouvre le contenu */
        .content {
            padding-bottom: 90px;
        }
            
        /* Carte du profil utilisateur */
        .profile-card {
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            padding: 30px;
            margin-top: 30px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
        }
        .profile-avatar {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 15px;
            margin-left: 30px;
            border: 3px solid #ff8c00;
        }
    </style>
""", unsafe_allow_html=True)

#Interface utilisateur pour entrer le nom
if st.session_state.user_trouve is None:

    st.title(" Connexion / Inscription")
    tab1, tab2 = st.tabs(["Connexion", "Inscription"])

    with tab1:
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            if check_users(username, password):
                st.session_state.user_trouve = username
                #path = os.path.join("chatlogs", username, "history.json")

                user_dir = os.path.join("chatlogs", username)

                st.rerun()

            else:
                st.error(" Nom d'utilisateur ou mot de passe incorrect. ")

    with tab2:
        new_user = st.text_input("Choisir un nom d'utilisateur")
        new_pass = st.text_input("Choisir un mot de passe", type="password")
        if st.button("Créer mon compte") :
            user_path = os.path.join("chatlogs", new_user)
            if os.path.exists(user_path) :
                st.warning("Ce nom d'utilisateur existe déjà.")
            else:
                create_user(new_user, new_pass)
                st.success("Compte créé avec succès. Vous pouvez maintenant vous connecter.")        

else :
    user = st.session_state.user_trouve
    user_dir = get_user_dir(user)

    #st.sidebar.title(f" Vos conversations ({user})")
    st.sidebar.success(f" Connecté en tant que ****{user}**** ")
    #print(check_admin(user))

    #if check_admin(user) : 
    #    st.session_state.is_admin = True

    ###########################################
    #       Modifications Paramètres
    ###########################################
    st.markdown('<div class="bottom-menu">', unsafe_allow_html=True)
    # Container du bas
    with st.sidebar.expander(" ⚙️ Paramétres ", expanded = False):
        
        col1, a = st.columns(2)
        with col1:
            if st.button(" Profil ", use_container_width=True):
                st.session_state.page = "profil"

        if check_admin(user) :
            coll, b = st.columns(2)
            with coll:
                if st.button(" Admin ", use_container_width=True):
                    st.session_state.page = "admin"

        col, c = st.columns(2)
        with col :
            if st.button(" Chats ", use_container_width=True):
                st.session_state.page = "chats"

        col2, d = st.columns(2)
        with col2:
            if st.button(" Déconnexion ", use_container_width=True):
                st.session_state.page = "decon"

        # Vérification User Admin
        ##############################
        ##############################
        

    st.markdown('<div>', unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Exemple de contenu central selon la page
    page = st.session_state.get("page", "chats")

    if page == "decon":
        #st.title("💬 Interface de conversation")
        st.session_state.clear()
        st.session_state.user_trouve = None
        st.session_state.full_chat_history = []
        st.session_state.current_chat = None
        #st.session_state.show_admin = False
        #st.session_state.is_admin = False
        st.session_state.sql_request = ""
        st.session_state.modif_clicked = False
        st.session_state.exec_clicked = False
        st.session_state.exec_req = False
        st.session_state.modif_req = False
        st.session_state.last_result = None
        st.rerun()
        
    elif page == "admin":
        print()
    
        st.title("Dashboard Administrateur")

        try:
            admin = pd.read_csv("Admin.csv", sep= ";")
            #print(admin.columns)
            #admin.head()
            admin["date"] = pd.to_datetime(admin["date"], yearfirst=True, format="%Y-%m-%d %H:%M")
            admin = admin.sort_values(by="date", ascending=False)

            #identifier le mois le plus récent
            mois_recent = admin["date"].dt.month.iloc[0]
            annee_recente = admin["date"].dt.year.iloc[0]

            #filtrer les données correspondant au mois en cours
            df_mois_recent = admin[(admin["date"].dt.month == mois_recent) & (admin["date"].dt.year == annee_recente)]
            #Ajouter une colonne "jour" qui contient le jour du mois
            df_mois_recent["jour"] = df_mois_recent["date"].dt.day

            # Statistiques globales
            nb_request = df_mois_recent['nb requete'].sum()
            cout_total = df_mois_recent['couts'].sum()
            last_question = df_mois_recent["questions"].iloc[0]
            cout_last_question = df_mois_recent['couts'].iloc[0]
            temps_last_question = df_mois_recent['temps'].iloc[0]

            # Affichage métriques
            st.subheader(" Les requêtes SQL ")
            col1, col2, col3 = st.columns(3)
            col1.metric(f" *Nombre total de requêtes SQL :* ", nb_request)
            col2.metric(F" *Coût total des requêtes SQL :* ", f"{cout_total:.4f} $")
            col3.metric(f" *Durée dernière requête SQL:* ", f"{temps_last_question:.4f} secondes")

            st.subheader(" Dernière requête SQL exécutée ")
            col1, col2 = st.columns(2)
            #col1.metric(f" *Requête :* ", f" {last_question} ")
            col1.write(f" *Requête :* \n\n\n {last_question} ")
            col2.metric(f" *Coût :* ", f" {cout_last_question:.4f} $ ")
            #st.write(f"**Requête : ** {last_question}")
            #st.write(f"**Coût : ** {cout_last_question:.2f} $ ")

            #courbe d'évolution du nombre de requêtes
            df_mois_recent = df_mois_recent[["nb requete", "jour", "couts", "temps"]]
            df_mois_recent = df_mois_recent.groupby("jour").sum().reset_index()
            fig_requetes = px.line(
                df_mois_recent,
                x="jour",
                y="nb requete",
                markers=True,
                title="Evolution du nombre de requêtes durant le mois",
                labels={"jour": "Jour du mois", "nb requete": "Nombre de requêtes"}
            )

            #courbe d'évolution des coûts
            fig_couts = px.line(
                df_mois_recent,
                x="jour",
                y="couts",
                markers=True,
                title="Evolution des coûts durant le mois",
                labels={"jour": "Jour du mois", "couts": "Coûts ($)"}
            )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_requetes, use_container_width=True)
            with col2:
                st.plotly_chart(fig_couts, use_container_width=True)

            ######### Dash Analyses

            admin_ana = pd.read_csv("Admin_ana.csv", sep= ";")
            #print(admin_ana.columns)
            #admin_ana.head()
            admin_ana["date"] = pd.to_datetime(admin_ana["date"], yearfirst=True, format="%Y-%m-%d %H:%M:%S")
            admin_ana = admin_ana.sort_values(by="date", ascending=False)

            nb_request_ana = admin_ana['nb requete'].sum()
            cout_total_ana = admin_ana['couts'].sum()

            st.subheader(" Requêtes d'analyses ")
            col1, col2 = st.columns(2)
            col1.metric(f" *Nombre requête :* ", f" {nb_request_ana} ")
            col2.metric(f" *Coût :* ", f" {cout_total_ana:.2f} $ ")

            st.subheader(" Historique complet des requêtes SQL ")
            st.dataframe(admin)

            st.subheader(" Historique complet des requêtes d'analyses ")
            st.dataframe(admin_ana)

        except FileNotFoundError:
            st.error(f" **Erreur de chargement de l'historique.** ")

    elif page == "profil" :
        path_info = os.path.join("chatlogs", user, "infos_user.json")
        with open(path_info) as f:
            info_user = json.load(f)

        st.title("👤 Profil utilisateur")

        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.session_state.file_path = info_user.get("photo_path")
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            st.image(st.session_state.file_path, width=120)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                    width=120, caption=None)
        st.subheader(info_user.get("username"))
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # --- MODIFICATION DU PROFIL ---
        st.subheader("✏️ Modifier les informations")

        new_nom = st.text_input("Nom d'utilisateur", info_user.get("username"))
        #new_email = st.text_input("Adresse email", user["email"])
        new_mdp = st.text_input("Mot de passe", info_user.get("password"))

        uploaded_photo = st.file_uploader("📸 Changer la photo de profil", type=["jpg", "png", "jpeg"])
        if uploaded_photo is not None:
            image = Image.open(uploaded_photo)
            PROFILE_DIR = os.path.join("chatlogs", user)
            st.session_state.file_path = os.path.join(PROFILE_DIR, f"{new_nom.replace(' ', '_')}.png")
            image.save(st.session_state.file_path)
            info_user.update({"photo_path" : st.session_state})
            #st.session_state.user["photo_path"] = file_path
            st.image(st.session_state.file_path, caption="Nouvelle photo enregistrée", width=120)

        if st.button("💾 Enregistrer les modifications"):
            info_user["username"] = new_nom
            info_user["password"] = new_mdp
            info_user["photo_path"] = st.session_state.file_path
            with open(path_info, "w", encoding="utf-8") as f:
                json.dump(info_user, f, indent=2, ensure_ascii=False)
            st.rerun()
            st.success(" ✅ Profil mis à jour ! ")          

    elif page == "chats" :
    # --- Bouton pour créer un nouveau chat ---
        if st.sidebar.button(" ➕ Nouveau Chat ", use_container_width= True):
            chat_name = f"Chat_{dt.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.current_chat = chat_name
            st.session_state.full_chat_history = [
                {
                "role": "Assistant",
                "content_request": f" Bienvenue {st.session_state.user_trouve} ",
                "content_data" : {}
                }
                ]
            st.session_state.sql_request = ""
            st.session_state.modif_clicked = False
            st.session_state.exec_clicked = False
            st.session_state.exec_req = False
            st.session_state.modif_req = False
            st.session_state.last_result = None
            st.session_state.user_msg = None
            save_history(st.session_state.full_chat_history, get_chat_path(user, chat_name))
            st.rerun()

        # --- Liste des chats existants ---
        chats = list_chats(user)
        chats = sorted(chats, key=extract_datetime, reverse=True)
        chats = [f.replace(".json", "") for f in chats]
        
        if chats:
            choice = st.sidebar.radio(" Vos conversations : ", chats, index=0)
            if "current_chat" not in st.session_state or st.session_state.current_chat != choice:
                st.session_state.current_chat = choice
                st.session_state.full_chat_history = load_history(get_chat_path(user, choice))
                #st.write(chats.sort(reverse=True))


        log_path  = os.path.join("chatlogs", user, "history.json")
        user_name = user.replace(".json", "")

        st.title("🤖 Requetage en langue naturelle")
        #st.subheader("Quelles informations voulez-vous savoir ?")

        # Affichage de l'historique de l'utilisateur
        x = {}
        for m in st.session_state.full_chat_history:
            #st.write(f"{m} \n {type(m)}")
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    st.markdown(m["content_request"])
                #elif m["role"] == "assistant" and "content_data" in m :
                #    st.markdown(m["content_request"])

                else :
                    #st.markdown(m["content_request"])
                    st.code(m["content_request"], language="sql")
                    if not m["content_data"] :
                        continue
                    else :
                        df = pd.DataFrame(m["content_data"])
                        df.to_csv("data.csv", sep= ";")
                        df = pd.read_csv("data.csv", sep = ";")
                        if "Unnamed: 0" in df.columns:
                            df.drop("Unnamed: 0", axis=1, inplace=True)

                        #print(df.info())
                        st.dataframe(m["content_data"])

                        char = display_charts(df)
                        #st.write(x)
                        if char != [] :
                        #for idx, (gtype, label, fig) in enumerate(st.session_state.charts):
                            st.subheader("Le Dashboard des résulats de la requete")

                        cols = st.columns(2, gap="large")
                        id = 0
                        
                        for idx, (gtype, label, fig) in enumerate(char):
                            #identifiant aléatoire 
                            unique_id = uuid.uuid4().hex[:8]  
                            key = f"{label}_{unique_id}"
                            #key = f"{label}_{id}"
                            with cols[idx % 2]:
                                st.plotly_chart(fig, use_container_width=True, height=250, key= key)
                            
                            id = id + 1

                    kpi_res = m.get("kpi_result")
                    if kpi_res is None : 
                    #    display_charts(kpi_res)
                    #if not m["kpi_result"] :
                        continue
                    else :
                        #df = pd.DataFrame(m["kpi_result"])
                        df = pd.DataFrame(kpi_res)
                        df.to_csv("Data.csv", sep= ";")
                        df = pd.read_csv("Data.csv", sep = ";")
                        if "Unnamed: 0" in df.columns:
                            df.drop("Unnamed: 0", axis=1, inplace=True)

                        #print(df.info())
                        #st.dataframe(df)

                        char = display_charts(df)
                        if char != [] :
                        #for idx, (gtype, label, fig) in enumerate(st.session_state.charts):
                            st.subheader("Les données et le Dashboard des KPI pour l'explication des résultats de la requete")
                        
                        st.dataframe(df)
                        
                        cols = st.columns(2, gap="large")
                        #id = 0
                        for idx, (gtype, label, fig) in enumerate(char) :
                            #identifiant aléatoire 
                            unique_id = uuid.uuid4().hex[:8]  
                            key = f"{label}_{unique_id}"
                            #key = f"{label}_{id}"
                            with cols[idx % 2]:
                                st.plotly_chart(fig, use_container_width=True, height=250, key= key)


                    recomd = m.get("recommandation")
                    if recomd is not None :
                        st.markdown(recomd) 
                    else :
                        continue

        user_msg = st.chat_input("Ecrivez votre message…")
        #send = False
        if user_msg:
            st.session_state.user_msg = user_msg

            # Affiche et stocke le message utilisateur
            st.chat_message("user").markdown(user_msg)
            st.session_state.full_chat_history.append({"role": "user", "content_request": user_msg})

            ##########   ##############
            #    Système 1
            ###################################
                
            #sql_request = system(catalog_table_desc, caracteristique, st.session_state.full_chat_history, info, user_msg )

            #st.session_state.sql_request = system(catalog_table_desc, caracteristique, st.session_state.full_chat_history, info, user_msg )

            ########### ######## #############
            #           Système 2
            ##################################
            path = f"{user_dir}/{user}"

            try :
                
                st.write(" Veuillez patientez, votre requete est en cours d'exécution ... ")
                start_time = time.time()

                #st.session_state.sql_request, request_cost = Agent_SQL_Gem(user_msg, path)
                st.session_state.sql_request = Agent_SQL_Gem(user_msg, path)
                #st.session_state.sql_request = SQL_Agent(user_msg, path)

                end_time = time.time()

                execution_time = end_time - start_time

                st.code(body=st.session_state.sql_request, language="sql")

                ### Gestion des caractéristiques de la requete

                #nouvelle_ligne = [1, user_msg, request_cost, execution_time, dt.now().strftime("%Y-%m-%d %H:%M"), st.session_state.sql_request]
                nouvelle_ligne = [1, user_msg, 0, execution_time, dt.now().strftime("%Y-%m-%d %H:%M"), st.session_state.sql_request]
                with open("Admin.csv", mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter= ";")
                    writer.writerow(nouvelle_ligne)

                st.write(f" Temps d'exécution : {execution_time:.4f} secondes ")
                st.session_state.exec_req = True
                st.session_state.modif_req = True

            except Exception as e :
                st.error(f"Le modèle ne répond pas pour le moment, veuillez réessayez plus tard : \n\n {e}.")
                st.session_state.sql_request = " "
                request_cost = 0
            
            
            #st.session_state.sql_request = sql_request
    
        if st.session_state.user_msg is not None:
            # Fonction pour la vérification du requete
            def is_not_sql_query(text: str) -> bool:
                sql_keywords = [
                    "SELECT", "GROUP BY", "ALTER", "TRUNCATE", "DISTINCT",
                    "FROM", "WHERE", "JOIN", "UNION", "WITH", "ORDER BY", "COUNT"
                ]
                
                text_upper = text.upper()
                return not any(keyword in text_upper for keyword in sql_keywords)
            
            if is_not_sql_query(st.session_state.sql_request) :
                print("")
                st.error(f"La requete n'a pas été produite. Soyez plus clair et plus précis dans votre question")
                
            else :

                B1,B2 = st.columns(2, gap="large")

                with B1:
                    if "exec_req" in st.session_state:
                        if st.button(" Exécuter la requête après modification ",use_container_width=True):
                            st.session_state.exec_clicked = True  

                with B2:
                    if "modif_req" in st.session_state:
                        if st.button(" Modifier ", use_container_width=True):
                            st.session_state.modif_clicked = True  

            #if "modif_req" in st.session_state :
            sql_request = st.session_state.sql_request
            
            if st.session_state.modif_clicked:
                #st.session_state.exec_clicked = False
                sql_request = st.text_area("-- Requête SQL générée :\n\n", value=st.session_state.sql_request, height=400)

                if st.session_state.exec_clicked :                
                    st.session_state.sql_request = sql_request
                    #st.session_state.exec_clicked = False
                    st.session_state.modif_clicked = False

                    st.rerun()
            #else :
            #    a = st.code(body=st.session_state.sql_request, language="sql")

            #if "exec_req" in st.session_state :
            if st.session_state.exec_clicked:
                #st.session_state.analyse = True

                try :
                    #result_request = connect_db_railway(request=st.session_state.sql_request)
                    result_request = connect_db(request=st.session_state.sql_request)
                    #result_request =pd.read_csv("Résultat_requete.csv", sep= ";")
                    path_result= f"chatlogs/{user}/Resultat_requete.csv"
                    #data = result_request.to_csv("Resultat_requete.csv", sep=";")
                    data = result_request.to_csv(path_result, sep=";")

                    # Affiche et stocke la réponse
                    st.chat_message("assistant").code(body= f"-- Requête SQL envoyée :\n\n{st.session_state.sql_request}", language= "sql")
                    #st.chat_message("assistant").markdown(f"Requête SQL envoyée :\n\n{st.session_state.sql_request}\n\n")
                    st.chat_message("assistant").dataframe(result_request)

                    # Enregistrement dans l'historique
                    st.session_state.full_chat_history.append({
                        "role": "assistant",
                        "content_request": f"-- Requête SQL envoyée :\n\n{st.session_state.sql_request}\n\n",
                        #"content_data": result_request
                        "content_data": result_request.to_dict()
                        })

                    #result = pd.read_csv("Resultat_requete.csv", sep= ";")
                    result = pd.read_csv(path_result, sep= ";")
                    if "Unnamed: 0" in result.columns:
                        result.drop("Unnamed: 0", axis=1, inplace=True)

                    #st.session_state.last_result = result_request
                    st.session_state.last_result = result
                    #display_charts(st.session_state.last_result)
                    
                except Exception :
                    # Affiche et stocke la réponse
                    st.chat_message("assistant").markdown(f"Aucune requête SQL n'a été générée ")
                    
                    st.session_state.full_chat_history.append({
                            "role": "assistant",
                            "content_request": f"-- Aucune requête SQL n'a été générée",
                            "content_data": {}      
                            })
                
                #result_request =pd.read_csv("Resultat_requete.csv", sep= ";")
                #display_charts(st.session_state.last_result)
                #st.session_state.exec_clicked = False
                st.session_state.modif_clicked = False

                char = display_charts(st.session_state.last_result)
                if char != [] :
                #for idx, (gtype, label, fig) in enumerate(st.session_state.charts):
                    st.subheader("Le Dashboard des résulats de la requete")

                cols = st.columns(2, gap="large")
                id = 0
                
                for idx, (gtype, label, fig) in enumerate(char):
                    #identifiant aléatoire 
                    unique_id = uuid.uuid4().hex[:8]  
                    key = f"{label}_{unique_id}"
                    #key = f"{label}_{id}"
                    with cols[idx % 2]:
                        st.plotly_chart(fig, use_container_width=True, height=250, key= key)

                st.session_state.exec_clicked = False

            #if "last_result" in st.session_state and st.session_state.exec_req:
            #df = st.session_state.last_result
            if "last_result" in st.session_state and st.session_state.last_result is not None :
                df = st.session_state.last_result
                #list_img = display_charts(st.session_state.last_result)
                list_img = display_charts(df)
                nb_img = len(list_img)
                if ("year" in df.columns and df["year"].nunique() > 1) or ("month" in df.columns and df["month"].nunique() > 1) or ("day" in df.columns and df["day"].nunique() > 1) or (nb_img != 0) :
            # if "last_result" in st.session_state and (df["year"].nunique() > 1 or df["month"].nunique() > 1):
                #df = st.session_state.last_result
                #if df["year"].nunique() > 1 or df["month"].nunique() > 1 :
                #display_charts(st.session_state.last_result)
                #st.subheader("Actions supplémentaires")
                    if st.button(" Analyzer ", use_container_width=True):
                        try:
                            #pio.defaults.image_exporter = "kaleido"
                            #x = pio.kaleido.kaleido_available()
                            #st.write(f"kal : {x}")
                            #pio.defaults.image_format = "png"             # format d'export
                            #pio.defaults.image_scale = 2  

                            dossier_dash = f"chatlogs/{user}/{user}_dash"
                            os.makedirs(dossier_dash, exist_ok=True)
                            #st.session_state.last_result = pd.read_csv("C:/Users/stg_gassama92247/Desktop/Sonatel/Sonatel/App/Resultat_requete.csv", sep= ";")
                            #list_img = display_charts(st.session_state.last_result)
                            #nb_img = len(list_img)

                            #nb_img, char = Auto_charts(st.session_state.last_result)
                            for idx, (gtype, label, fig) in enumerate(list_img) : 
                                # CONFIGURATION PNG HAUTE QUALITÉ
                                pio.defaults.image_engine = "kaleido"
                                pio.defaults.image_format = "png"
                                pio.defaults.image_scale = 2
                                
                                #pio.to_image(fig, format="png", engine = "kaleido")
                                fig.write_image(f"{dossier_dash}/{idx}.png")
                                #pio.write_image(fig= fig, file= f"{dossier_dash}/{idx}.png", format="png", engine="kaleido", scale=2)
                            
                            echant = st.session_state.last_result.head()
                            echant = echant.to_dict(orient='records')
                            path = f"{user_dir}/{user}"

                            start_time = time.time()
                            
                            sql_query, info_kpi =Agent_KPI_req_gemini(echant, user_msg,st.session_state.sql_request, path)
                            #sql_query, info_kpi, reqest_cost =Agent_KPI_req(echant, user_msg,st.session_state.sql_request, path)

                            result_kpi = connect_db(request=sql_query)
                            #result_request =pd.read_csv("Résultat_requete.csv", sep= ";")
                            path_result= f"chatlogs/{user}/Resultat_kpi.csv"
                            #data = result_request.to_csv("Resultat_requete.csv", sep=";")
                            data = result_kpi.to_csv(path_result, sep=";")

                            kpi = pd.read_csv(path_result, sep= ";")
                            if "Unnamed: 0" in kpi.columns:
                                kpi.drop("Unnamed: 0", axis=1, inplace=True)

                            #img_nb, graphe = Auto_charts(kpi, 'img_kpi')
                            dossier_dash_kpi = f"chatlogs/{user}/{user}_dash_kpi"
                            os.makedirs(dossier_dash_kpi, exist_ok=True)
                            list_img_kpi = display_charts(kpi)
                            img_nb = len(list_img_kpi)
                            for idx, (gtype, label, fig) in enumerate(list_img_kpi) : 
                                fig.write_image(f"{dossier_dash_kpi}/{idx}.png")

                            if (img_nb < 1 or list_img_kpi == []) :
                                st.error("L'analyse complet n'a pas été faite, essayez de le relancé ")
                            else :
                                recomandation = analyse_recommand_gemini(img_dash=dossier_dash, img_dash_kpi=dossier_dash_kpi, request=user_msg,nb_imgD=nb_img, nb_imgKPI=img_nb, info_kpi=info_kpi)
                                st.session_state.full_chat_history[-1].update({"kpi_result" : kpi.to_dict()})

                                #########################""
                                st.dataframe(kpi)

                                ############################
                                char = display_charts(kpi)
                                
                                if char != [] :
                                #for idx, (gtype, label, fig) in enumerate(st.session_state.charts):
                                    st.subheader("Le Dashboard des KPI pour l'explication des résultats de la requete")
                                    #id = 0
                                cols = st.columns(2, gap="large")
                                for idx, (gtype, label, fig) in enumerate(char):
                                    #identifiant aléatoire 
                                    unique_id = uuid.uuid4().hex[:8]  
                                    key = f"{label}_{unique_id}"
                                    #key = f"{label}_{id}"
                                    with cols[idx % 2]:
                                        st.plotly_chart(fig, use_container_width=True, height=250, key= key)

                                #recomandation = analyse_recommand(request=user_msg, nb_img=nb_img)
                                #recomandation = analyse_recommand(request=st.session_state.user_msg, nb_img=nb_img)
                                st.chat_message("assistant").markdown(f" {recomandation} ")
                                
                                #st.session_state.full_chat_history[-1]["recommandation"] = recomandation
                                st.session_state.full_chat_history[-1].update({"recommandation" : recomandation})
                            

                            end_time = time.time()
                            execution_time = end_time - start_time

                            ################### Gestion des caractéristiques de l'analyse

                            #nouvelle_ligne = [1, reqest_cost, dt.now().strftime("%Y-%m-%d %H:%M:%S"), execution_time]
                            nouvelle_ligne = [ 1, 0, dt.now().strftime("%Y-%m-%d %H:%M:%S"), execution_time ]
                            with open("Admin_ana.csv", mode="a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f, delimiter= ";")
                                writer.writerow(nouvelle_ligne)
                            
                        except Exception as e:
                            st.error(f" Erreur lors de la génération des graphiques : {e}. ")
                            #reqest_cost = 0

                #else : 
                #    st.warning(" Exécuter la requete avant de cliquer sur le bouton Analyzer ")

        # Sauvegarde finale
        #save_history(st.session_state.full_chat_history, log_path)
        save_history(st.session_state.full_chat_history, get_chat_path(user, st.session_state.current_chat))

    st.markdown('<div>', unsafe_allow_html=True)
