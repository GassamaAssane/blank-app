import pandas as pd
import json
from datetime import datetime

# Fonction pour la recupération des tables et leurs descriptions

def extract_tableDescription (path: str) -> list:

    tables = pd.ExcelFile(path)
    sheet = tables.sheet_names[0]

    #for sheet_name in tables.sheet_names:
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


# "App/DATA_CATALOG_ECHANTILLON_REFINED_VUE360 5.xlsx"
def extract_col_Description (path: str) -> list:
# Charger toutes les feuilles du fichier Excel
    excel_path = path
    sheets = pd.read_excel(excel_path, sheet_name=None)  # Retourne un dictionnaire {nom_feuille: dataframe}

    # Parcourir les feuilles
    global_json = {}
    j = 0
    for sheet_name, df in sheets.items():
        #df.fillna(0)                   
        
        df = df.fillna("NULL")          
        j = j + 1
        #if j < 45 :
        global_json[sheet_name] = df.to_dict(orient="records")

    return global_json

def extract_KPI (path: str) -> list:

    df = pd.ExcelFile(path)
    sheet = df.sheet_names[0]

    #kpi = {}

    #for sheet_name in tables.sheet_names:
    df = df.parse(sheet)
    kpi = df.to_dict(orient="records")

    return kpi

#with open("toutes_les_tables.json", "w", encoding="utf-8") as f:
#    json.dump(global_json, f, indent=2, ensure_ascii=False)

#print(" Toutes les feuilles ont été enregistrées dans toutes_feuilles.json")

def convert_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # exemple : '2025-07-07T12:00:00'
    raise TypeError(f"Type non sérialisable : {type(obj)}")


global_json = extract_col_Description("DATA_CATALOG_ECHANTILLON_REFINED_VUE360 5.xlsx")

with open("All_colonnes_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(global_json, f, indent=2, ensure_ascii=False, default=convert_datetime)
    print("KO")

"""
kpi_infos = extract_KPI("Info_KPI.xlsx")

with open("KIP.json", "w", encoding="utf-8") as f:
    json.dump(kpi_infos, f, indent=2, ensure_ascii=False, default=convert_datetime)
    print("ok")
#info = extract_tableDescription("info_supp.xlsx")

"""

synthese = "Description Tables.xlsx"

catalog_table_desc = extract_tableDescription(synthese)

with open("All_tables_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(catalog_table_desc, f, indent=2, ensure_ascii=False)
    print("ok")


