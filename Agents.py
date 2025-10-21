import pandas as pd
import numpy as np
import random
import json
#from crewai import Agent, Task, LLM, Crew, Process
#from crewai_tools import JSONSearchTool, RagTool, TXTSearchTool, DallETool
#from crewai.memory import LongTermMemory
import streamlit as st
import datetime
import re
from typing import Dict, List, Any, Optional
from itertools import combinations
import random
from pathlib import Path
import base64
import os
from openai import OpenAI
from PIL import Image
import math
import os
import json
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
#from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

from langchain.docstore.document import Document
import os



openai_key = "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"
#openai_key = "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"
"""
gpt = LLM(
    #model= "gpt-4-turbo",
    #model= "gpt-3.5-turbo",
    model = "gpt-4.1",
    base_url= "https://api.openai.com/v1",
    api_key= openai_key,
    temperature= 0.1
)

gpt4 = LLM(
    #model= "gpt-4-turbo",
    #model= "gpt-3.5-turbo",
    model = "gpt-4o",
    base_url= "https://api.openai.com/v1",
    api_key= openai_key,
    temperature= 0.1
)

gpt_4 = LLM(
    model= "gpt-4o-mini",
    #model= "gpt-3.5-turbo",
    #model = "gpt-4o",
    base_url= "https://api.openai.com/v1",
    api_key= openai_key,
    temperature= 0.1
)

#llm_openai = ChatOpenAI(
#    model_name="gpt-4o",   # ou "gpt-3.5-turbo"
#    temperature=0.5,
#    api_key = openai_key
#)
"""
# Systeme d'Agent
new_api = "AIzaSyD9ZstqQTUvpjTPN1wLUP_k4eW3tSdJq9o"
api_k = "AIzaSyDRMK4upPL-nEIXd8Nurjgcy3IZyTYoGK0"
model_LLM = ['gemini/gemini-2.5-flash-preview-04-17', 'gemini/gemini-2.0-flash', 'gemini/gemini-2.5-flash']
apis_gem = ["AIzaSyCT-YT7kIIpUvxVhwYCqD3NkUjRQKPwolk", "AIzaSyC3TF0w1DL4wOGf50jNqnv_JKJMjsedP5M", 
            "AIzaSyDztFTQ8q3ydSqwaNu6y-EDptS-3gpfw30", "AIzaSyBxwZpwLsAl3YDWBfnsXd35djouNV3lX3E" ]
#gem = random.choice(model_LLM)
#api = random.choice(apis_gem)

"""
def system(table_desc, caracteristique, history, info_sup,
           request = "Quelles sont les clients qui ont utilisé les services de l'application maxit pour faire un transfert d'argent ou acheter un illiflex pendant le mois de janvier.") :
    last_turns = history[-4:]
    context = "\n".join(f"{m['role']}: {m['content_request']}" for m in last_turns)
#def system(table_desc, colonne_desc, caracteristique, 
#           request = "Quelles sont les clients qui ont utilisé les services de l'application maxit pour faire un transfert d'argent ou acheter un illiflex pendant le mois de janvier.") :
    memory_tab_concerne = LongTermMemory( path="memory_tab_concerne.txt")
    memory_request = LongTermMemory(path="memory_request.txt")

    gemini = LLM(
        #model='gemini/gemini-2.0-flash',
        #model='gemini/gemini-2.5-flash',
        model = 'gemini/gemini-2.0-flash-lite',
        #model='gemini/gemini-2.5-flash-preview-04-17',
        #model = gem,
        temperature = 0.1,
        #api_key = api,
        #max_tokens= 32000
        api_key= "AIzaSyDztFTQ8q3ydSqwaNu6y-EDptS-3gpfw30"
        #api_key= new_api
    )

    Gemini = LLM(
        model='gemini/gemini-2.5-flash',

        #model='gemini/gemini-1.5-flash',

        #model = 'gemini/gemini-2.0-flash-lite',
        #model='gemini/gemini-2.5-flash-preview-04-17',
        
        temperature = 0.1,
        #api_key = api,
        #max_tokens= 32000
        #api_key= "AIzaSyBxwZpwLsAl3YDWBfnsXd35djouNV3lX3E"
        api_key= new_api
    )

    groq_Llama = LLM(
        #model='gemini/gemini-1.5-flash',
        model = 'groq/llama-3.3-70b-versatile',
        temperature=0.1,
        #api_key= os.getenv('GEMINI_API_KEY')
        api_key= "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"
    )

    groq_llama = LLM(
        #model='gemini/gemini-1.5-flash',
        model = 'groq/llama-3.3-70b-versatile',
        temperature=0.1,
        api_key= "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"
        #api_key= "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"
    )

    geminii = LLM(
        model='gemini/gemini-2.5-flash',
        #model = 'groq/llama-3.1-8b-instant',
        temperature=0.1,
        #api_key= os.getenv('GEMINI_API_KEY')
        api_key= "AIzaSyAUBvZ4O0FjpOlJSOLACcPwTE9B0YS7NQg"
    )

    

    agent_analyst_db = Agent(
        role = "Expert Analyste base de données",
        goal = "Analyser les caractéristiques d'une base de donnée pour répondre à une question métier.",
        backstory = "Vous etes un expert en analyse de données avec plus de 25 ans d'expérience"
                    "et expert en gouvernance de données : tu documentes des bases  de données métier "
                    "pour que les équipes comme le BI comprennent chaque champ de la base pour tirer des "
                    "informations précises.",
        verbose= True,
        reasoning= True,
        llm= gemini,
        #llm = gpt4,
        #llm = groq_llama_I,
        max_rpm= 10,
        max_iter= 20,
        memory = memory_tab_concerne,

        #allow_delegation= True
        
        #response_template= ex_reponse
    )

    
    agent_sql = Agent(
        role = "Spécialiste en Production de requete SQL sous Hive",
        goal = "Mise en place d'une requete SQL en utilisant la sortie (la ou les table(s)) de l'agent 'Analyste base de données'.",
        backstory = "Vous etes un expert et professionnel en SQL avec plus de 30 ans d'expérience. Vous traduisez " \
                " les questions métiers en une requete SQL pour recupérer des informations sur la base donnée de votre entreprise.",
        verbose = True,
        reasoning = True,
        #llm= Gemini,
        llm= gemini,
        max_rpm = 10,
        max_iter= 20,
        #llm = gpt4,
        #llm = groq_llama,
        
        memory = memory_request
    )

    agent_infos_sup = Agent(
        role= "Informateur de données supplémentaire",
        goal= "Votre objectif est d'analyser les valeurs uniques de quelques colonnes de la base de donnée.",
        backstory= "Vous travaillez dans la gestion d'une base de données d'une entreprise de télécommunication depuis plus de 30 ans." \
                "Vous connaissez les valeurs uniques de chaque colonne des différents tables.",
        #llm= groq_Llama,
        llm= Gemini,
        reasoning= True,
        #llm = gpt,
        #llm = gpt4,
        #llm= geminii,
        max_rpm= 10,
        max_iter= 20,
        verbose=True
    )
    
    task_analyst_db = Task(
        name= "Analyse table",
        agent= agent_analyst_db,
        prompt_context = "Vous etes dans le contexte d'une entreprise de télécommunication (SONATEL ORANGE SENEGAL) qui veut consulter"
                         "sa base de donnée pour tirer des informations.",

        description= "Vous avez une base de données d'une entreprise de télécommunication avec les caractéristiques suivants : "
                     " - Les tables et leurs description : {table_desc} "
                     " - Tout les tables et leurs caractéristiques (colonnes, description, etc): {caracteristique}. " \
                     #"et " {colonne_desc}
                     #"(les caractéristiques des colonnes pour chaque table) {caracteristique}. " \
                     "Ses informations vous donnent le nom des tables et leurs descriptions, pour chaque tables, "
                     "vous avez le nom des colonnes et la descrition de chaque colonne, de meme qu'aussi pour chaque "
                     "colonne vous avez son type, quelques exemples valeurs de la colonne, le sens de la colonne par rapport au donnée. " \
                     
                    "Lorsque tu réponds à une question de l’utilisateur,ne propose que les tables et colonnes réellement présentes dans le catalogue.Si une table ne contient pas une colonne \
                    mentionnée dans la question, ne l’invente pas.Par exemple pour une question sur les régions, si une table contient commune et/ou département mais pas " \
                    "région, n'invente pas une colonne région dans cette table mais cherche une autre table qui peut etre liée pour répondre à la question. \
                    Si la réponse nécessite une colonne absente de la table interrogée, identifie d’autres tables qui contiennent cette information.Dans ce cas,explique la laison \
                    ou jointure entre ces tables.Retourne uniquement les tables et colonnes(tout les colonnes de chaque table) pour répondre à la question, \
                    sans ajouter d’informations fictives. Ne confond pas les colonnes régions, départements et communes. si vous avez une question qui parle de l'une d'eux, " \
                    "donnez les tables qui contiennent les informations des trois." \
                        
                     "Notez bien que les tables commençant comme nom 'daily' contiennent des données journalières des clients. " \
                     "celles commençant par 'monthly' répertorient les données mensuelles des clients et celles commençant par 'reporting' font une reporting des clients."
                     "Vous rencontrez 'sargal' dans la base sachez que c'est un programme de fidélité lancé par l'entreprise pour récompenser " \
                     "ses clients en fonction du niveau de consommation et engagement sur les services proposés. Autrement dit, les clients accumulent " \
                     "des points automatiquement à chaque rechargement de crédit. Plus ils consomment, plus ils cumulent de points. "
                     "Notez qu'en télécommunications, le terme 'parc' désigne l’ensemble des clients actifs ou abonnés à un service.Sachez que les questions " \
                     "sur le parc actif orange(ou parc actif ou parc orange), vous trouverez les réponses de ses questions sur les tables avec '_parc_' "
                     "Notez qu'aussi Maxit est une application mobile de l'entreprise qui offre plusieurs services (Achat, transfert d'argent, paiement, etc). " \
                     "A chaque fois que vous recevez une question qui parle de 'data', sachez que cela fait référence à internet. Autrement dit, dans les données, 'data' veut dire 'internet'."
                     "Avec toute ses informations données sur la base, votre mission est de rechercher la ou les tables de données qui peut ou peuvent fournir une réponse à cette "
                     "question : {requete}. Faite une analyse approfondie des caractéristiques ou documents de la base en étudiant le role de chaque tables et colonnes, "
                     "les relations qui peuvent exister entre les différents tables. Après, vous devrez répondre à cette question "
                     "métier suivante dans la base de donnée : {requete}. " \
                     "Sur les questions concernant les parc des takers, si maxit ou digital n'est pas renseigné, vous ne devez pas interroger les tables suffixées MAXIT."
                     "Derrière, vous pouvez créer des index pour les tables (les tables concernant les offre, souscriptions, sargal, chiffre " \
                     "d'affaire, recharges, data, etc). Cela vous facilitera la recherche de la réponse. " \
                     "Et aussi notez que le nombre de clients (ou abonnées ou bénéficiaires, etc) est différent au nombre de souscription. Un client peut souscrire dans une " \
                     "offre une ou plusieurs fois. L'identifiant des clients est donné par la colonne 'MSISDN'. Alors, le nombre de client doit toujours etre " \
                     "déterminé par une table contenant cette colonne. Notez qu'abonnés, bénéficaires, etc font référence aux clients."
                     "Vous allez prendre les résultats de votre analyse sur les caractérisques de la base pour determiner la ou les table(s) nécessaire(s) "
                      
                     "pour répondre à cette question métier. Vous allez étudier le but de la question métier qui a "
                     "été posée. C'est à dire quelles sont les informations que l'utilisateur veut savoir par rapport à cette question : " \
                     "{requete}. Si la réponse peut etre récupérer dans une seule table, en sortie vous donneriez seulement la table et tout les "
                     "colonnes de la table. Si la réponse est dans 2 ou plusieurs tables, Vous donnerez tout les tables concernées et leurs " \
                     "colonnes pour répondre à la question. Analysez bien la question pour voir, est ce que vous devez donner une seule table ou " \
                     "plusieurs tables (jointure entre tables). Notez que 'OM' signifie 'Orange Money'."
                     #nécessaires pour répondre à la question
                    #"Avant la sortie, vérifiez bien que les colonnes que vous avez mis dans chaque table "
                    #"correspond aux memes colonnes de la table ou des tables qui sont dans les caractéristiques "
                    #"de la base. C'est à dire que est-ce-que les colonnes qui sont dans la ou les table(s) XY que " \
                    #"vous donnez en sortie sont les memes qui sont dans la ou les table(s) XY sur les caractéristiques "
                    #"de la base donnée qui ont été donnés ci-dessus. La ou les tables (respectivement la " \
                    #"ou les colonnes) de sortie(s) doivent etre identique à celles qui ont été donnée sur les " \
                    #"caractéristiques. " \
                    "",
                     

        expected_output = "La ou les tables nécessaire(s) et la ou les colonnes pour répondre à cette question métier : "
                          "{requete}. Le résultat sera sous la forme : [{ nom_table : TableA, colonnes : { {colonne1 : description de la colonne1}, " \
                          "{colonne2 : description de la colonne2}, ... , {colonneN : description de la colonneN} } , "
                          ", { nom_table : TableE, colonnes : { {colonne1 : description de la colonne1}, {colonne2 : description de la colonne2}, " \
                          "... , {colonneM : description de la colonneM} }} ] avec quelques explications si nécessaires." \
                          "La sortie du ou des documents doit ou doivent etre en document(s) JSON correct. Ne donne jamais en retour un nom de table " \
                          "qui n'existe pas dans le catalogue de données fournies." \
                          ,
        output_file= "table_col.txt"
    )

    task_sup = Task(
        name= "Infos supplémentaire",
        agent= agent_infos_sup,
        description = "Vous étudiez les colonnes des différents tables fourni par la tache 'Analyse table'.Vous allez regarder est ce qu'il " \
                      "y a une colonne ou des colonnes qui sont dans les informations supplémentaires (colonne : valeurs uniques ou présence " \
                      "du nom partiel de colonne : valeurs uniques): {info_sup}.Ces derniers contiennent les informations d'une colonne ou d'un ensemble " \
                      "de colonne.Autrement dit, vous avez des noms de colonnes avec leurs valeurs uniques (exemple : SEGMENT_RECHARGE : S0, Mass-Market, " \
                      "Middle, High, Super High) ou des noms (groupe de mots) qui représentent plusieurs colonnes avec leurs valeurs uniques (exemple :  " \
                      "Pour les colonnes réprésentant les regions : TAMBACOUNDA, ZIGUINCHOR, DIOURBEL, DAKAR, THIES,SAINT LOUIS, etc)" \
                      "Si après votre étude, vous trouvez une ou des colonnes sur les infos supplémentaire,alors vous allez étudiez la " \
                      "question pour voir la ou les valeurs unique(s) de la ou les colonnes qui permet(tent) de répondre à la "
                      "question : {requete}, puis vous allez récupérerer la ou les valeurs et les passez à la tache 'Tache SQL' qui l'utilisera " \
                      "sur sa production de requete SQL au niveau de la clause WHERE.Ce qui permettra à la 'Tache SQL' de produire une requete " \
                      "avec un clause WHERE qui s'adapte avec les informations de la base de données.Par exemple, pour une question comme " \
                      "'Quelle est le montant total des paiements SENELEC', pour " \
                      "répondre à cette question, vous étudiez la colonne qui permet de répondre à cette question, et sur ses éléments, " \
                      "quelle est ou sont l'élément(s) unique(s) de la colonne qui permet de répondre à la question.Dans cet exemple, ça correspond à la " \
                      "la colonne 'service' avec sa valeur unique 'BY_SENELEC' qui sera utilisée sur la requete qui sera produite par " \
                      "la 'Tache SQL' sur sa clause WHERE.Un autre exemple,'donne moi le parc des illimix jour de février 2025',dans cet exemple vous allez " \
                      "recupérer l'élément unique correspondant au illimix jour dans le nom des offres et le donner en sortie (ici c'est 'illimix jour 500F'). Il en est de meme " \
                      "pour tout autres questions de ce genres.Par contre, si vous avez une question où vous avez besoins de tout les éléments de la " \
                      "colonnes, dans ce cas, ce n'est pas la peine de donner en sortie des informations supplémentaires.Par exemple,'Donne moi " \
                      "le chiffre d'affaires des offres en 2025',sur cet exemple, vous avez besoin de toute la colonne qui représente les offres, alors " \
                      "ce n'est pas la peine d'envoyer en sortie les valeurs uniques des offres.Un autre exemple, 'Donne moi le revenu HT des segments " \
                      "marchés',pour répondre à cette question, vous avez besoin de tout les segments marché, alors ce n'est pas la " \
                      "peine de retourner les valeurs uniques de la colonne qui représente les segments marchés.L'agent 'Producteur de requete SQL' " \
                      "se chargera de la gestion de ses valeurs uniques.Il en est de meme pour toute ses genres de questions.En résumé,vous allez retourner " \
                      "uniquement des valeurs lorsque la question qui a été posée spécifie une ou quelques valeur(s) unique(s) d'un ou des colonne(s). Notez que les informations" \
                      "supplémentaires contiennent quelques variables catégorielles et leur valeurs uniques.Votre mission est de recupérer " \
                      "la ou les valeur(s) unique(s) nécessaire(s) de la ou les colonne(s) pour répondre à la question qui a été posée.Notez que 'OM' signifie 'Orange Money'." \
                      "Votre mission est de recupérer la ou les valeur(s) unique(s) nécessaire(s) de la ou les colonne(s) pour répondre à la question qui a été posée." \
                      "Donner en sortie une ou des colonnes (s'il y en a dans les infos supp) avec quelque(s) de sa ou ses valeurs uniques qui permet(tent) \
                      de répondre à la question : {requete}. Par exemple, [ service : 'By_SENELEC', segment_marche : ['JEUNES', 'ENTRANT'], etc] avec quelques " \
                      "explications de la ou les colonne(s) avec sa ou ses valeurs unique(s) sur son utilisation sur un clause WHERE."
                      "Si la question ne nécessite pas d'info supplémentaire, donnez retour qu'elle n'a pas besoin d'infos sup.",
        expected_output= "Une ou des colonnes avec quelque(s) de sa ou ses valeurs uniques qui permet(tent) de répondre à la question : {requete}." \
                        "Par exemple, [ service : 'By_SENELEC', segment_marche : ['JEUNES', 'ENTRANT'], " \
                        "ca_cr_commune_90 : 'DAKAR',offer_name = ['PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO'], etc] avec une petite explication " \
                        "sur la ou les colonne(s) et leur(s) valeur(s) unique(s) ou bien pas besoins d'informations " \
                        "supplémentaires s'il n'en a pas.",

        output_file= "info_sup.txt",
        context= [task_analyst_db]
    )


    task_resquest_db = Task(
        name = "Tache SQL",
        agent = agent_sql,
        #tools= [sql_tools],

        description = "Vous allez recupérer uniquement la ou les table(s) de données fournie(s) par la tache 'Analyse table' en sortie et tout leur(s) "
                    # et les informations supplémentaires données en sortie par la tache 'Infos supplémentaire'(s'il y en a) .Ses infos supplémentaires 
                    # sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut 
                    # etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous
                    # permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une valide et correcte par rapport à nos données.
                      "colonne(s) et les informations supplémentaires données en sortie par la tache 'Infos supplémentaire'(s'il y en a) " \
                      "pour produire une requete SQL valide qui permet de répondre à cette question métier : {requete}.Ses infos supplémentaires" \
                      "sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut" \
                      "etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous" \
                      "permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une requete SQL " \
                      "valide et correcte par rapport à nos données.Sur lA clause WHERE de la requete, utilisez toujours la ou les valeur(s) par les infos supplémentaire." \
                      "Notez qu'à chaque fois que vous recevez une question qui parle de 'data',sachez que cela fait référence à internet. Autrement " \
                      "dit, dans les données,'data' veut dire 'internet' ou 'pass internet' (Par exemple : offre data signifie offre pass internet)." \
                      "Abonnés, bénéficaires, etc font référence aux clients.'OM' signifie 'Orange Money'."
                      #"Et aussi le nombre de clients (ou abonnées,etc) est différent au nombre de souscription"
                      " Analyse bien la ou les tables reçue(s) pour voir est ce que l'information de la question posée est dans une ou plusieurs " \
                      "colonnes. Regarde bien aussi est-ce-que vous avez reçu une ou plusieurs table(s)." \
                      "Parfois, vous pouvez recevoir plusieurs tables, et que chaque table peut répondre à la question posée. Dans ce cas, prenez " \
                      "la table qui donne le plus d'information pour répondre à la question(Privilégiez toujours les tables commençant par 'reporting', " \
                      "ou 'monthly' à moins que la question se refére sur les informations quotidiennes ('daily'))."
                      "Si vous avez reçu une seule table, produit une requete SQL correcte"
                      " avec cette table seulement en prenant que les colonnes qui contiennent " \
                      "l'information de la réponse sur la table. S'il y a 2 "
                      "ou plusieurs tables reçues et que la réponse se trouve sur les différents tables, analyse les colonnes de chaque table, puis étudier les dépendances "
                      "entre les tables, c'est à dire les colonnes qui permet de faire la liaison entre les tables."
                      "Après cela, produit une requete SQL correcte avec les colonnes qui contiennent l'information sur les différentes " \
                      "tables reçues pour répondre à cette question métier : "
                      "{requete}. Si la question est trop vaste, (par exemple 'Je veux le chiffre d'affaire', 'Quelle formule tarifaire " \
                      "génère le plus de trafic 4G sur le mois d'avril', ect) vous essayerez toujours de répondre en donnant une " \
                      "requete SQL qui donne les informations les plus récentes. Dans cet exemple, vous donneriez le chiffre d'affaire ou " \
                      "formule tarifaire  de l'année en cours (2025), Si la question " \
                      "est posée en 2025, vous donnez une requete qui donne les information de 2025, de meme que si c'est en 2026, etc. " \
                      "Si vous avez une requete qui nécessite une condition et si la condition "
                      "doit se faire avec des caractères ou chaines de caractères (Ex : Région de Dakar), sur "
                      "la clause WHERE, Utilise LIKE plutot égal (=) par exemple '%Dakar%', '%DAKAR' ou '%kar%', etc ." \
                      "Exemple de question : Quelle commune a généré le plus de CA data pour JAMONO NEW S’COOL? Sur la clause WHERE vous pouvez mettre " \
                      "par exemple variable LIKE '%JAMONO' ou variable LIKE '%NEW S’C%', etc. Appliquez ses exemples dans ses genres cas." \
                      "En résumé n'utilise jamais égal dans une clause WHERE avec caractère ou chaine de caractères, utilise toujours LIKE avec une partie du groupe de mot."
                      "N'utilisez jamais tout les mots donnez sur la question sur la condition de la requete (par exemple : variable LIKE '%JAMONO NEW S’COOL%' comme dans l'exemple précédent)"
                      "Sur les questions concernant les parc des takers, si 'maxit' ou 'digital' n'est pas renseigné dans la question, vous ne devez pas interroger les tables suffixées 'maxit'."
                      "Attention ne fait jamais une requete pour supprimer, "
                      "pour modifier ou pour mettre à jour ou pour insérer dans la base. votre but est de sélectionner, alors "
                      "mettez seulement des requetes SQL qui permet de faire la sélection. Sélectionnez toujours les colonnes 'year' et 'month' " \
                      "sur la requete et utilisez les memes nom de colonne pour les alias (exemple : as year et as month). Sachez que le mois est toujours sous format numérique." \
                      "Si l'année n'est pas spécifiée ou renseignée sur la question, filtrez toujours sur l'année en cours ou le max des années pour ne pas retourner les données de tout les " \
                      "années.Autrement dit, fait toujours un filtre de l'année(2025 toujours si l'année n'est pas renseigné sur la question) sur la requete et utilise toujours des 'Alias' par exemple 'nom_table.nom_colonne' " \
                      "pour éviter d'avoir des erreurs d'exécution provoqué par un nom de colonne.A chaque fois que vous recevez une question qui parle de "
                      "'data',sachez que cela fait référence à internet. Autrement dit, dans les données, 'data' veut dire 'internet' ou 'pass internet'."
                     "Notez que 'Airtime' signifie recharge à partir de ton crédit."
                      "Par rapport au question de segmentation, analyse bien la question pour donner en retour une réponse claire qui permet de définir bien les différents segments(ou clusters) demandés." \
                      "Sur la requete qui sera produite, ne met jamais une limite à moins que la requete vous " \
                      "l'oblige à le faire. Par exemple, vous pouvez avoir comme requete : Donnez le top 10 des chiffres d'affaires des régions ou Quelle commune "
                      "a la balance moyenne SARGAL la plus élevée.Dans ses genres de question,vous pouvez utilisez la clause LIMIT dans la requete." \
                      "La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une requete SQL valide qui peut etre exécutée " \
                      "sur n'importe quelle version de HiveSQL sans erreur. Pour les questions sur la corrélation, n'utilise jamais la fonction " \
                      "d'aggrégation 'CORR()' car cela n'a pas marché. l'erreur dit que cette fonction n'est pas supportée, essaie plutot de " \
                      "le calculer en appliquant la formule de la corrélation.Autrement dit, n'utilise jamais une fonction obsolète dans la requete." \
                      "Ne met jamais de requete avec une sous requete sur la clause WHERE. Par exemple,  WHERE year = (SELECT ....), Utilise plutot " \
                      
                      #"sélectionner tout les colonnes reçues, si sélectionner tout sinon prenez deux colonnes (ou "
                      #"plus) qui ont plus de sens par rapport à la question {requete}. "
                      "JOIN ON à place.Sur la requete SQL, ordonnez toujours le résultat du plus récents au plus anciens ou du plus grand au plus petit."
                      "Analysez la question et les données fournies pour voir, est ce que vous devez de faire une requete simple,d'aggrégation,de jointure ou combiné" \
                      #"simple,d'aggrégation,de jointure ou combiné."
                      f"Voici une petite historique des derniers messages \n:{context}\n",
        
        context = [task_analyst_db, task_sup],
        expected_output = "Une requete SQL uniquement qui est claire, structurée et correcte qui répond à la question:{requete}." \
                          "Ne prend pas de colonne unutile et aussi n'ajoute jamais des commentaires sur la requete, prenez seulement " \
                          "tout les colonnes qui contiennent l'information de la question et sélectionnez tout les colonnes qui ont " \
                          "été utilisées sur la requete.Utilisez toujours des alias dans la requete et ajouter les colonnes year " \
                          "et month dans la sélection.La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une " \
                          "requete SQL valide sous Hive. Voici quelques genres d'exemples de requete SQL pour la sortie : "
                          "SELECT c.year as year,c.month as month,c.id, c.nom, c.ville, c.date_vente FROM clients as c WHERE c.ville LIKE 'Dakar';"
                          "SELECT v.year as year,v.month as month,v.id, v.nom, v.ville, MONTH(v.date_vente), SUM(v.montant) FROM "
                          "ventes v GROUP BY MONTH(v.date_vente) ORDER BY MONTH(v.date_vente);"
                          "SELECT clients.year as year, clients.month as month, clients.id_client, clients.nom, clients.ville, " \
                          "ventes.date_vente ventes.montant FROM clients JOIN ventes ON clients.id_client = ventes.id_client; " \
                          "SELECT rcdm.year AS year, rcdm.month AS month,(AVG(rcdm.volumes * rcdm.ca_data) - AVG(rcdm.volumes) * AVG(rcdm.ca_data)) / "
                          "(STDDEV_POP(rcdm.volumes) * STDDEV_POP(rcdm.ca_data)) AS correlation_volume_revenu FROM reporting_ca_data_monthly AS rcdm JOIN " \
                          "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS max_year_subquery ON rcdm.year=max_year_subquery.max_year " \
                          "GROUP BY rcdm.year, rcdm.month ORDER BY rcdm.year DESC, rcdm.month DESC;" \
                          "SELECT T1.year AS year,T1.month AS month,SUM(T1.ca_data) AS revenu FROM reporting_ca_data_monthly AS T1 JOIN "
                          "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS T2 ON T1.year = T2.max_year GROUP BY " \
                          "T1.year, T1.month ORDER BY T1.year DESC, T1.month DESC;" ,
        output_file = "result.txt"
    )

    systeme = Crew(
        agents= [agent_analyst_db, agent_infos_sup,agent_sql],
        tasks= [task_analyst_db,task_sup, task_resquest_db],
        #memory= True,
        process=Process.sequential,
        #verbose= True
    )

    input = {
        "table_desc" : table_desc,
        #"colonne_desc" : colonne_desc,
        "requete" : request ,
        "caracteristique" : caracteristique,
        "info_sup" : info_sup
    }

    try :
        st.write("Veuillez patientez, votre requete est en cours d'exécution ...")
        system_multi_agent = systeme.kickoff(inputs=input)
        print(system_multi_agent.token_usage)
        #sql_query = extract_sql("result.txt")
        sql_query = Extract_sql("result.txt")

    except Exception as e :
        st.error(f"Le modèle ne répond pas pour le moment, veuillez réessayez plus tard : {e}.")
        sql_query = None

    #with open("result.txt", "r", encoding="utf-8") as f:
    #    content = f.read()

    # Extraction de la requête
    #match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
    #sql_query = match.group(1) if match else None
    #st.write(f"votre requete SQL \n: {sql_query}")

    #sql_query = extract_sql("result.txt")

    #print(system_multi_agent.token_usage)

    return sql_query
    """


def extract_sql(path_txt: str) -> str:
    """
    Extrait la requête SQL contenue dans un fichier texte.
    - Si un bloc ```sql … ``` est présent, renvoie son contenu.
    - Sinon, renvoie tout le contenu du fichier (stripé).
    """
    txt = Path(path_txt).read_text(encoding="utf-8")

    # Regex : bloc démarrant par ```sql (ou ```SQL) et finissant par ```
    #pattern = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

    # ‑‑ Regex :   SELECT … ;   (DOTALL = le point capture les retours à la ligne)
    pattern = re.compile(r"(?is)SELECT.+?;", re.IGNORECASE | re.DOTALL)   # (?is) => ignore‑case + DOTALL

    match = pattern.search(txt)
    #print(match.group(0).strip())
    return match.group(0).strip() if match else txt.strip()

def Extract_sql(path_txt: str) -> str:
    """
    Extrait la requête SQL contenue dans un fichier texte.
    - Si un bloc ```sql … ``` est présent, renvoie son contenu.
    - Sinon, renvoie tout le contenu du fichier (stripé).
    """
    txt = Path(path_txt).read_text(encoding="utf-8")

    # Regex : bloc démarrant par ```sql (ou ```SQL) et finissant par ```
    #pattern = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    pattern = re.compile(r"```(?:sql|hivesql|hql)\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

    match = pattern.search(txt)
    #print(match.group(0).strip())
    return match.group(1).strip() if match else txt.strip()


def Agent_analyse_recommand(openai_key = openai_key, request = None, nb_img = 1, nb_imgKPI = 1):
    
    agent = OpenAI(api_key=openai_key)

    # Dossier contenant toutes les images
    #images_dir = "Image"  

    # toutes les images du dossier
    #image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_paths = [f"Image/{i}.png" for i in range(nb_img)]
    dash = combine_images_grid(image_paths)

    image_kpi = [f"img_kpi/{i}.png" for i in range(nb_imgKPI)]

    kpi = combine_images_grid(image_kpi, output_path= "img_kpi/dash.png")

    images = []
    with open(dash, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
        images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })

    with open(kpi, "rb") as f:
        image_base64_kpi = base64.b64encode(f.read()).decode("utf-8")
        images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64_kpi}"}
            })

    # Appel à GPT-4o pour analyse visuelle
    response = agent.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Voici un visuel : {dash}.Ce visuel fait partie des graphiques conçus par des données de la \
                            réponse de cette question : {request}.Fais une analyse visuelle globale pour expliquer les tendances(en 4 points au \
                            max) et donner des recommandations (ex : activer une campagne,réviser une tarification,cibler un segment,etc) \
                            5 points au max.Tenez en compte des relations entre les graphes.Faite comme si que vous etes un analyste data senior avec une forte expertise business et \
                            marketing à la SONATEL(Société Nationale de Télécommunication du Sénégal)."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        #max_tokens=200,
        max_completion_tokens=600
    )

    # Affiche les résultats
    
    #print(response.choices[0].message.content)

    #x = response.choices[0].message.content

    #responses = f"{responses} \n\n{x}"
    responses = response.choices[0].message.content

    # calcule du cout pour gpt-4o
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # Coût en dollars
    cost = (prompt_tokens * 0.0025 / 1000) + (completion_tokens * 0.01 / 1000)
    print(cost)

    return responses, cost, usage
 

def combine_images_grid(image_paths, output_path="Image/dash.png", images_per_row=3, padding=10, background_color=(255, 255, 255)):
    images = [Image.open(p) for p in image_paths]
    
    # Redimensionner les images à la même taille (optionnel mais recommandé)
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)
    resized_images = [img.resize((max_width, max_height)) for img in images]

    total_images = len(resized_images)
    rows = math.ceil(total_images / images_per_row)
    
    grid_width = images_per_row * max_width + (images_per_row - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding

    new_im = Image.new("RGB", (grid_width, grid_height), color=background_color)

    for index, img in enumerate(resized_images):
        row = index // images_per_row
        col = index % images_per_row
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        new_im.paste(img, (x, y))

    new_im.save(output_path)
    return output_path


##################################################################

#           Partie RAG         #

#OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key 

def dict_to_text(data_item):
    """
    Convertit un dict ou une liste de dicts en texte lisible
    """
    if isinstance(data_item, list):
        # Si c'est une liste, on convertit chaque dict
        return "\n\n".join([dict_to_text(d) for d in data_item])
    elif isinstance(data_item, dict):
        
        # Si c'est un dict, on convertit chaque clé/valeur
        return "\n".join([f"{k}: {v}" for k, v in data_item.items()])
    else:
        # Sinon on force en string
        return str(data_item)

def Extract_tables(path_txt: str) -> str:
    """
    Extrait la requête SQL contenue dans un fichier texte.
    - Si un bloc ```json … ``` est présent, renvoie son contenu.
    - Sinon, renvoie tout le contenu du fichier (stripé).
    """
    txt = Path(path_txt).read_text(encoding="utf-8")

    # Regex : bloc démarrant par ```sql (ou ```SQL) et finissant par ```
    pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    #pattern = re.compile(r"```(?:json)\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

    match = pattern.search(txt)
    #print(match.group(0).strip())
    return match.group(1).strip() if match else txt.strip()

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


catalog_table_desc = ""
with open("All_tables_descriptions.json", "r", encoding="utf-8") as f:
    catalog_table_desc = json.load(f)

caracteristique = ""
    # Chargement des caractéristiques des tables 
with open("All_colonnes_descriptions.json", "r", encoding="utf-8") as f:
    caracteristique = json.load(f)

with open("KIP.json", "r", encoding="utf-8") as f:
    data_kpi = json.load(f)

data_supp = extract_tableDescription("info_supp.xlsx")

tables = catalog_table_desc[0]
documents = [
    Document(
        page_content= f"Table : {key}\nDescription : {value}\nCaractéristiques : {caracteristique[key]} \n\n"  
    )
    for key, value in tables.items()
        ]

print("ko")

#Découper le texte en chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40)
#text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=80)
#text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)

#text_splitter = RecursiveCharacterTextSplitter(
#                    chunk_size=2000,   # chaque chunk fait 1000 caractères max
#                    chunk_overlap=200  # un petit chevauchement pour le contexte
#                                                )

docs = text_splitter.split_documents(documents)
print("KO")

def Agent_analyst_RAG(requete) :

    #Créer les embeddings et la base vectorielle
    # model="text-embedding-3-small"
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-large")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)

    #Créer la chaîne RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        # gpt-4o-min
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        #llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    
    # Tester une question
    query = "Vous un Expert Analyste base de données et recherches documentaire.Votre but est de rechercher sur les caractéristiques d'une  \
        base de donnée pour répondre à une question métier.En résumé,vous etes un expert en analyse de données avec plus de 25 ans d'expérience \
        et expert en gouvernance de données : tu documentes des bases  de données métier pour que les équipes comme le BI comprennent chaque \
        champ de la base pour tirer des informations précises.Maintenant Quelle(s) est ou sont la ou les table(s) qui permet(tent) de répondre à la question  \
        suivante sur le RAG:" f"{requete}.""Donne moi en sortie le(s) nom(s) de la ou des tables et tout ses caractéristiques (tout les \
        colonnes avec leurs noms exactes mentionnés sur le RAG, types, descriptions, etc) et aussi la description de la ou des table(s). \
        Les informations du RAG vous donnent le nom des tables, leurs descriptions et ses caractéristiques,pour chaque tables, vous avez le nom des colonnes et la descrition de chaque \
        colonne, de meme qu'aussi pour chaque colonne vous avez son type, quelques exemples de valeurs de la colonne, le sens ou le contexte ou l'utilité de la colonne par rapport au donnée. \
        \
        Autrement dit, Tu as accès à un RAG qui contient la documentation d’une base de données : le nom des tables, leur description, la liste des colonnes avec leurs types et leur signification. \
        Lorsque tu réponds à une question de l’utilisateur,ne propose que les tables et colonnes réellement présentes dans le catalogue.Si une table ne contient pas une colonne \
        mentionnée dans la question, ne l’invente pas.Par exemple pour une question sur les régions, si une table contient commune et département mais pas région, n'invente pas une colonne région dans cette table. \
        Si la réponse nécessite une colonne absente de la table interrogée, identifie d’autres tables qui contiennent cette information.Dans ce cas,explique qu’il faut \
        faire une jointure ou une liaison entre ces tables.Retourne uniquement les tables et colonnes (tout les colonnes de chaque table) avec leurs noms réelles ou exactes qui ont été définit dans le catalogue pour répondre à la question, \
        sans ajouter d’informations fictives. \
        \
        Notez bien que les tables commençant comme nom 'daily' contiennent des données journalières des clients celles commençant par 'monthly' \
        répertorient les données mensuelles des clients et celles commençant par 'reporting' font une reporting des clients. \
        Si vous rencontrez 'sargal' dans la base sachez que c'est un programme de fidélité lancé par l'entreprise pour récompenser  \
        ses clients en fonction du niveau de consommation et d'engagement sur les services proposés. Autrement dit, les clients accumulent \
        des points automatiquement à chaque rechargement de crédit. Plus ils consomment, plus ils cumulent de points.\
        Notez qu'en télécommunications, le terme 'parc' désigne l’ensemble des clients actifs ou abonnés à un service.Sachez que les questions \
        sur le parc actif orange(ou parc actif ou parc orange), vous trouverez les réponses de ses questions sur les tables 'daily' plus \
        précisément 'daily_parc'.Autrement dit, pour les questions sur parc actif orange (sans la précision des clients digitaux), consultez la table 'daily_parc'. \
        Notez qu'aussi Maxit est une application mobile de l'entreprise qui offre plusieurs services (Achat, transfert d'argent, paiement, etc). \
        \
        Avec toute ses informations données sur la base, votre mission est de rechercher la ou les tables de données qui peut ou peuvent fournir une réponse à cette \
        question :" f"{requete}"". Faite une analyse approfondie des caractéristiques ou documents de la base en étudiant le role de chaque tables et colonnes, \
        les relations qui peuvent exister entre les différents tables. Après, vous devrez répondre à cette question métier suivante dans la base de donnée :" f"{requete}"". \
        Sur les questions concernant les parc des takers, si maxit ou digital n'est pas renseigné, vous ne devez pas interroger les tables suffixées MAXIT.\
        Derrière, vous pouvez créer des index pour les tables (les tables concernant les offre, souscriptions, sargal, chiffre \
        d'affaire, recharges, data, etc). Cela vous facilitera la recherche de la réponse. \
        Et aussi notez que le nombre de clients (ou abonnées ou bénéficiaires, etc) est différent au nombre de souscription. Un client peut souscrire dans une \
        offre une ou plusieurs fois. L'identifiant des clients est donné par la colonne 'MSISDN'. Alors, le nombre de client doit toujours etre  \
        déterminé par une table contenant cette colonne 'MSISDN'. Notez qu'abonnés, bénéficaires, etc font référence aux clients. \
        Vous allez prendre les résultats de votre analyse sur les caractérisques de la base pour determiner la ou les table(s) nécessaire(s) \
        pour répondre à cette question métier. Vous allez étudier le but de la question métier qui a \
        été posée. C'est à dire quelles sont les informations que l'utilisateur veut savoir par rapport à cette question : \
        "f"{requete}"". Si la réponse peut etre récupérer dans une seule table, en sortie vous donneriez seulement la table et tout les \
        colonnes de la table. Si la réponse est dans 2 ou plusieurs tables, Vous donnerez tout les tables concernées et leurs \
        colonnes pour répondre à la question. Analysez la question pour voir, est ce que vous devez de faire une requete \
        simple , d'aggrégation, de jointure ou combiné. Notez que 'OM' signifie 'Orange Money'. Ne donne jamais un " \
        "nom de table qui n'existe pas dans le catalogue de données fournies dans le RAG et aussi ne jamais donner un nom de " \
        "colonne qui n'existe pas dans une table des données de la catalogues. " \
        "Notez que la jointure entre les tables peut etre fait entre des colonnes qui ne représente pas l'identifiant " \
        "des clients. Faites bien la différence entre les tables parc, de l'application Maxit, de souscriptions, du trafic réseau ou de " \
        "consommation de donnée (data) mobile, du programme de fidélité Sargal, de voix, recharges ou chiffre d'affaires (ca), pour les trafic à " \
        "l'international (ca, durée, parc, etc à l'international), etc. Analyse bien la question posée avant de " \
        "donner la ou les tables pour la sortie. Si vous voyez une question qui parle de paiement (par exemple paiement senelec, sen'eau, " \
        "etc) consultez la table 'daily_maxit' qui contient les infos quotidiennes sur les services de l'application maxit avec sa colonne 'service'." \
        \
        "Les tables commençant par 'daily' et 'monthly' contiennent chacune une colonne 'MSISDN' " \
        "qui représente l'identifiant des clients ou abonnés. Sachez que les tables commençant par 'reporting' n'ont pas une colonne " \
        "pour l'identifiant des clients. Pour les questions sur Maxit, consultez les tables Maxit, pour les questions sur Sargal " \
        "allez sur les tables Sargal, pour les questions sur la data, consultez les tables data, pour les questions sur l'international, " \
        "allez sur les tables internationales, etc. Notez que les tables de données avec comme suffixe 'international' " \
        "(monthly_international, reporting_monthly_international) " \
        "sur le nom de la table contiennent des infos sur les trafics à l'international (le parc à l'international (parc_pass, parc_international, parc_payg), le chiffre d'affaires à " \
        "l'international (ca_pass, ca_payg), la durée des communications à l'international (ci_pass_dur, duration_mn_payg), etc)" \
        \
        "NB : Notez que la table daily_oss_5g contiennent les informations sur les cellules et leurs trafics (nom cellule, trafic moyen des utilisateurs, etc). daily_clients et " \
        "monthly_clients contiennent tout les informations quotidiennes et mensuelles des clients (identifiant, infos géographiques et démographiques, segment d'appartenance,etc.). daily_clients_digitaux donne les infos des clients qui utilisent les " \
        "plateformes digitaux de la SONATEL.daily_conso fournit le montant des consommations des appels (ou voix) internationaux de chaque client " \
        "(ca_voix_international).monthly_international et reporting_monthly_international donnent le montant des consommations des communications internationaux pour les forfaits, pour les usages hors forfait et " \
        "leur durée totale. daily_data donne le volume de trafic de la consommation des données mobiles sur le réseau 2G, 3G, 4G et 5G en " \
        "MégaOctets(Mo) tandisque monthly_data ajoute la commune et les segments (marché et recharges) d'appartenance et la formule tarifaire " \
        "souscrite par le client.daily_delta_parc et reporting_parc_monthly donnent les infos sur la sortie et le parc des clients (sortant(0/1), reactive(0/1),parc(0/1),nouvelles " \
        "arrivées(0/1), etc) alors que monthly_sortant ajoute la localité de référence du client (région, département, commune, zone_drv, " \
        "nom_cellule, etc) tandisque daily_parc fournit les infos globlal du parc des clients (nombre d'entrant(entrant_cnt), de sortant (sortant_cnt) et de régularité (regularite), le parc (0/1), etc)." \
        "daily_habit_5g_user donne la consommation Data mobile des Clients (volume_go) sur le réseau 5G par type d'application.daily_heavy_user_4g " \
        "donne le suivi de la consommation 4G des Heavy user. daily_infos_bts donne les informations détaillées sur les sites physiques et leurs " \
        "cellules radios correspondant (NOM_SITE, NOM_CELLULE, CELLID, ID, TYPE_CELLULE, la REGION, le DEPARTEMENT, etc). daily_infos_otarie et monthly_terminaux donnent " \
        "le parc des clients avec leurs appareils mobiles et leurs marques utilisés, le parc d'utilisation du reseau (5G, 4G, 3G, 2G) et les " \
        "volumes de données consommées pour chaque réseau en MégaOctets (Mo). daily_localisation_5g fournit les " \
        "infos géographiques des clients. daily_maxit et monthly_maxit donnent les clients qui se sont connectés sur l'application mobile " \
        "MAXIT tandisque reporting_daily_maxit, reporting_monthly_maxit, reporting_daily_parc_maxit et reporting_usage_maxit font le reporting des services de l'application mobile Maxit." \
        "daily_parc_recharges, daily_parc_pass, daily_parc_maxit, daily_parc_maxit_new, daily_parc_illimix, daily_parc_illiflex, " \
        "daily_parc_data_4g, daily_parc_data et daily_parc_sargal fournissent les informations sur les clients ou le parc client sur la dernière date d'activation " \
        "(last_active_date) ainsi que la fréquence (nb_j_distinct) pour les recharges, les pass, d'utilisation maxit, d'achat illimix et illiflex et data " \
        " du trafic le réseau 4G, data trafiqué le réseau global et partciper au programme Sargal. daily_recharges, monthly_recharges et reporting_ca_monthly donne les " \
        "chiffres d'affaires total (ca_recharge) ou générés pour tout les recharges (ca_recharges) sur les différents canaux ainsi que par canal (par OM " \
        "(ca_pass_glob_om_jour, ca_credit_om), par Wave (ca_wave), par Seddo(ca_seddo), par cartes (ca_cartes), à l'international (ca_iah), " \
        "et par self_top_up).monthly_sargal, daily_sargal, reporting_sargal_echangeurs_mon, reporting_sargal_gift_daily, " \
        "reporting_sargal_gift_monthly et reporting_sargal_inscrits donne les infos sur la participation du programma de fidélité Sargal. " \
        "monthly_souscription et daily_souscription fournissent des informations ou le CA des souscriptions (ca_data et ca_voix) des catégories (Pass Internet, illimix, illiflex, bundles, Mixel, International, etc) et " \
        "types d'offres souscrite (ILLIMIX JOUR 500F, PASS 2.5GO,MIXEL 690F, illiflex mois, etc). reporting_ca_data_monthly fait le reporting du Chiffre d'affaire data (ca_data) ou internet mensuelle. reporting_daily_offer, " \
        "reporting_offer_monthly et reporting_souscription_monthly font le le reporting sur les souscriptions des offres (sous_mnt,ca_data, ca_voix, ca_sous_HT) suivant le " \
        "types de souscription et le type d'offre souscrit. reporting_monthly_terminaux fait le reporting des Data_user ou No Data_user avec la colonne data_status " \
        "et leur utilisation de smartphone ou pas. daily_sva, monthly_sva et reporting_monthly_sva donne le montant de la souscription et le parc des " \
        "services à valeur ajouté.daily_voix, monthly_voix et reporting_monthly_voix fournissent le volume ou la durée des appels sortants des abonnées " \
        "sur les différents opérateurs téléphoniques (Orange, Free, Expresso, ProMobile, etc) ainsi leurs parcs clients correspondant. reporting_5g_daily regroupe les sites techniques " \
        "avec une série de KPI quotidiens (Key Performance Indicators) liés à l’usage du réseau 5G. reporting_daily_parc fait le reporting quotidienne " \
        "du comportement (actifs ou inactifs) ou parc des abonnées sur les 90 derniers jours. reporting_daily_ca_pag_xarre, " \
        "reporting_daily_data_xarre, reporting_daily_trafic_xarre, reporting_recharge_daily_xarre et reporting_xarre_offer_nrt " \
        "donnent le chiffres d'affaires généré des voix et SMS en mode PAYG (Pay-As-You-Go) pour les formules ou offres commerciales XARRE, " \
        "le trafic Internet et le parc d’abonnés par technologie réseau (2G, 3G, 4G, 5G) ventilées par formule commerciale XARRE, le volume " \
        "de trafic sortant et les parcs d’utilisateurs par type de réseau et d’opérateur (Orange, Expresso, Free, ProMobile, etc) sur les " \
        "offres XARRE, le chiffre d'affaires généré par les recharges des offres XARRE ainsi que du parc actif pour chaque formule tarifaire " \
        "XARRE et les souscriptions aux offres XARRE." \
        \
        "Sur les souscriptions des offres, Notez que nous avons des catégories ou types d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, NC) avec leur " \
        "formule tarifaire ou commerciale (JAMONO NEW S'COOL, JAMONO ALLO, JAMONO PRO, JAMONO MAX, AUTRES) et leurs segments recharges (Mass-Market, High, Middle, S0, super high) " \
        "et marché (JEUNES, ENTRANT, KIRENE AVEC ORANGE, AUTRES,MILIEU DE MARCHE,HAUT DE MARCHE, TRES HAUT DE MARCHE ) correspondant. Chaque type ou catégorie d'offres contient " \
        "plusieurs offres de souscriptions. " \
        \
        "Si vous recevez une question sur le chiffre d'affaire tout court (Donne moi le Ca, quel est l'évolution du CA, etc) sans que l'utilisateur précise le chiffre " \
        "d'affaire, sachez que l'utilisateur veut simplement le chiffre d'affaire total qui n'est rien d'autre que le chiffre d'affaire des recharges " \
        "(ca recharges). Le chiffre d'affaire seulement fait référence au chiffres d'affaires des recharges(ca_recharges) ou CA total." \
        \
        "Evitez les erreurs ou les mélanges sur les noms de colonnes et tables que vous fournissez en sortie, cela repercutera négativement sur l'exécution " \
        "de la requete qui sera produite. Réponds uniquement avec les informations issues des documents fournits sans inventer ou modifier un nom de " \
        "colonne ou table ni mélanger les noms de colonnes des différents tables: " \
        

    #result = qa_chain(query)
 
    with get_openai_callback() as cb:
        result = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Coût (USD) : {cb.total_cost}")

    return result["result"], cb.total_cost
    #return result["result"]

##########################################################################
## Analyste mis à jour
def Agent_analyst_RAG_1(requete) :
    #Créer les embeddings et la base vectorielle
    # model="text-embedding-3-small"
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-large")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)

    #Créer la chaîne RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        # gpt-4o-min
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        #llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    
    # Tester une question
    query = "Vous un Expert Analyste base de données et recherches documentaire.Votre but est de rechercher sur les caractéristiques d'une  \
        base de donnée les tables pour répondre à une question métier.En résumé,vous etes un expert en analyse de données avec plus de 25 ans d'expérience \
        et expert en gouvernance de données : tu documentes des bases  de données métier pour que les équipes comme le BI comprennent chaque \
        champ de la base pour tirer des informations précises.Maintenant Quelle(s) est ou sont la ou les table(s) qui permet(tent) de répondre à la question  \
        suivante sur le RAG:" f"{requete}.""Donne moi en sortie un fichier JSON avec les noms des tables comme clé et leurs descriptions décrit sur le RAG comme valeurs. \
        Les informations du RAG vous donnent le nom des tables, leurs descriptions et ses caractéristiques,pour chaque tables, vous avez le nom des colonnes et la descrition de chaque \
        colonne, de meme qu'aussi pour chaque colonne vous avez son type, quelques exemples de valeurs de la colonne, le sens ou le contexte ou l'utilité de la colonne par rapport au donnée. \
        \
        Autrement dit, Tu as accès à un RAG qui contient la documentation d’une base de données : le nom des tables, leur description, la liste des colonnes avec leurs types et leur signification. \
        Lorsque tu réponds à une question de l’utilisateur,ne propose que les tables réellement présentes dans le catalogue.Retourne uniquement  \
        les tables avec leurs noms réelles ou exactes qui ont été définit dans le catalogue pour répondre à la question, sans ajouter d’informations fictives. \
        \
        Notez bien que les tables commençant comme nom 'daily' contiennent des données journalières des clients celles commençant par 'monthly' \
        répertorient les données mensuelles des clients et celles commençant par 'reporting' font une reporting des clients. \
        Si vous rencontrez 'Sargal' dans la base sachez que c'est un programme de fidélité lancé par l'entreprise pour récompenser  \
        ses clients en fonction du niveau de consommation et d'engagement sur les services proposés. Autrement dit, les clients accumulent \
        des points automatiquement à chaque rechargement de crédit. Plus ils consomment, plus ils cumulent de points.\
        Notez qu'en télécommunications, le terme 'parc' désigne l’ensemble des clients actifs ou abonnés à un service.Sachez que les questions \
        sur le parc actif orange(ou parc actif ou parc orange), vous trouverez les réponses de ses questions sur les tables 'daily' plus \
        précisément 'daily_parc'.Autrement dit, pour les questions sur parc actif orange (sans la précision des clients digitaux), consultez la table 'daily_parc'. \
        Notez qu'aussi Maxit est une application mobile de l'entreprise qui offre plusieurs services (Achat, transfert d'argent, paiement, etc). \
        \
        Avec toute ses informations données sur la base, votre mission est de rechercher la ou les tables de données qui peut ou peuvent fournir une réponse à cette \
        question :" f"{requete}"". Faite une analyse approfondie des caractéristiques ou documents de la base en étudiant le role de chaque tables et colonnes, \
        les relations qui peuvent exister entre les différents tables. Après, vous devrez répondre à cette question métier suivante dans la base de donnée :" f"{requete}"". \
        Sur les questions concernant les parc des takers, si maxit ou digital n'est pas renseigné, vous ne devez pas interroger les tables suffixées MAXIT.\
        Derrière, vous pouvez créer des index pour les tables (les tables concernant les offre, souscriptions, sargal, chiffre \
        d'affaire, recharges, data, etc). Cela vous facilitera la recherche de la réponse. \
        Et aussi notez que le nombre de clients (ou abonnées ou bénéficiaires, etc) est différent au nombre de souscription. Un client peut souscrire dans une \
        offre une ou plusieurs fois. L'identifiant des clients est donné par la colonne 'MSISDN' dans les tables daily et monthly. Alors, le nombre de client doit toujours etre  \
        déterminé par une table contenant cette colonne 'MSISDN'. Notez qu'abonnés, bénéficaires, etc font référence aux clients. \
        Vous allez prendre les résultats de votre analyse sur les caractérisques de la base pour determiner la ou les table(s) nécessaire(s) \
        pour répondre à cette question métier. Vous allez étudier le but de la question métier qui a \
        été posée. C'est à dire quelles sont les informations que l'utilisateur veut savoir par rapport à cette question : \
        "f"{requete}"". Si la réponse peut etre récupérer dans une seule table, en sortie vous donneriez seulement la table et sa description en JSON indiqué ci-dessus. \
        Si la réponse est dans 2 ou plusieurs tables, Vous donnerez tout les tables concernées \
        pour répondre à la question. Analysez la question pour voir, est ce que vous devez de faire une requete \
        simple , d'aggrégation, de jointure ou combiné. Notez que 'OM' signifie 'Orange Money'. Ne donne jamais un " \
        "nom de table qui n'existe pas dans le catalogue de données fournies dans le RAG. " \
        "Notez que la jointure entre les tables peut etre fait entre des colonnes qui ne représente pas l'identifiant " \
        "des clients. Faites bien la différence entre les tables parc, de l'application Maxit, de souscriptions des offres, du trafic réseau (2G/3G/4G/5G) ou de " \
        "consommation de donnée (data) mobile, du programme de fidélité Sargal, de voix, recharges ou chiffre d'affaires (ca), pour les trafic à " \
        "l'international (ca, durée, parc, etc à l'international), etc. Analyse bien la question posée avant de " \
        "donner la ou les tables de sortie. Si vous voyez une question qui parle de paiement (par exemple paiement senelec, sen'eau, " \
        "etc) consultez la table 'daily_maxit' qui contient les infos quotidiennes sur les services de l'application maxit avec sa colonne 'service'." \
        \
        "Les tables commençant par 'daily' et 'monthly' contiennent chacune une colonne 'MSISDN' " \
        "qui représente l'identifiant des clients ou abonnés. Sachez que les tables commençant par 'reporting' n'ont pas une colonne " \
        "pour l'identifiant des clients. Pour les questions sur Maxit, consultez les tables Maxit, pour les questions sur Sargal " \
        "allez sur les tables Sargal, pour les questions sur la data, consultez les tables data, pour les questions sur l'international, " \
        "allez sur les tables internationales, pour les questions sur la souscription des offres, consultez les tables de souscription etc. " \
        "Notez que les tables de données avec comme suffixe 'international' (monthly_international, reporting_monthly_international) " \
        "sur le nom de la table contiennent des infos sur les trafics à l'international (le parc à l'international (parc_pass, parc_international, parc_payg), le chiffre d'affaires à " \
        "l'international (ca_pass, ca_payg), la durée des communications à l'international (ci_pass_dur, duration_mn_payg), etc)" \
        \
        "NB : Notez que la table daily_oss_5g contiennent les informations sur les cellules et leurs trafics (nom cellule, trafic moyen des utilisateurs, etc). daily_clients et " \
        "monthly_clients contiennent les informations quotidiennes et mensuelles des clients (identifiant, infos géographiques et démographiques, segment d'appartenance,etc.). daily_clients_digitaux donne les infos des clients qui utilisent les " \
        "plateformes digitaux.daily_conso fournit le montant des consommations des appels (ou voix) internationaux de chaque client " \
        "(ca_voix_international).monthly_international et reporting_monthly_international donnent le montant ou CA des consommations des communications internationaux pour les forfaits, pour les usages hors forfait et " \
        "leur durée totale. daily_data donne le volume de trafic de la consommation des données mobiles sur le réseau 2G, 3G, 4G et 5G en " \
        "MégaOctets(Mo) tandisque monthly_data ajoute la commune et les segments (marché et recharges) d'appartenance et la formule tarifaire " \
        "souscrite par le client.daily_delta_parc et reporting_parc_monthly donnent les infos sur la sortie et le parc des clients (sortant(0/1), reactive(0/1),parc(0/1),nouvelles " \
        "arrivées(0/1), etc) alors que monthly_sortant ajoute la localité de référence du client (région, département, commune, zone_drv, " \
        "nom_cellule, etc) tandisque daily_parc fournit les infos globlal du parc des clients (nombre d'entrant(entrant_cnt), de sortant (sortant_cnt) et de régularité (regularite), le parc (0/1), etc)." \
        "daily_habit_5g_user donne la consommation Data mobile des Clients (volume_go) sur le réseau 5G par type d'application.daily_heavy_user_4g " \
        "donne le suivi de la consommation 4G des Heavy user. daily_infos_bts donne les informations détaillées sur les sites physiques et leurs " \
        "cellules radios correspondant (NOM_SITE, NOM_CELLULE, CELLID, ID, TYPE_CELLULE, la REGION, le DEPARTEMENT, etc). daily_infos_otarie et monthly_terminaux donnent " \
        "le parc des clients avec leurs appareils mobiles et leurs marques utilisés, le parc d'utilisation du reseau (5G, 4G, 3G, 2G) et les " \
        "volumes de données consommées pour chaque réseau en MégaOctets (Mo). daily_localisation_5g fournit les " \
        "infos géographiques des clients. daily_maxit et monthly_maxit donnent les clients qui se sont connectés sur l'application mobile " \
        "MAXIT avec le service utilisé tandisque reporting_daily_maxit, reporting_monthly_maxit, reporting_daily_parc_maxit et reporting_usage_maxit font le reporting des services et parcs de l'application mobile Maxit." \
        "daily_parc_recharges, daily_parc_pass, daily_parc_maxit, daily_parc_maxit_new, daily_parc_illimix, daily_parc_illiflex, " \
        "daily_parc_data_4g, daily_parc_data et daily_parc_sargal fournissent les informations sur les clients ou le parc client sur la dernière date d'activation " \
        "(last_active_date) ainsi que la fréquence (nb_j_distinct) pour les recharges, les pass, d'utilisation maxit, d'achat illimix et illiflex et data " \
        " du trafic le réseau 4G, de data trafiqué le réseau global et de partciper au programme Sargal. daily_recharges, monthly_recharges et reporting_ca_monthly donne les " \
        "chiffres d'affaires total (ca_recharge) ou générés pour tout les recharges (ca_recharges) sur les différents canaux ainsi que par canal (par OM " \
        "(ca_pass_glob_om_jour, ca_credit_om), par Wave (ca_wave), par Seddo(ca_seddo), par cartes (ca_cartes), à l'international (ca_iah), " \
        "et par self_top_up).monthly_sargal, daily_sargal, reporting_sargal_echangeurs_mon, reporting_sargal_gift_daily, " \
        "reporting_sargal_gift_monthly et reporting_sargal_inscrits donne les infos sur la participation du programme de fidélité Sargal. " \
        "monthly_souscription et daily_souscription fournissent des informations ou le CA des souscriptions (ca_data et ca_voix) des catégories ou type d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, etc) et " \
        "et de l'offre souscrite (ILLIMIX JOUR 500F, PASS 2.5GO,MIXEL 690F, illiflex mois, etc) et permettent aussi de déterminer les clients qui utilisent du data ou voix ou les deux à la fois. reporting_ca_data_monthly fait le reporting du Chiffre d'affaire data (ca_data) mensuelle. reporting_daily_offer, " \
        "reporting_offer_monthly et reporting_souscription_monthly font le le reporting sur les souscriptions des offres (sous_mnt,ca_data, ca_voix, ca_sous_HT) suivant le " \
        "types de souscription et le type d'offre souscrit. reporting_monthly_terminaux fait le reporting des Data_user ou No Data_user avec la colonne data_status " \
        "et leur utilisation de smartphone ou pas. daily_sva, monthly_sva et reporting_monthly_sva donne le montant de la souscription et le parc des " \
        "services à valeur ajouté.daily_voix, monthly_voix et reporting_monthly_voix fournissent le volume ou la durée des appels sortants des abonnées " \
        "sur les différents opérateurs téléphoniques (Orange, Free, Expresso, ProMobile, etc) ainsi leurs parcs clients correspondant. reporting_5g_daily regroupe les sites techniques " \
        "avec une série de KPI quotidiens (Key Performance Indicators) et leur volumes de données consommées liés à l’usage du réseau 5G. reporting_daily_parc fait le reporting quotidienne " \
        "du comportement (actifs ou inactifs) ou parc des abonnées sur les 90 derniers jours. reporting_daily_ca_pag_xarre, " \
        "reporting_daily_data_xarre, reporting_daily_trafic_xarre et reporting_recharge_daily_xarre " \
        "donnent le chiffres d'affaires généré des voix et SMS en mode PAYG (Pay-As-You-Go) pour les formules commerciales XARRE, " \
        "le trafic Internet et le parc d’abonnés par technologie réseau (2G, 3G, 4G, 5G) ventilées par formule commerciale XARRE, le volume " \
        "de trafic sortant et les parcs d’utilisateurs par type de réseau et d’opérateur (Orange, Expresso, Free, ProMobile, etc) sur les " \
        "offres XARRE, le chiffre d'affaires généré par les recharges des offres XARRE ainsi que du parc actif pour chaque formule tarifaire " \
        "XARRE. reporting_xarre_offer_nrt donne le parc, le ca et le volume des souscriptions aux offres XARRE par type d'offres, par offre, par " \
        "segment, par tranches d'heure de souscriptions (ex : 00h-01h, 13h-14h, etc). Donnez en sortie les tables les plus pertinantes pour répondre à la question :"f" {requete}." \
        \
        "Sur les souscriptions des offres, Notez que nous avons des catégories ou types d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, NC) avec leur " \
        "formule tarifaire ou commerciale (JAMONO NEW S'COOL, JAMONO ALLO, JAMONO PRO, JAMONO MAX, AUTRES) et leurs segments recharges (Mass-Market, High, Middle, S0, super high) " \
        "et marché (JEUNES, ENTRANT, KIRENE AVEC ORANGE, AUTRES,MILIEU DE MARCHE,HAUT DE MARCHE, TRES HAUT DE MARCHE ) correspondant. Chaque type ou catégorie d'offres contient " \
        "plusieurs offres de souscriptions et formule et catégorie d'offres sont différentes. Notez qu'aussi les offres data font références aux offres de type PASS INTERNET." \
        \
        "Si vous recevez une question sur le chiffre d'affaire tout court (Donne moi le Ca, quel est l'évolution du CA, etc) sans que l'utilisateur précise le chiffre " \
        "d'affaire, sachez que l'utilisateur veut simplement le chiffre d'affaire total qui n'est rien d'autre que le chiffre d'affaire des recharges " \
        "(ca recharges). Le chiffre d'affaire seulement fait référence au chiffres d'affaires des recharges(ca_recharges) ou CA total. Pour les questions " \
        "sur la segmentation, ne choisissez jamais les tables de reporting car elle ne sont pas adapté, utilise plutot les tables monthly (mensuelles) " \
        "ou daily (quotidiennes)" \
        \
        "Evitez les erreurs sur les noms de tables que vous fournissez en sortie, cela repercutera négativement sur l'exécution " \
        "de la requete qui sera produite. Réponds uniquement avec les informations issues des documents fournits sans inventer ou modifier un nom de " \
        "table. " \
        "Voici un exemple de sortie : " \
        "{" \
        "   'daily_parc_maxit' : 'C’est une table journalière qui fournit les informations sur les clients ou le parc clients qui ont eu à faire  \
                                une activité sur l’application mobile MAXIT. Elle contient les données d’usage de l’application MAXIT pour \
                                chaque abonné avec des informations sur la fréquence, la récence et la périodicité de l’utilisation. Chaque \
                                ligne correspond à un MSISDN (identifiant unique d’un abonné) et inclut la date de dernière activité (last_active_date) \
                                ainsi que le nombre de jours distincts d’activité (nb_j_distinct). Les colonnes year, month et day précisent la période \
                                d’utilisation de l’application. Elle permet d’identifier les clients actifs ou inactifs de l’application mobile MAXIT ou \
                                en déclin d’usage, de segmenter les utilisateurs MAXIT selon la fréquence d’usage (nb_j_distinct) et la récence \
                                (last_active_date). Elle est conçue pour analyser l’engagement des clients sur l’application mobile MAXIT' " \
        "}" \
        

    result = qa_chain(query)
    
    with get_openai_callback() as cb:
        result = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Coût (USD) : {cb.total_cost}")
    
    print(result["result"])
    return result["result"], cb.total_cost
    #return result["result"]



def Agent_reflex_pattern(requete, result_table):
    #Découper le texte en chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    #text_splitter = RecursiveCharacterTextSplitter(
    #                    chunk_size=2000,   # chaque chunk fait 1000 caractères max
    #                    chunk_overlap=200  # un petit chevauchement pour le contexte
    #                                                )

    docs = text_splitter.split_documents(documents)
    #docs = text_splitter.split_documents(docu)

    #Créer les embeddings et la base vectorielle
    # model="text-embedding-3-small"
    embeddings = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-large")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)

    #Créer la chaîne RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        #gpt-4o-min
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        #llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    
    # Tester une question
    query = "Vous etes un Expert Analyste base de données, recherches documentaire, Reflex Pattern Agent,Schema Verifier et Catalogue RAG Checker.Votre " \
    "but est de Vérifier l'existence des tables(et leurs colonnes) dans le catalogue RAG et signaler celles inconnues avec suggestions. \
    Pour chaque table valide, vérifiez l'existence des colonnes et proposer des alternatives si besoin et Réagissez immédiatement: si tout est \
    valide, confirmer le mapping tables->colonnes.Sinon, corriger automatiquement (tables & colonnes) et produire un mapping corrigé. " \
    "Vous etes aussi Gardien du catalogue, tu maîtrises les noms exacts de tables et leurs variantes.Tu connais parfaitement les schémas de chaque table." \
    "Considérez vous comme un Agent réflexe qui applique des règles if-then: vérifie, corrige, et renvoie des données propres pour l'exécution SQL. \
    Maintenant, documentez le RAG pour voir est ce que les informations suivantes sur les tables correspondent exactement à celles du RAG \n\n" 
    f"{result_table}.""Donne moi en sortie le(s) nom(s) de la ou des tables et tout ses caractéristiques (tout les colonnes, type, "
    "description, etc) après correction s'il y a en.Si le nom de la table n'existe pas dans le catalogue, fait une autre recherche sur le RAG catalogue pour déterminer "
    "la ou les table(s) de données existant qui permet(tent) de répndre à la question métier suivante :"f"{requete} \
        \
        Autrement dit, Tu as accès à un RAG qui contient la documentation d’une base de données : le nom des tables, leur description, la liste des colonnes avec leurs types et leur signification. \
        Lorsque tu réponds à une question de l’utilisateur,ne propose que les tables et colonnes réellement présentes dans le catalogue.Si une table ne contient pas une colonne \
        mentionnée dans la question, ne l’invente pas.Par exemple pour une question sur les régions, si une table contient commune et département mais pas région, n'invente pas une colonne région dans cette table. \
        Si la réponse nécessite une colonne absente de la table interrogée, identifie d’autres tables qui contiennent cette information.Dans ce cas,explique qu’il faut \
        faire une jointure ou une liaison entre ces tables.Retourne uniquement les tables et colonnes(tout les colonnes de chaque table) pour répondre à la question, \
        sans ajouter d’informations fictives. \
        \
        Notez bien que les tables commençant comme nom 'daily' contiennent des données journalières des clients celles commençant par 'monthly' \
        répertorient les données mensuelles des clients et celles commençant par 'reporting' font une reporting des clients. \
        Vous rencontrez 'sargal' dans la base sachez que c'est un programme de fidélité lancé par l'entreprise pour récompenser  \
        ses clients en fonction du niveau de consommation et engagement sur les services proposés. Autrement dit, les clients accumulent \
        des points automatiquement à chaque rechargement de crédit. Plus ils consomment, plus ils cumulent de points.\
        Notez qu'en télécommunications, le terme 'parc' désigne l’ensemble des clients actifs ou abonnés à un service.Sachez que les questions \
        sur le parc actif orange(ou parc actif ou parc orange), vous trouverez les réponses de ses questions sur les tables ayant sur le nom le \
        mot '_parc_'. \
        Notez qu'aussi Maxit est une application mobile de l'entreprise qui offre plusieurs services (Achat, transfert d'argent, paiement, etc). \
        A chaque fois que vous recevez une question qui parle de 'data', sachez que cela fait référence à internet. Autrement dit, dans les données, 'data' veut dire 'internet'.\
        Avec toute ses informations données sur la base, votre mission est de rechercher la ou les tables de données qui peut ou peuvent fournir une réponse à cette \
        question si les tables de données fournies ci-dessus n'existent pas:" f"{requete}"". Faite une analyse approfondie des caractéristiques ou documents de la base en étudiant le role de chaque tables et colonnes, \
        les relations qui peuvent exister entre les différents tables. Après, vous devrez répondre à cette question métier suivante dans la base de donnée :" f"{requete}"". \
        Sur les questions concernant les parc des takers, si maxit ou digital n'est pas renseigné, vous ne devez pas interroger les tables suffixées MAXIT.\
        Derrière, vous pouvez créer des index pour les tables (les tables concernant les offre, souscriptions, sargal, chiffre \
        d'affaire, recharges, data, etc). Cela vous facilitera la recherche de la réponse. \
        Et aussi notez que le nombre de clients (ou abonnées ou bénéficiaires, etc) est différent au nombre de souscription. Un client peut souscrire dans une \
        offre une ou plusieurs fois. L'identifiant des clients est donné par la colonne 'MSISDN'. Alors, le nombre de client doit toujours etre  \
        déterminé par une table contenant cette colonne. Notez qu'abonnés, bénéficaires, etc font référence aux clients. \
        Vous allez prendre les résultats de votre analyse sur les caractérisques de la base pour determiner la ou les table(s) nécessaire(s) \
        pour répondre à cette question métier. Vous allez étudier le but de la question métier qui a \
        été posée. C'est à dire quelles sont les informations que l'utilisateur veut savoir par rapport à cette question : \
        "f"{requete}"". Si la réponse peut etre récupérer dans une seule table, en sortie vous donneriez seulement la table et tout les \
        colonnes de la table. Si la réponse est dans 2 ou plusieurs tables, Vous donnerez tout les tables concernées et leurs \
        colonnes pour répondre à la question. Analysez la question pour voir, est ce que vous devez de faire une requete \
        simple , d'aggrégation, de jointure ou combiné. Notez que 'OM' signifie 'Orange Money'. Ne donne jamais un " \
        "nom de table qui n'existe pas dans le catalogue de données fournies et aussi ne jamais un nom de " \
        "colonne qui n'existe pas dans une table des données de la catalogues."

    #result = qa_chain(query)
    #     
    with get_openai_callback() as cb:
        result = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Coût (USD) : {cb.total_cost}")
    

    return result["result"], cb.total_cost
    #return result["result"]



def Agent_info_supp(result_table, requete, user_dir) :
    document = [
        #Document(page_content= dict_to_text(dataa[item]))
        Document(page_content=dict_to_text(item))

        #Document(page_content= dataa[item])
        #Document(page_content= str(dataa[item]))
        #metadata={"table": item}
        for item in data_supp
        
    ]

    #Découper le texte en chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap= 75)
    #docs = text_splitter.split_documents(documents)
    docs = text_splitter.split_documents(document)

    #les embeddings et la base vectorielle
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    #C RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        #gpt-4o-min
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        #llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    #result_table = Agent_analyst_RAG()
    print(f"\n\n{result_table}")

    query = "Vous étudiez les colonnes des différents tables fourni :"f"{result_table}.Vous allez regarder est ce qu'il  \
        y a une colonne ou des colonnes qui sont dans le RAG (colonne : valeurs uniques ou présence  \
        du nom partiel de colonne : valeurs uniques).Le RAG contient des informations d'une colonne ou d'un ensemble  \
        de colonne.Autrement dit, vous avez des noms de colonnes avec leurs valeurs uniques (exemple : SEGMENT_RECHARGE : S0, Mass-Market, \
        Middle, High, Super High) ou des noms (groupe de mots) qui représentent plusieurs colonnes avec leurs valeurs uniques (exemple : \
        Pour les colonnes réprésentant les regions : TAMBACOUNDA, ZIGUINCHOR, DIOURBEL, DAKAR, THIES,SAINT LOUIS, etc) \
        Si après votre étude, vous trouvez une ou des colonnes dans le RAG,alors vous allez étudiez la \
        question pour voir la ou les valeurs unique(s) de la ou les colonnes qui permet(tent) de répondre à la \
        question : "f"{requete}, puis vous allez récupérerer la ou les valeurs et les donnez en sortie pour que un autre agent l'utilise \
        sur sa production de requete SQL au niveau de la clause WHERE.Ce qui permettra à l'agent de produire une requete \
        avec un clause WHERE qui s'adapte avec les informations de la base de données.Par exemple, pour une question comme \
        'Quelle est le montant total des paiements SENELEC', pour \
        répondre à cette question, vous étudiez la colonne qui permet de répondre à cette question, et sur ses éléments, \
        quelle est ou sont l'élément(s) unique(s) de la colonne qui permet de répondre à la question.Dans cet exemple, ça correspond à la \
        la colonne 'service' avec sa valeur unique 'BY_SENELEC' qui sera utilisée sur la requete qui sera produite par  \
        un agent sur sa clause WHERE.Un autre exemple,'donne moi le parc des illimix jour de février 2025',dans cet exemple vous allez  \
        recupérer l'élément unique correspondant au illimix jour dans le nom des offres et le donner en sortie (ici c'est 'illimix jour 500F'). Il en est de meme  \
        pour tout autres questions de ce genres.Par contre, si vous avez une question où vous avez besoins de tout les éléments de la \
        colonnes, dans ce cas, ce n'est pas la peine de donner en sortie des informations supplémentaires.Par exemple,'Donne moi \
        le chiffre d'affaires des offres en 2025',sur cet exemple, vous avez besoin de toute la colonne qui représente les offres, alors \
        ce n'est pas la peine d'envoyer en sortie les valeurs uniques des offres.Un autre exemple, 'Donne moi le revenu HT des segments \
        marchés',pour répondre à cette question, vous avez besoin de tout les segments marché, alors ce n'est pas la \
        peine de retourner les valeurs uniques de la colonne qui représente les segments marchés.L'agent 'Producteur de requete SQL' \
        se chargera de la gestion de ses valeurs uniques.Il en est de meme pour toute ses genres de questions.En résumé,vous allez retourner \
        uniquement des valeurs lorsque la question qui a été posée spécifie une ou quelques valeur(s) unique(s) d'un ou des colonne(s). \
         Notez que le RAG contient quelques variables catégorielles et leur valeurs uniques.Votre mission est de recupérer \
        la ou les valeur(s) unique(s) nécessaire(s) de la ou les colonne(s) pour répondre à la question qui a été posée.Notez que 'OM' \
        signifie 'Orange Money'.Si la question ne nécessite pas d'info supplémentaire sur le RAG, donnez en retour qu'elle n'a pas besoin d'infos sup. \
        Donner en sortie une ou des colonnes (s'il y en a dans le RAG) avec quelque(s) de sa ou ses valeurs uniques qui permet(tent) \
        de répondre à la question : {requete}.Par exemple, [ service : 'By_SENELEC' pour les paiement SENELEC, segment_marche : ['JEUNES', 'ENTRANT'] pour le marché des JEUNES et des ENTRANT,  \
        ca_cr_commune_90 : 'DAKAR' pour la commune de Dakar sur les 90 derniers jours ,offer_name = [ 'PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO'], etc pour \
        des souscriptions à des offres comme 'PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO', etc] ou bien pas besoins d'informations \
        supplémentaires s'il n'en a pas. Si la question n'est pas spécifique à une valeur d'une variable, cela veut dire qu'on a \
        besoin de tout les infos de la variables, dans ce cas ce n'est pas la peine de retourner des valeurs uniques de la variables"

    #result = qa_chain(query)

    # Gestion des couts
    with get_openai_callback() as cb:
        result = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Cout (USD) : {cb.total_cost}")

    path = f"{user_dir}_infos_RAG.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(result["result"])

    #print("Réponse :", result["result"])
    #print("Source :", result["source_documents"][0])

    return result["result"], cb.total_cost
    #return result["result"]


####################################################################
#       infos sup mis à jour
def Agent_info_supp_1(result_table, requete, user_dir) :
    document = [
        #Document(page_content= dict_to_text(dataa[item]))
        Document(page_content=dict_to_text(item))

        #Document(page_content= dataa[item])
        #Document(page_content= str(dataa[item]))
        #metadata={"table": item}
        for item in data_supp
        
    ]

    #Découper le texte en chunks
    #text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap= 40)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap= 75)
    #docs = text_splitter.split_documents(documents)
    docs = text_splitter.split_documents(document)

    #les embeddings et la base vectorielle
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    #C RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        #gpt-4o-min
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        #llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    #result_table = Agent_analyst_RAG()
    print(f"\n\n{result_table}")

    query = "Vous étudiez les colonnes des différents tables fourni :"f"{result_table}.Vous allez regarder est ce qu'il  \
        y a une colonne ou des colonnes qui sont dans le RAG (colonne : valeurs uniques ou présence  \
        du nom partiel de colonne : valeurs uniques).Le RAG contient des informations de plusieurs colonnes catégorielles.  \
        Autrement dit, vous avez des noms de colonnes avec leurs valeurs uniques (exemple : SEGMENT_RECHARGE : S0, Mass-Market, \
        Middle, High, Super High) ou des noms (groupe de mots) qui représentent plusieurs colonnes avec leurs valeurs uniques (exemple : \
        Pour les colonnes réprésentant les regions : TAMBACOUNDA, ZIGUINCHOR, DIOURBEL, DAKAR, THIES,SAINT LOUIS, etc) \
        Si après votre étude, vous trouvez une ou des colonnes dans le RAG,alors vous allez étudiez la \
        question pour voir la ou les valeurs unique(s) de la ou les colonnes qui permet(tent) de répondre à la \
        question : "f"{requete}, puis vous allez récupérerer la ou les valeurs et les donnez en sortie pour que un autre agent l'utilise \
        sur sa production de requete SQL au niveau de la clause WHERE.Ce qui permettra à l'agent de produire une requete \
        avec un clause WHERE qui s'adapte avec les informations de la base de données.Par exemple, pour une question comme \
        'Quelle est le montant total des paiements SENELEC', pour \
        répondre à cette question, vous étudiez la colonne qui permet de répondre à cette question, et sur ses éléments, \
        quelle est ou sont l'élément(s) unique(s) de la colonne qui permet de répondre à la question.Dans cet exemple, ça correspond à la \
        la colonne 'service' avec sa valeur unique 'BY_SENELEC' qui sera utilisée sur la requete qui sera produite par  \
        un agent sur sa clause WHERE.Un autre exemple,'donne moi le parc des illimix jour de février 2025',dans cet exemple vous allez  \
        recupérer l'élément unique correspondant au illimix jour dans le nom des offres et le donner en sortie (ici c'est 'illimix jour 500F'). Il en est de meme  \
        pour tout autres questions de ce genres.Par contre, si vous avez une question où vous avez besoins de tout les éléments de la \
        colonnes, dans ce cas, ce n'est pas la peine de donner en sortie des informations supplémentaires.Par exemple,'Donne moi \
        le chiffre d'affaires des offres en 2025',sur cet exemple, vous avez besoin de toute la colonne qui représente les offres, alors \
        ce n'est pas la peine d'envoyer en sortie les valeurs uniques des offres.Un autre exemple, 'Donne moi le revenu HT des segments \
        marchés',pour répondre à cette question, vous avez besoin de tout les segments marché, alors ce n'est pas la \
        peine de retourner les valeurs uniques de la colonne qui représente les segments marchés.L'agent 'Producteur de requete SQL' \
        se chargera de la gestion de ses valeurs uniques.Il en est de meme pour toute ses genres de questions.En résumé,vous allez retourner \
        uniquement des valeurs lorsque la question qui a été posée spécifie une ou quelques valeur(s) unique(s) d'un ou des colonne(s). \
        Notez que le RAG contient quelques variables catégorielles et leur valeurs uniques.Votre mission est de recupérer \
        la ou les valeur(s) unique(s) nécessaire(s) de la ou les colonne(s) pour répondre à la question qui a été posée.Notez que 'OM' \
        signifie 'Orange Money'.Si la question ne nécessite pas d'info supplémentaire sur le RAG, donnez en retour qu'elle n'a pas besoin d'infos sup. \
        Donner en sortie une ou des colonnes (s'il y en a dans le RAG) avec quelque(s) de sa ou ses valeurs uniques qui permet(tent) \
        de répondre à la question : {requete}.Par exemple, [ service : 'By_SENELEC' pour les paiement SENELEC, segment_marche : ['JEUNES', 'ENTRANT'] pour une question sur le marché des JEUNES et des ENTRANT,  \
        ca_cr_commune_90 : 'DAKAR' pour une question sur la commune de Dakar sur les 90 derniers jours ,offer_name = [ 'PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO'], etc pour une question\
        les souscriptions aux offres 'PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO', etc] ou bien pas besoins d'informations \
        supplémentaires s'il n'en a pas. Si la question n'est pas spécifique à une valeur d'une variable, cela veut dire qu'on a \
        besoin de tout les infos de la variables, dans ce cas ce n'est pas la peine de retourner des valeurs uniques de la variables"

    #result = qa_chain(query)

    # Gestion des couts
    
    with get_openai_callback() as cb:
        result = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Cout (USD) : {cb.total_cost}")
    
    path = f"{user_dir}_infos_RAG.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(result["result"])

    #print("Réponse :", result["result"])
    #print("Source :", result["source_documents"][0])

    return result["result"], cb.total_cost
    #return result["result"]




def Agent_SQL(requete, user_dir) :

    #result_table, cout1 = Agent_analyst_RAG(requete)
    result_table, cout1 = Agent_analyst_RAG_1(requete)
    print(result_table)

    path = f"{user_dir}_output_tab.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(result_table)
    
    result_table = Extract_tables(path)

    data = json.loads(result_table) 
    list_tab = []
    path = f"{user_dir}_desc_tab_output.txt"
    for key, value in data.items():
        info_tab = {
            "nom_table" : key,
            "description" : value,
            "colonnes" : caracteristique[key]
        }
        list_tab.append(info_tab)
    result_table = list_tab

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{list_tab}")

    #info_sup, cout2 = Agent_info_supp(result_table, requete, user_dir)
    info_sup, cout2 = Agent_info_supp_1(result_table, requete, user_dir)
    print(info_sup)

    agent = OpenAI(api_key=openai_key)

    # Appel à GPT-4
    response = agent.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Vous allez recupérer uniquement la ou les table(s) de données suivante(s): {result_table}, " \
                    # et les informations supplémentaires données en sortie par la tache 'Infos supplémentaire'(s'il y en a) .Ses infos supplémentaires 
                    # sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut 
                    # etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous
                    # permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une valide et correcte par rapport à nos données colonne(s) et.
                      "et les informations supplémentaires (s'il y en a) :"f" {info_sup} " \
                      "pour produire une requete SQL valide qui permet de répondre à cette question métier :"f" {requete}.Ses infos supplémentaires" \
                      "sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut" \
                      "etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous" \
                      "permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une requete SQL " \
                      "valide et correcte par rapport à nos données.Sur la clause WHERE de la requete, utilisez la ou les valeur(s) donnée "
                      "par les infos supplémentaire si la question posée spécifie une ou des éléments d'une ou des colonnes fournies (Par exemple : donne "
                      "moi le CA des clients Mass-market? donne moi le parc des takers de l'offre PASS NUIT 5 GO?, etc). Si la question concerne tout les "
                      "éléments d'une colonne fournie dans les infos supplémentaires, ne met jamais cette colonne sur la clause WHERE car elle se sera unutile.  " \
                      "Notez qu'à chaque fois que vous recevez une question qui parle de 'data',sachez que cela fait référence à internet. Autrement " \
                      "dit, dans les données,'data' veut dire 'internet' ou 'pass internet' (Par exemple : offres data signifie offres de type 'pass internet')." \
                      "Abonnés, bénéficaires, etc font référence aux clients.'OM' signifie 'Orange Money'."
                      #"Et aussi le nombre de clients (ou abonnées,etc) est différent au nombre de souscription"
                      " Analyse bien la ou les tables reçue(s) pour voir est ce que l'information de la question posée est dans une ou plusieurs " \
                      "colonnes. Regarde bien aussi est-ce-que vous avez reçu une ou plusieurs table(s)." \
                      "Parfois, vous pouvez recevoir plusieurs tables, et que chaque table peut répondre à la question posée. Dans ce cas, prenez " \
                      "la table qui donne le plus d'information pour répondre à la question.Si vous avez reçu "
                      "une seule table, produit une requete SQL correcte avec cette table seulement en prenant que les colonnes qui contiennent"
                      "l'information de la réponse sur la table. S'il y a 2 "
                      "ou plusieurs tables reçues et que la réponse se trouve sur les différents tables, analyse les colonnes de chaque table, puis étudier les dépendances "
                      "entre les tables, c'est à dire les colonnes qui permet de faire la liaison entre les tables."
                      "Après cela, produit une requete SQL correcte avec les colonnes qui contiennent l'information sur les différentes " \
                      "tables reçues en faisant des jointures pour répondre à cette question métier : "
                      f"{requete}"". Si la question est trop vaste, (par exemple 'Je veux le chiffre d'affaire', 'Quelle formule tarifaire " \
                      "génère le plus de trafic 4G sur le mois d'avril', ect) vous essayerez toujours de répondre en donnant une " \
                      "requete SQL qui donne les informations les plus récentes. Dans cet exemple, vous donneriez le chiffre d'affaire ou " \
                      "formule tarifaire de l'année en cours (YEAR(CURRENT_DATE()))), Si vous avez une requete qui nécessite une condition et si la condition " \
                      "doit se faire avec des caractères ou chaines de caractères (Ex : Région de Dakar), sur "
                      "la clause WHERE, Utilise LIKE plutot égal (=) par exemple '%Dakar%', '%DAKAR' ou '%kar%', etc .Référez vous aussi sur les infos supplémentaires données ci-dessus " \
                      "Exemple de question : Quelle commune a généré le plus de CA data pour JAMONO NEW S’COOL? Sur la clause WHERE vous pouvez mettre " \
                      "par exemple variable LIKE '%JAMONO' ou variable LIKE '%NEW S’C%', etc. Appliquez ses exemples dans ses genres cas." \
                      "En résumé n'utilise jamais égal dans une clause WHERE avec caractère ou chaine de caractères, utilise toujours LIKE avec une partie du groupe de mot."
                      "N'utilisez jamais tout les mots donnés sur la question sur la condition de la requete (par exemple : variable LIKE '%JAMONO NEW S’COOL%' comme dans l'exemple précédent)"
                      "Sur les questions concernant les parc des takers, si 'maxit' ou 'digital' n'est pas renseigné sur la question, vous "
                      "ne devez pas interroger les tables suffixées 'maxit'.Attention ne fait jamais une requete pour supprimer, "
                      "pour modifier ou pour mettre à jour ou pour insérer dans la base. votre but est de sélectionner, alors "
                      "mettez seulement des requetes SQL qui permet de faire la sélection. Sélectionnez toujours les colonnes 'year' et 'month' " \
                      "sur la requete et utilisez les memes nom de colonne pour les alias (exemple : as year et as month). Sachez que le mois est toujours sous format numérique." \
                      "Si l'année n'est pas spécifiée ou renseignée sur la question, filtrez toujours sur l'année en cours (YEAR(CURRENT_DATE()))) ou le max des années pour ne pas "
                      "retourner les données de tout les années.Autrement dit, fait toujours un filtre de l'année en cours si l'année n'est " \
                      "pas renseigné sur la question, utilise toujours des 'Alias' par exemple 'nom_table.nom_colonne' " \
                      "pour éviter d'avoir des erreurs d'exécution provoqué par un nom de colonne.Notez que les colonnes avec comme nom 'day' représente les "
                      "jours et ont pour format numérique (yyyymmdd).Gardez cela en mémoire, il vous servira pour les questions sur les données quotidiennes.A "
                      "Notez que 'Airtime' signifie recharge à partir de ton crédit.Par rapport au question de segmentation, "
                      "analyse bien la question pour donner en retour une réponse claire qui permet de définir bien les différents segments(ou clusters) demandés." \
                      "Sur la requete qui sera produite, ne met jamais une limite à moins que la requete vous " \
                      "l'oblige à le faire. Par exemple, vous pouvez avoir comme requete : Donnez le top 10 des chiffres d'affaires des régions ou Quelle commune "
                      "a la balance moyenne SARGAL la plus élevée.Dans ses genres de question,vous pouvez utilisez la clause LIMIT dans la requete." \
                      "La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une requete SQL valide qui peut etre exécutée " \
                      "sur n'importe quelle version de HiveSQL sans erreur. Pour les questions sur la corrélation, n'utilise jamais la fonction " \
                      "d'aggrégation 'CORR()' car cela n'a pas marché. l'erreur dit que cette fonction n'est pas supportée, essaie plutot de " \
                      "le calculer en appliquant la formule de la corrélation.Autrement dit, n'utilise jamais une fonction SQL obsolète dans "
                      "la requete.Ne met jamais de requete avec une sous requete sur la clause WHERE." \
                      "Par exemple,  WHERE year = (SELECT ....), Utilise plutot JOIN ON à place. "
                      
                      "Sur la requete SQL, ordonnez toujours le résultat du plus récents au plus anciens ou du plus grand au plus petit."
                      "Analysez la question pour voir, est ce que vous devez de faire une requete simple,d'aggrégation,de jointure ou combiné." \
                      
                      "Sur les questions sur les souscriptions des offres, Notez que nous avons des catégories ou types d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, NC) avec leur " \
                    "formule tarifaire ou commerciale (JAMONO NEW S'COOL, JAMONO ALLO, JAMONO PRO, JAMONO MAX, AUTRES) et leurs segments recharges (Mass-Market, High, Middle, S0, super high) " \
                    "et marché (JEUNES, ENTRANT, KIRENE AVEC ORANGE, AUTRES,MILIEU DE MARCHE,HAUT DE MARCHE, TRES HAUT DE MARCHE ) correspondant. Chaque type ou catégorie d'offres contient " \
                    "plusieurs offres de souscriptions. Notez qu'aussi les offres data font références aux offres de type PASS INTERNET." \
                    \
                    "Si vous recevez une question sur le chiffre d'affaire tout court (Donne moi le Ca, quel est l'évolution du CA, etc) sans que l'utilisateur précise le chiffre " \
                    "d'affaire, sachez que l'utilisateur veut simplement le chiffre d'affaire total qui n'est rien d'autre que le chiffre d'affaire des recharges " \
                    "(ca recharges). Le chiffre d'affaire seulement fait référence au chiffres d'affaires des recharges(ca_recharges) ou CA total." \
                    \
                    "Produit en retour une requete SQL qui est claire, structurée, correcte et exécutable sous Hive ,prenez seulement tout les "
                      "colonnes qui contiennent l'information de la question.Utilisez toujours des alias dans la requete et ajouter les colonnes year"
                      
                        "et month dans la sélection.La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une " \
                        "requete SQL valide sous Hive. Voici quelques genres d'exemples de requete SQL pour la sortie : "
                        "SELECT c.year as year,c.month as month,c.id, c.nom, c.ville, c.date_vente FROM clients as c WHERE c.ville LIKE "
                        "'Dakar' AND c.year = YEAR(CURRENT_DATE());"
                        "SELECT v.year as year,v.month as month,v.id, v.nom, v.ville, MONTH(v.date_vente), SUM(v.montant) FROM "
                        "ventes v WHERE v.year = YEAR(CURRENT_DATE()) GROUP BY MONTH(v.date_vente) ORDER BY MONTH(v.date_vente);"
                        "SELECT clients.year as year, clients.month as month, clients.id_client, clients.nom, clients.ville, " \
                        "ventes.date_vente ventes.montant FROM clients JOIN ventes ON clients.id_client = ventes.id_client WHERE clients.year = 2024; " \
                        "SELECT rcdm.year AS year, rcdm.month AS month,(AVG(rcdm.volumes * rcdm.ca_data) - AVG(rcdm.volumes) * AVG(rcdm.ca_data)) / "
                        "(STDDEV_POP(rcdm.volumes) * STDDEV_POP(rcdm.ca_data)) AS correlation_volume_revenu FROM reporting_ca_data_monthly AS rcdm JOIN " \
                        "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS max_year_subquery ON rcdm.year=max_year_subquery.max_year " \
                        "GROUP BY rcdm.year, rcdm.month ORDER BY rcdm.year DESC, rcdm.month DESC;" \
                        "SELECT T1.year AS year,T1.month AS month,SUM(T1.ca_data) AS revenu FROM reporting_ca_data_monthly AS T1 JOIN "
                        "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS T2 ON T1.year = T2.max_year GROUP BY " \
                        "T1.year, T1.month ORDER BY T1.year DESC, T1.month DESC;"
                        
                        "NB : N'oubliez pas de faire la filtre sur les années, si l'année n'est pas précisé sur la question ({requete}), faite un filtre "
                        "sur l'année en cours (par exemple year = YEAR(CURRENT_DATE())). Le YEAR(CURRENT_DATE())) est obligatoire si l'année n'est pas spécifié sur la question. Si vous n'avez pas reçu de tables, n'invente pas de tables et ne "
                        "produit pas de requete, dites juste à l'utilisateur de reformuler sa question pour qu'il soit beaucoup plus clair.Utilisez que les "
                        "colonnes qui vous ont été fourni sur la ou les tables et n'essayez pas d'ajouter ou d'inventer une colonne meme si les informations "
                        "de la ou des table(s) sont insuffisant"
                      
                    }
                ]
            }
        ],
        #max_tokens=200,
        #max_completion_tokens=600
    )

    # Affiche les résultats
    
    #print(response.choices[0].message.content)

    #x = response.choices[0].message.content

    #responses = f"{responses} \n\n{x}"
    responses = response.choices[0].message.content
    #print(f"{response}\n\n")
    # calcule du cout pour gpt-4o
    
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    #cout en dollars
    cout3 = (prompt_tokens * 0.0025 / 1000) + (completion_tokens * 0.01 / 1000)

    cost = cout2 + cout1 + cout3
    print(f"Cout total : {cost} $")

    path = f"{user_dir}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(responses)
    
    sql_query = Extract_sql(path)
    print(sql_query)

    return sql_query, cost
    #return sql_query

def Agent_RAG_KPI(echantillon, requete) :
    document = [
        #Document(page_content= dict_to_text(dataa[item]))
        Document(page_content=dict_to_text(item))

        for item in data_kpi
        
    ]
    #Découper le texte en chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #docs = text_splitter.split_documents(documents)
    docs = text_splitter.split_documents(document)

    #les embeddings et la base vectorielle
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_documents(docs, embeddings)

    #RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        #gpt-4o-mini
        #llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        retriever=retriever,
        return_source_documents=True
    )
    query = f"""
        Tu es un agent IA spécialisé dans l’analyse de données surtout sur la recherche de KPI.Ta mission est d’examiner cette échantillon :\n
        {echantillon},\n puis de rechercher et d’identifier sur le RAG les KPI les plus pertinents pour mesurer sa performance, que ce soit en 
        termes d’évolution, de variation, de tendance ou de comparaison.Pour chaque KPI proposé, explique à quoi il correspond concrètement, 
        en quoi cet indicateur permet de juger de la performance globale et pourquoi il est adapté au contexte métier.L'échantillon fourni 
        réprésente que les 5 premiers lignes du dataset.Le dataset est issue de la réponse à question métier suivante : {requete}.
        Notez que sur le RAG, vous avez les familles de KPI(Revenu, Parc, trafic,etc), les KPI primaires(Ca recharge, Ca data, Ca souscrition, 
        Parc, Trafic,etc), les KPI recommandés (Entrants, sortants, NA (nouveaux abonnés) Ca data, Ca bundles, Ca pass, Ca illiflex, Ca par région, Ca par marché,
        etc), les raisons d'utilisations des KPI.Pour chaque KPI primaire, il y a un KPI recommandé pour calculer sa performance.Notez que le 
        parc est le nombre de clients abonnés ou actifs. Le Ca est le chiffre d'affaire ou revenu généré par les clients.Pour les questions sur 
        le parc ou nombre d'abonnés ou de clients, etc utilisez les KPI liés au Parc, pour les questions de Ca, c'est les KPI liés au revenu et 
        pour les questions de voix ou de data, le KPI lié au Trafic est recommandé. Mais parfois vous pouvez avoir des questions qui peuvent 
        liées à la fois des Revenus et des Parc, des Parc et des Trafic,etc. Donnez
        une sortie très claire et très structurée des différentes KPI que vous trouverez à la fin de votre recherche afin de permettre de 
        faire au futur une analyse claire et exploitable par les décideurs. Mettez un format de sortie avec une petite explication globale 
        des différents KPI trouvé et après pour chaque KPI, vous donnez une explication très claire par rapport à la question métier qui a  
        été posé et au résultat de son échantillon(par exemple, une explication global des KPI après tu va à la ligne vous mettez 
        KPI : explication et raison).
        """
    
    result = qa_chain(query)
    ################# Cout du modèle
    """
    with get_openai_callback() as cb:
        result = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Cout (USD) : {cb.total_cost}")
    """


    with open("infoKPI.txt", "w", encoding="utf-8") as f:
        f.write(result['result'])
    
    #return result['result'], cb.total_cost
    return result['result']


def Agent_tables_KPI(requete, echantillon, info_kpi) :
    #Créer les embeddings et la base vectorielle
    # model="text-embedding-3-small"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #embeddings = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-large")
    db = FAISS.from_documents(docs, embeddings)

    #Créer la chaîne RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0, google_api_key = api_k),
        # gpt-4o-min
        #llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        retriever=retriever,
        return_source_documents=True
    )

    exemple_sortie = { "daily_parc" : "C’est une table journalière qui fournit les informations sur les clients ou parc clients qui appartiennent au \
            parc actif Orange. Elle présentent des informations détaillées sur l’activité des abonnés (Parc actif). Chaque ligne \
            correspond à un msisdn (identifiant de l'abonné) et inclut le nombre d’appels entrants (entrant_cnt) et sortants (sortant_cnt) \
            ainsi qu’un indicateur de régularité (regularite) et la date de dernière activité (last_active_date). Les colonnes d_prem_activ \
            et status renseignent respectivement la date de première activation et le statut de la ligne (A, G, E, D, etc .). Les colonnes \
            famprod, id_etat_ligne, d_resiliation et parc apportent des informations supplémentaires sur la famille de produits (prépayée, \
            postpayée, etc) l’état de la ligne, la date de résiliation éventuelle et le parc associé (0/1). Enfin, les colonnes year, month, et \
            day indiquent la date d’enregistrement des données. Elle est utile pour mesurer d'engagement utilisateur (via interactions \
            entrantes/sortantes, régularité) et pour analyser le comportement, les churn (via d_resiliation, statut, ancienneté), l’usage des \
            abonnés sur le réseau, etc. Elle donne  aussi l'indice de régularité d'activité du client (regularite)." }

    query = f"""
        Vous un Expert Analyste base de données et recherches documentaire.Votre but est de faire une recherche sur les caractéristiques d'une  \
        base de donnée pour trouver les tables qui permettent de calculer les KPI de certains données de base.En résumé,vous etes un expert en analyse de données avec plus de 25 ans d'expérience \
        et expert en gouvernance de données : tu documentes des bases de données métier (RAG) pour que les équipes comme le BI comprennent les \
        indicateurs de performance d'une partie des données de la base pour tirer des informations précises.Maintenant voici la question qui a 
        été posée, un échantillon de son résultat et les informations sur les KPI pertinentes pour cette question et sa réponse: \n Question : 
        {requete}, \n Echantillon du résultat : {echantillon}, \n Information sur les KPI : {info_kpi}\n.Alors votre misson est d'étudier et d'analyser  
        ses informations pour me donner en retour les tables qui permettent de mesurer les kpi du résultat de la question qui a été posée au 
        depart.Ses KPI nous permettrons de savoir 'le pourquoi' de l'évolution(augmentation ou baisse) des données.Donne moi en sortie 
        le(s) nom(s) de la ou des table(s) et leurs descriptions sous la forme d'un document JSON avec le nom ou les nom(s) de la ou le(s) 
        tables comme clé et leurs descriptions comme valeurs. \
        Les informations du RAG vous donnent le nom des tables, leurs descriptions et ses caractéristiques,pour chaque tables, vous avez le 
        nom des colonnes et la descrition de chaque colonne, de meme qu'aussi pour chaque colonne vous avez son type, quelques exemples de 
        valeurs de la colonne, le sens ou le contexte ou l'utilité de la colonne par rapport aux données. \
        Notez que les tables journalières sont mis à jour quotidiennement et les mensuelles mensuellement. Voici un exemple de sortie : \
        
        ```json
        {json.dumps(exemple_sortie, indent=2, ensure_ascii=False)}
        ```
        
    """
    resultat = qa_chain(query)

    ################# Cout du modèle
    """
    with get_openai_callback() as cb:
        resultat = qa_chain(query)
        print(f"Prompt tokens : {cb.prompt_tokens}")
        print(f"Completion tokens : {cb.completion_tokens}")
        print(f"Total tokens : {cb.total_tokens}")
        print(f"Cout (USD) : {cb.total_cost}")
    """

    #with open("info_Tab_KPI.txt", "w", encoding="utf-8") as f:
    #    f.write(resultat['result'])

    #return resultat['result'], cb.total_cost
    return resultat['result']

def Agent_KPI_req(echantillon, requete, requete_sql, user_dir) :

    agent = OpenAI(api_key=openai_key)
    #info_kpi = Agent_RAG_KPI(echantillon, requete)
    info_kpi, cout1 = Agent_RAG_KPI(echantillon, requete)
    #tables_kpi = Agent_tables_KPI(requete,echantillon,info_kpi)
    tables_kpi, cout2 = Agent_tables_KPI(requete,echantillon,info_kpi)

    path = f"{user_dir}_output_tab_KPI.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(tables_kpi)
    
    tables_kpi = Extract_tables(path)

    data = json.loads(tables_kpi) 
    list_tab = []
    path = f"{user_dir}_desc_tab_output_KPI.txt"
    for key, value in data.items():
        info_tab = {
            "nom_table" : key,
            "description" : value,
            "colonnes" : caracteristique[key]
        }
        list_tab.append(info_tab)
    tables_kpi = list_tab

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{list_tab}")


    response = agent.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Vous allez recupérer uniquement la ou les table(s) de données fournie(s) pour les KPI: {tables_kpi}, " \
                      f"pour produire une requete SQL valide qui permet de calculer les KPI avec les informations suivante {info_kpi} :" \
                      #"Et aussi le nombre de clients (ou abonnées,etc) est différent au nombre de souscription"
                      " Analyse bien la ou les tables reçue(s) pour voir est ce que l'information des KPI est dans une ou plusieurs " \
                      "colonnes. Regarde bien aussi est-ce-que vous avez reçu une ou plusieurs table(s)." \
                      "Parfois, vous pouvez recevoir plusieurs tables, et que chaque table peut répondre à la question. Dans ce cas, prenez la table qui contient " \
                      "le plus d'information.Si vous avez reçu une seule table, produit une requete SQL correcte avec cette table seulement en prenant que les colonnes "
                      "qui contiennent l'information sur les KPI. S'il y a 2 "
                      "ou plusieurs tables reçues et que la réponse se trouve sur les différents tables, analyse les colonnes de chaque table, puis étudier les dépendances "
                      "entre les tables, c'est à dire les colonnes qui permet de faire la liaison entre les tables."
                      "Après cela, produit une requete SQL correcte avec les colonnes qui contiennent l'information sur les différentes " \
                      f"tables reçues en faisant des jointures pour calculer les KPI de cette échantillon de donnée suivante issue du résultat de la \
                      question, '{requete}' : {echantillon}.Voici la requete qui a donné ses résultats : \n{requete_sql} \n.Référez vous de cette requete "
                      "pour produire une requete qui est dans la meme période, avec les memes informations, etc. Autrement dit la requete qui sera produite "
                      "doit avoir les memes conditions que celle donnée ci-dessus, et etre sur le meme période. Sélectionnez toujours les colonnes 'year' et 'month' sur la requete et utilisez les memes" \
                      " nom de colonne pour les alias (exemple : as year et as month). Sachez que le mois est toujours sous format numérique." \
                      "utilise toujours des 'Alias' par exemple 'nom_table.nom_colonne' " \
                      "pour éviter d'avoir des erreurs d'exécution provoqué par un nom de colonne.Notez que les colonnes avec comme nom 'day' représente les "
                      "jours et ont pour format numérique (yyyymmdd).Gardez cela en mémoire, il vous servira pour les questions sur les données quotidiennes. "
                       
                      "La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une requete SQL valide qui peut etre exécutée " \
                      "sur n'importe quelle version de HiveSQL sans erreur. N'utilise jamais une fonction obsolète dans la "
                      "requete. Ne met jamais de requete avec une sous requete sur la clause WHERE." \
                      "Par exemple,  WHERE year = (SELECT ....), Utilise plutot JOIN ON à place. "
                      
                      "Sur la requete SQL, ordonnez toujours le résultat du plus récents au plus anciens ou du plus grand au plus petit."
                      "Analysez la question pour voir, est ce que vous devez de faire une requete simple,d'aggrégation,de jointure ou combiné" \
                      
                      "Produit en retour une requete SQL qui est claire, structurée, correcte et exécutable sous Hive ,prenez seulement "
                      #"" \
                          #" et sélectionnez tout les colonnes qui ont été utilisées sur la requete" \
                        "tout les colonnes qui contiennent l'information des KPI.Utilisez toujours des alias dans la requete et ajouter les colonnes year " \
                        "et month dans la sélection.La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une " \
                        "requete SQL valide sous Hive. Voici quelques genres d'exemples de requete SQL pour la sortie : "
                        "SELECT rcm.year AS year, rcm.month AS month, rcm.ca_data AS chiffre_affaire FROM reporting_ca_monthly rcm WHERE rcm.year = YEAR(CURRENT_DATE())) "
                        "GROUP BY rcm.year, rcm.month ORDER BY rcm.year DESC,rcm.month DESC;"
                        "SELECT dp.year AS year,dp.month AS month,COUNT(DISTINCT dp.msisdn) AS active_orange_subscribers FROM daily_parc dp WHERE dp.year = 2025 "
                        "AND dp.month = 1 AND dp.parc = 1 GROUP BY dp.year, dp.month;"
                        "SELECT T1.year AS year,T1.month AS month,SUM(T1.sortant_gratuite),SUM(T1.sortant_acquisition) FROM daily_delta_parc AS T1 JOIN "
                        "(SELECT MAX(year) AS max_year FROM daily_delta_parc) AS T2 ON T1.year = T2.max_year GROUP BY " \
                        "T1.year, T1.month ORDER BY T1.year DESC, T1.month DESC; "
                      
                    }
                ]
            }
        ],
        #max_tokens=200,
        #max_completion_tokens=600
    )
    responses = response.choices[0].message.content

    ############### cout modèle
    
    responses = response.choices[0].message.content
    print(f"{response}\n\n")
    # calcule du cout pour gpt-4o
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    #cout en dollars
    cout3 = (prompt_tokens * 0.0025 / 1000) + (completion_tokens * 0.01 / 1000)

    cost = cout2 + cout1 + cout3
    print(f"Cout total : {cost} $")

    path = f"{user_dir}_kip.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(responses)
    
    sql_query = Extract_sql(path)

    return sql_query, info_kpi, cost
    #return sql_query, info_kpi
    


##################################################
#           Requete avec KPI
##################################################

def Agent_KPI_req_gemini(echantillon, requete, requete_sql, user_dir):
    # Configuration Gemini
    genai.configure(api_key = api_k)

    info_kpi = Agent_RAG_KPI(echantillon, requete)
    print(info_kpi)
    #tables_kpi = Agent_tables_KPI(requete,echantillon,info_kpi)
    tables_kpi = Agent_tables_KPI(requete,echantillon,info_kpi)
    print(tables_kpi)

    path = f"{user_dir}_output_tab_KPI.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(tables_kpi)
    
    tables_kpi = Extract_tables(path)

    data = json.loads(tables_kpi) 
    list_tab = []
    path = f"{user_dir}_desc_tab_output_KPI.txt"
    for key, value in data.items():
        info_tab = {
            "nom_table" : key,
            "description" : value,
            "colonnes" : caracteristique[key]
        }
        list_tab.append(info_tab)
    tables_kpi = list_tab

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{list_tab}")


    #info_kpi, c1 = Agent_RAG_KPI(echantillon, requete)
    #tables_kpi, c2 = Agent_tables_KPI(requete,echantillon,info_kpi)
    #print(f"{info_kpi} \n\n")
    #print(f"{tables_kpi} \n\n")

    # Sélectionner le modèle Gemini multimodal
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    #Prompt
    prompt = f"""
    Vous allez recupérer uniquement la ou les table(s) de données fournie(s) pour les KPI: {tables_kpi},  \
    pour produire une requete SQL valide qui permet de calculer les KPI avec les informations suivante : {info_kpi} \
    
    Analyse bien la ou les tables reçue(s) pour voir est ce que l'information des KPI est dans une ou plusieurs \
    colonnes. Regarde bien aussi est-ce-que vous avez reçu une ou plusieurs table(s). \
    Parfois, vous pouvez recevoir plusieurs tables, et que chaque table peut répondre à la question. Dans ce cas, prenez la table qui contient \
    le plus d'information.Si vous avez reçu une seule table, produit une requete SQL correcte avec cette table seulement en prenant que les colonnes \
    qui contiennent l'information sur les KPI. S'il y a 2 \
    ou plusieurs tables reçues et que la réponse se trouve sur les différents tables, analyse les colonnes de chaque table, puis étudier les dépendances \
    entre les tables, c'est à dire les colonnes qui permet de faire la liaison entre les tables. \
    Après cela, produit une requete SQL correcte avec les colonnes qui contiennent l'information sur les différentes \
    tables reçues en faisant des jointures pour calculer les KPI de cette échantillon de donnée suivante issue du résultat de la \
    question, '{requete}' : {echantillon}.Voici la requete qui a donné ses résultats : \n{requete_sql} \n.Référez vous de cette requete \
    pour produire une requete qui est dans la meme période, avec les memes informations, etc. Autrement dit, la requete qui sera produite \
    doit avoir les memes conditions que celle donnée ci-dessus, et etre sur le meme période. Sélectionnez toujours les colonnes 'year' et 'month' sur la requete et utilisez les memes \
    nom de colonne pour les alias (exemple : as year et as month). Sachez que le mois est toujours sous format numérique. \
    utilise toujours des 'Alias' par exemple 'nom_table.nom_colonne' \
    pour éviter d'avoir des erreurs d'exécution provoqué par un nom de colonne.Notez que les colonnes avec comme nom 'day' représente les \
    jours et ont pour format numérique (yyyymmdd).Gardez cela en mémoire, il vous servira pour les questions sur les données quotidiennes. \
    
    La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une requete SQL valide qui peut etre exécutée \
    sur n'importe quelle version de HiveSQL sans erreur. N'utilise jamais une fonction obsolète dans la \
    requete. Ne met jamais de requete avec une sous requete sur la clause WHERE. \
    Par exemple,  WHERE year = (SELECT ....), Utilise plutot JOIN ON à place. \
    
    Sur la requete SQL, ordonnez toujours le résultat du plus récents au plus anciens ou du plus grand au plus petit. \
    Analysez la question pour voir, est ce que vous devez de faire une requete simple,d'aggrégation,de jointure ou combiné .\
    
    Produit en retour une requete SQL qui est claire, structurée, correcte et exécutable sous Hive, prenez uniquement 

    les colonnes qui contiennent l'information sur les KPI.Utilisez toujours des alias dans la requete et ajouter les colonnes year \
    et month dans la sélection.La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une \
    requete SQL valide sous Hive. Voici quelques genres d'exemples de requete SQL pour la sortie : \
    SELECT rcm.year AS year, rcm.month AS month, rcm.ca_data AS chiffre_affaire FROM reporting_ca_monthly rcm WHERE rcm.year = YEAR(CURRENT_DATE())) \
    GROUP BY rcm.year, rcm.month ORDER BY rcm.year DESC,rcm.month DESC; \
    SELECT dp.year AS year,dp.month AS month,COUNT(DISTINCT dp.msisdn) AS active_orange_subscribers FROM daily_parc dp WHERE dp.year = 2025 \
    AND dp.month = 1 AND dp.parc = 1 GROUP BY dp.year, dp.month; \
    SELECT T1.year AS year,T1.month AS month,SUM(T1.sortant_gratuite),SUM(T1.sortant_acquisition) FROM daily_delta_parc AS T1 JOIN \
    (SELECT MAX(year) AS max_year FROM daily_delta_parc) AS T2 ON T1.year = T2.max_year GROUP BY \
    T1.year, T1.month ORDER BY T1.year DESC, T1.month DESC;
    """

    # Appel Gemini (texte + image)
    response = model.generate_content(
        prompt,
        #generation_config={"max_output_tokens": 600}
    )

    path = f"{user_dir}_kip.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    sql_query = Extract_sql(path)
    print(f"\n {sql_query}\n")

    return sql_query, info_kpi
    #return response.text

###################################################################################
#               Analyse et recommande                                              #
def analyse_recommand_gemini(api_key = api_k, request=None, nb_imgD=1,nb_imgKPI=1, info_kpi=None):
    # Configuration Gemini
    genai.configure(api_key = api_key)

    # Charger les images (même logique que ton code OpenAI)
    image_paths_dash = [f"Image/{i}.png" for i in range(nb_imgD)]

    # Créer une grille (si tu veux conserver ta fonction combine_images_grid)
    
    dash = combine_images_grid(image_paths_dash)
    #dash = 'Dashboard.png'

    image_paths_kpi = [f"img_kpi/{i}.png" for i in range(nb_imgKPI)]

    # Créer une grille (si tu veux conserver ta fonction combine_images_grid)
    kpi = combine_images_grid(image_paths_kpi, output_path= "img_kpi/dash.png")
    
    # Charger l’image du dash
    image_dash = Image.open(dash)
    image_kpi = Image.open(kpi)

    # Sélectionner le modèle Gemini multimodal
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    # Prompt 
    prompt = f"""
    Voici deux visuel : {dash} , {kpi}. Le premier visuel est le dash à analyser et le second permet de faire l'analyse en utilisant les kpi.Autrement dit, 
    le second image permet de faire l'analyse du premier image en cherchant le pourquoi il y a une évolution ou variation sur les données 
    du premier image.C'est à dire que le second dash contient les kpi pour expliquer les données du premier dash.Voici quelques informations
    sur les KPI :\n {info_kpi} \n
    Faites une :
    Analyse globale : explique les tendances du premier visuel en se basant ou référant sur le deuxième en 4 points max.  
    Recommandations stratégiques (5 points max) : actions concrètes (campagne, tarification, ciblage, etc).  
    Exemple : 
    Entre février et avril, le parc client actif connaît une hausse significative de +18 %. Cette croissance est principalement due 
    aux nouveaux entrants (nombre élevé de reactives et entrant_cnt). A partir de mai, la croissance ralentit légèrement, indiquant une 
    stabilisation ou un début de churn. Dakar concentre 45% du chiffre d’affaires total (toujours en FCFA), particulièrement sur les services data et 
    recharges digitales (Wave, SEDDO). Les zones rurales montrent une faible adoption de la 4G, mais une forte consommation voix.

    Maintenir la dynamique en lançant une campagne ciblée sur les nouveaux entrants pour les fidéliser (ex : offre bienvenue, bonus de 
    recharge).Mettre en place un modèle prédictif de churn pour identifier les clients à risque et leur proposer des offres 
    personnalisées. Développer davantage les services premium pour augmenter l'ARPU,Optimiser les infrastructures pour anticiper la 
    surcharge réseau à Dakar. Dans les zones rural, il faut Promouvoir des offres mixtes voix + petit volume data pour stimuler la 
    migration vers la 4G.

    Les analyses sont faites sur le premier visuel et leurs causes sont sur le second visuel. Evitez les fautes sur les chiffres qui 
    sont sur le Dashboard. Donnez les éléments exact du Dashboard. Parfois, vous pouvez rencontrer des exemples de valeurs comme 15K, 
    2M, 7B, etc. le 'K' représente les millièmes, le 'M' les millions et le 'B' les milliards.Si vous voyez des graphes qui parlent d'argent 
    (chiffres d'affaires (ca), revenu, ect.), sachez que les données sont en FCFA.
    
    Notez que msisdn représente l'identifiants des clients.
    Contexte : vous êtes un **analyste data senior avec une forte expertise business et marketing à la SONATEL**.
    Donnez directement l'analyse et les recommandations sans évoquer "En tant que Analyste de la SONATEL ..." ou ses genres de phrases.
    Passez directement à l'action. Et aussi sur la réponse, je ne veux pas des termes comme sur le visuel 1 ou visuel 2 ou du genres.
    """

    # Appel Gemini (texte + image)
    response = model.generate_content(
        [
            prompt,
            image_dash,
            image_kpi
        ],
        #generation_config={"max_output_tokens": 600}
    )

    return response.text




####################################################################################
#                           RAG avec Google                                         #
####################################################################################


def Agent_analyst_RAG_Gemini(requete):

    # 1. Créer les documents (comme dans ton code)
    #documents = [
    #    Document(
    #        page_content=f"Table : {key}\nDescription : {value}\nCaractéristiques : {data[key]} \n\n"
    #    )
    #    for key, value in tab.items()
    #]

    # 2. Découper en chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # 3. Créer les embeddings avec Gemini
    # Assure-toi que ta clé est bien définie :
    # setx GOOGLE_API_KEY "ta_clé_API"
    #if not os.getenv("GOOGLE_API_KEY"):
    #    raise ValueError("La variable d'environnement GOOGLE_API_KEY n'est pas définie")

    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
    #                                          google_api_key = os.environ['GOOGLE_API_KEY'])
    #embeddings.embed_query()
    # Construire la base vectorielle
    db = FAISS.from_documents(docs, embeddings)

    # Récupérateur
    retriever = db.as_retriever()

    # Chaîne RAG avec Gemini comme LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    exemple_sortie = {
        "daily_parc_maxit" : "C’est une table journalière qui fournit les informations sur les clients ou le parc clients qui ont eu à faire  \
                                une activité sur l’application mobile MAXIT. Elle contient les données d’usage de l’application MAXIT pour \
                                chaque abonné avec des informations sur la fréquence, la récence et la périodicité de l’utilisation. Chaque \
                                ligne correspond à un MSISDN (identifiant unique d’un abonné) et inclut la date de dernière activité (last_active_date) \
                                ainsi que le nombre de jours distincts d’activité (nb_j_distinct). Les colonnes year, month et day précisent la période \
                                d’utilisation de l’application. Elle permet d’identifier les clients actifs ou inactifs de l’application mobile MAXIT ou \
                                en déclin d’usage, de segmenter les utilisateurs MAXIT selon la fréquence d’usage (nb_j_distinct) et la récence \
                                (last_active_date). Elle est conçue pour analyser l’engagement des clients sur l’application mobile MAXIT."
    }
    
    # Tester une question
    query = "Vous un Expert Analyste base de données et recherches documentaire.Votre but est de rechercher sur les caractéristiques d'une  \
        base de donnée les tables pour répondre à une question métier.En résumé,vous etes un expert en analyse de données avec plus de 25 ans d'expérience \
        et expert en gouvernance de données : tu documentes des bases  de données métier pour que les équipes comme le BI comprennent chaque \
        champ de la base pour tirer des informations précises.Maintenant Quelle(s) est ou sont la ou les table(s) qui permet(tent) de répondre à la question  \
        suivante sur le RAG:" f"{requete}.""Donne moi en sortie un fichier JSON avec les noms des tables comme clé et leurs descriptions décrit sur le RAG comme valeurs. \
        Les informations du RAG vous donnent le nom des tables, leurs descriptions et ses caractéristiques,pour chaque tables, vous avez le nom des colonnes et la descrition de chaque \
        colonne, de meme qu'aussi pour chaque colonne vous avez son type, quelques exemples de valeurs de la colonne, le sens ou le contexte ou l'utilité de la colonne par rapport au donnée. \
        \
        Autrement dit, Tu as accès à un RAG qui contient la documentation d’une base de données : le nom des tables, leur description, la liste des colonnes avec leurs types et leur signification. \
        Lorsque tu réponds à une question de l’utilisateur,ne propose que les tables réellement présentes dans le catalogue.Retourne uniquement  \
        les tables avec leurs noms réelles ou exactes qui ont été définit dans le catalogue pour répondre à la question, sans ajouter d’informations fictives. \
        \
        Notez bien que les tables commençant comme nom 'daily' contiennent des données journalières des clients celles commençant par 'monthly' \
        répertorient les données mensuelles des clients et celles commençant par 'reporting' font une reporting des clients. \
        Si vous rencontrez 'Sargal' dans la base sachez que c'est un programme de fidélité lancé par l'entreprise pour récompenser  \
        ses clients en fonction du niveau de consommation et d'engagement sur les services proposés. Autrement dit, les clients accumulent \
        des points automatiquement à chaque rechargement de crédit. Plus ils consomment, plus ils cumulent de points.\
        Notez qu'en télécommunications, le terme 'parc' désigne l’ensemble des clients actifs ou abonnés à un service.Sachez que les questions \
        sur le parc actif orange(ou parc actif ou parc orange), vous trouverez les réponses de ses questions sur les tables 'daily' plus \
        précisément 'daily_parc'.Autrement dit, pour les questions sur parc actif orange (sans la précision des clients digitaux), consultez la table 'daily_parc'. \
        Notez qu'aussi Maxit est une application mobile de l'entreprise qui offre plusieurs services (Achat, transfert d'argent, paiement, etc). \
        \
        Avec toute ses informations données sur la base, votre mission est de rechercher la ou les tables de données qui peut ou peuvent fournir une réponse à cette \
        question :" f"{requete}"". Faite une analyse approfondie des caractéristiques ou documents de la base en étudiant le role de chaque tables et colonnes, \
        les relations qui peuvent exister entre les différents tables. Après, vous devrez répondre à cette question métier suivante dans la base de donnée :" f"{requete}"". \
        Sur les questions concernant les parc des takers, si maxit ou digital n'est pas renseigné, vous ne devez pas interroger les tables suffixées MAXIT.\
        Derrière, vous pouvez créer des index pour les tables (les tables concernant les offre, souscriptions, sargal, chiffre \
        d'affaire, recharges, data, etc). Cela vous facilitera la recherche de la réponse. \
        Et aussi notez que le nombre de clients (ou abonnées ou bénéficiaires, etc) est différent au nombre de souscription. Un client peut souscrire dans une \
        offre une ou plusieurs fois. L'identifiant des clients est donné par la colonne 'MSISDN' dans les tables daily et monthly. Alors, le nombre de client doit toujours etre  \
        déterminé par une table contenant cette colonne 'MSISDN'. Notez qu'abonnés, bénéficaires, etc font référence aux clients. \
        Vous allez prendre les résultats de votre analyse sur les caractérisques de la base pour determiner la ou les table(s) nécessaire(s) \
        pour répondre à cette question métier. Vous allez étudier le but de la question métier qui a \
        été posée. C'est à dire quelles sont les informations que l'utilisateur veut savoir par rapport à cette question : \
        "f"{requete}"". Si la réponse peut etre récupérer dans une seule table, en sortie vous donneriez seulement la table et sa description en JSON indiqué ci-dessus. \
        Si la réponse est dans 2 ou plusieurs tables, Vous donnerez tout les tables concernées \
        pour répondre à la question. Analysez la question pour voir, est ce que vous devez de faire une requete \
        simple , d'aggrégation, de jointure ou combiné. Notez que 'OM' signifie 'Orange Money'. Ne donne jamais un " \
        "nom de table qui n'existe pas dans le catalogue de données fournies dans le RAG. " \
        "Notez que la jointure entre les tables peut etre fait entre des colonnes qui ne représente pas l'identifiant " \
        "des clients. Faites bien la différence entre les tables parc, de l'application Maxit, de souscriptions des offres, du trafic réseau (2G/3G/4G/5G) ou de " \
        "consommation de donnée (data) mobile, du programme de fidélité Sargal, de voix, recharges ou chiffre d'affaires (ca), pour les trafic à " \
        "l'international (ca, durée, parc, etc à l'international), etc. Analyse bien la question posée avant de " \
        "donner la ou les tables de sortie. Si vous voyez une question qui parle de paiement (par exemple paiement senelec, sen'eau, " \
        "etc) consultez la table 'daily_maxit' qui contient les infos quotidiennes sur les services de l'application maxit avec sa colonne 'service'." \
        \
        "Les tables commençant par 'daily' et 'monthly' contiennent chacune une colonne 'MSISDN' " \
        "qui représente l'identifiant des clients ou abonnés. Sachez que les tables commençant par 'reporting' n'ont pas une colonne " \
        "pour l'identifiant des clients. Pour les questions sur Maxit, consultez les tables Maxit, pour les questions sur Sargal " \
        "allez sur les tables Sargal, pour les questions sur la data, consultez les tables data, pour les questions sur l'international, " \
        "allez sur les tables internationales, pour les questions sur la souscription des offres, consultez les tables de souscription etc. " \
        "Notez que les tables de données avec comme suffixe 'international' (monthly_international, reporting_monthly_international) " \
        "sur le nom de la table contiennent des infos sur les trafics à l'international (le parc à l'international (parc_pass, parc_international, parc_payg), le chiffre d'affaires à " \
        "l'international (ca_pass, ca_payg), la durée des communications à l'international (ci_pass_dur, duration_mn_payg), etc)" \
        \
        "NB : Notez que la table daily_oss_5g contiennent les informations sur les cellules et leurs trafics (nom cellule, trafic moyen des utilisateurs, etc). daily_clients et " \
        "monthly_clients contiennent les informations quotidiennes et mensuelles des clients (identifiant, infos géographiques et démographiques, segment d'appartenance,etc.). daily_clients_digitaux donne les infos des clients qui utilisent les " \
        "plateformes digitaux.daily_conso fournit le montant des consommations des appels (ou voix) internationaux de chaque client " \
        "(ca_voix_international).monthly_international et reporting_monthly_international donnent le montant ou CA des consommations des communications internationaux pour les forfaits appelé ca_pass, pour les usages hors forfait appelé ca_payg et " \
        "leur durée totale(duration_mn_payg, duration_mn_pass), et leur parc (parc_pass, parc_payg, parc_international). Autrement dit, les deux tables donnent le CA payg et CA pass, durée des pass et des payg, le parc payg et pass, mais notez que " \
        "c'est des infos sur les clients international. daily_data donne le volume de trafic de la consommation des données mobiles sur le réseau 2G, 3G, 4G et 5G en " \
        "MégaOctets(Mo) tandisque monthly_data ajoute la commune et les segments (marché et recharges) d'appartenance et la formule tarifaire " \
        "souscrite par le client.daily_delta_parc et reporting_parc_monthly donnent les infos sur la sortie et le parc des clients (sortant(0/1), reactive(0/1),parc(0/1),nouvelles " \
        "arrivées(0/1), etc) alors que monthly_sortant ajoute la localité de référence du client (région, département, commune, zone_drv, " \
        "nom_cellule, etc) tandisque daily_parc fournit les infos globlal du parc des clients (nombre d'entrant(entrant_cnt), de sortant (sortant_cnt) et de régularité (regularite), le parc (0/1), etc)." \
        "daily_habit_5g_user donne la consommation Data mobile des Clients (volume_go) sur le réseau 5G par type d'application.daily_heavy_user_4g " \
        "donne le suivi de la consommation 4G des Heavy user. daily_infos_bts donne les informations détaillées sur les sites physiques et leurs " \
        "cellules radios correspondant (NOM_SITE, NOM_CELLULE, CELLID, ID, TYPE_CELLULE, la REGION, le DEPARTEMENT, etc). daily_infos_otarie et monthly_terminaux donnent " \
        "le parc des clients avec leurs appareils mobiles et leurs marques utilisés, le parc d'utilisation du reseau (5G, 4G, 3G, 2G) et les " \
        "volumes de données consommées pour chaque réseau en MégaOctets (Mo). daily_localisation_5g fournit les " \
        "infos géographiques des clients. daily_maxit et monthly_maxit donnent les clients qui se sont connectés sur l'application mobile " \
        "MAXIT avec le service utilisé tandisque reporting_daily_maxit, reporting_monthly_maxit, reporting_daily_parc_maxit et reporting_usage_maxit font le reporting des services et parcs de l'application mobile Maxit." \
        "daily_parc_recharges, daily_parc_pass, daily_parc_maxit, daily_parc_maxit_new, daily_parc_illimix, daily_parc_illiflex, " \
        "daily_parc_data_4g, daily_parc_data et daily_parc_sargal fournissent les informations sur les clients ou le parc client sur la dernière date d'activation " \
        "(last_active_date) ainsi que la fréquence (nb_j_distinct) pour les recharges, les pass, d'utilisation maxit, d'achat illimix et illiflex et data " \
        " du trafic le réseau 4G, de data trafiqué le réseau global et de partciper au programme Sargal. daily_recharges, monthly_recharges et reporting_ca_monthly donne les " \
        "chiffres d'affaires total (ca_recharge) ou générés pour tout les recharges (ca_recharges) sur les différents canaux ainsi que par canal (par OM " \
        "(ca_pass_glob_om_jour, ca_credit_om), par Wave (ca_wave), par Seddo(ca_seddo), par cartes (ca_cartes), à l'international (ca_iah), " \
        "et par self_top_up).monthly_sargal, daily_sargal, reporting_sargal_echangeurs_mon, reporting_sargal_gift_daily, " \
        "reporting_sargal_gift_monthly et reporting_sargal_inscrits donne les infos sur la participation du programme de fidélité Sargal. " \
        "monthly_souscription et daily_souscription fournissent des informations ou le CA des souscriptions (ca_data et ca_voix) des catégories ou type d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, etc) et " \
        "et de l'offre souscrite (ILLIMIX JOUR 500F, PASS 2.5GO,MIXEL 690F, illiflex mois, etc) et permettent aussi de déterminer les clients qui utilisent du data ou voix ou les deux à la fois. reporting_ca_data_monthly fait le reporting du Chiffre d'affaire data (ca_data) mensuelle. reporting_daily_offer, " \
        "reporting_offer_monthly et reporting_souscription_monthly font le le reporting sur les souscriptions des offres (sous_mnt,ca_data, ca_voix, ca_sous_HT) suivant le " \
        "types de souscription et le type d'offre souscrit. reporting_monthly_terminaux fait le reporting des Data_user ou No Data_user avec la colonne data_status " \
        "et leur utilisation de smartphone ou pas. daily_sva, monthly_sva et reporting_monthly_sva donne le montant de la souscription et le parc des " \
        "services à valeur ajouté.daily_voix, monthly_voix et reporting_monthly_voix fournissent le volume ou la durée des appels sortants des abonnées " \
        "sur les différents opérateurs téléphoniques (Orange, Free, Expresso, ProMobile, etc) ainsi leurs parcs clients correspondant. reporting_5g_daily regroupe les sites techniques " \
        "avec une série de KPI quotidiens (Key Performance Indicators) et leur volumes de données consommées liés à l’usage du réseau 5G. reporting_daily_parc fait le reporting quotidienne " \
        "du comportement (actifs ou inactifs) ou parc des abonnées sur les 90 derniers jours. reporting_daily_ca_pag_xarre, " \
        "reporting_daily_data_xarre, reporting_daily_trafic_xarre et reporting_recharge_daily_xarre " \
        "donnent le chiffres d'affaires généré des voix et SMS en mode PAYG (Pay-As-You-Go) pour les formules commerciales XARRE, " \
        "le trafic Internet et le parc d’abonnés par technologie réseau (2G, 3G, 4G, 5G) ventilées par formule commerciale XARRE, le volume " \
        "de trafic sortant et les parcs d’utilisateurs par type de réseau et d’opérateur (Orange, Expresso, Free, ProMobile, etc) sur les " \
        "offres XARRE, le chiffre d'affaires généré par les recharges des offres XARRE ainsi que du parc actif pour chaque formule tarifaire " \
        "XARRE. reporting_xarre_offer_nrt donne le parc, le ca et le volume des souscriptions aux offres XARRE par type d'offres, par offre, par " \
        "segment, par tranches d'heure de souscriptions (ex : 00h-01h, 13h-14h, etc). Donnez en sortie les tables les plus pertinantes pour répondre à la question :"f" {requete}." \
        \
        "Sur les souscriptions des offres, Notez que nous avons des catégories ou types d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, NC) avec leur " \
        "formule tarifaire ou commerciale (JAMONO NEW S'COOL, JAMONO ALLO, JAMONO PRO, JAMONO MAX, AUTRES) et leurs segments recharges (Mass-Market, High, Middle, S0, super high) " \
        "et marché (JEUNES, ENTRANT, KIRENE AVEC ORANGE, AUTRES,MILIEU DE MARCHE,HAUT DE MARCHE, TRES HAUT DE MARCHE ) correspondant. Chaque type ou catégorie d'offres contient " \
        "plusieurs offres de souscriptions et formule et catégorie d'offres sont différentes. Notez qu'aussi les offres data font références aux offres de type PASS INTERNET." \
        \
        "Si vous recevez une question sur le chiffre d'affaire tout court (par exemple : Donne moi le Ca de 2025, quel est l'évolution du CA par segment, etc) sans que l'utilisateur précise le chiffre " \
        "d'affaire, sachez que l'utilisateur veut simplement le chiffre d'affaire total qui n'est rien d'autre que le chiffre d'affaire des recharges " \
        "(ca recharges). Le chiffre d'affaire seulement fait référence au chiffres d'affaires des recharges(ca_recharges) ou CA total. Pour les questions " \
        "sur la segmentation, ne choisissez jamais les tables de reporting car elle ne sont pas adapté, utilise plutot les tables monthly (mensuelles) " \
        "ou daily (quotidiennes)" \
        \
        "Evitez les erreurs sur les noms de tables que vous fournissez en sortie, cela repercutera négativement sur l'exécution " \
        "de la requete qui sera produite. Réponds uniquement avec les informations issues des documents fournits sans inventer ou modifier un nom de " \
        "table. " \
        "Voici un exemple de sortie : " \
        \
        f"```json \
        {json.dumps(exemple_sortie, indent=2, ensure_ascii=False)} \
        ```" \

    result = qa_chain(query)
    print("OK")

    return result["result"]

def Agent_info_supp_Gem(result_table, requete, user_dir) :
    data_supp = extract_tableDescription("info_supp.xlsx")
    document = [
        #Document(page_content= dict_to_text(dataa[item]))
        Document(page_content=dict_to_text(item))

        #Document(page_content= dataa[item])
        #Document(page_content= str(dataa[item]))
        #metadata={"table": item}
        for item in data_supp
        
    ]

    #Découper le texte en chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap = 50)
    #docs = text_splitter.split_documents(documents)
    docs = text_splitter.split_documents(document)

    #les embeddings et la base vectorielle
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    #C RAG
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = api_k),
        retriever=retriever,
        return_source_documents=True
    )

    #result_table = Agent_analyst_RAG()
    print(f"\n\n{result_table}\n\n")

    query = "Vous étudiez les colonnes des différents tables fourni :"f"{result_table}.Vous allez regarder est ce qu'il  \
        y a une colonne ou des colonnes qui sont dans le RAG (colonne : valeurs uniques ou présence  \
        du nom partiel de colonne : valeurs uniques).Le RAG contient des informations de plusieurs colonnes catégorielles.  \
        Autrement dit, vous avez des noms de colonnes avec leurs valeurs uniques (exemple : SEGMENT_RECHARGE : S0, Mass-Market, \
        Middle, High, Super High) ou des noms (groupe de mots) qui représentent plusieurs colonnes avec leurs valeurs uniques (exemple : \
        Pour les colonnes réprésentant les regions : TAMBACOUNDA, ZIGUINCHOR, DIOURBEL, DAKAR, THIES,SAINT LOUIS, etc) \
        Si après votre étude, vous trouvez une ou des colonnes dans le RAG,alors vous allez étudiez la \
        question pour voir la ou les valeurs unique(s) de la ou les colonnes qui permet(tent) de répondre à la \
        question : "f"{requete}, puis vous allez récupérerer la ou les valeurs et les donnez en sortie pour que un autre agent l'utilise \
        sur sa production de requete SQL au niveau de la clause WHERE.Ce qui permettra à l'agent de produire une requete \
        avec un clause WHERE qui s'adapte avec les informations de la base de données.Par exemple, pour une question comme \
        'Quelle est le montant total des paiements SENELEC', pour \
        répondre à cette question, vous étudiez la colonne qui permet de répondre à cette question, et sur ses éléments, \
        quelle est ou sont l'élément(s) unique(s) de la colonne qui permet de répondre à la question.Dans cet exemple, ça correspond à la \
        la colonne 'service' avec sa valeur unique 'BY_SENELEC' qui sera utilisée sur la requete qui sera produite par  \
        un agent sur sa clause WHERE.Un autre exemple,'donne moi le parc des illimix jour de février 2025',dans cet exemple vous allez  \
        recupérer l'élément unique correspondant au illimix jour dans le nom des offres et le donner en sortie (ici c'est 'illimix jour 500F'). Il en est de meme  \
        pour tout autres questions de ce genres.Par contre, si vous avez une question où vous avez besoins de tout les éléments de la \
        colonnes, dans ce cas, ce n'est pas la peine de donner en sortie des informations supplémentaires.Par exemple,'Donne moi \
        le chiffre d'affaires des offres en 2025',sur cet exemple, vous avez besoin de toute la colonne qui représente les offres, alors \
        ce n'est pas la peine d'envoyer en sortie les valeurs uniques des offres.Un autre exemple, 'Donne moi le revenu HT des segments \
        marchés',pour répondre à cette question, vous avez besoin de tout les segments marché, alors ce n'est pas la \
        peine de retourner les valeurs uniques de la colonne qui représente les segments marchés.L'agent 'Producteur de requete SQL' \
        se chargera de la gestion de ses valeurs uniques.Il en est de meme pour toute ses genres de questions.En résumé,vous allez retourner \
        uniquement des valeurs lorsque la question qui a été posée spécifie une ou quelques valeur(s) unique(s) d'un ou des colonne(s). \
        Notez que le RAG contient quelques variables catégorielles et leur valeurs uniques.Votre mission est de recupérer \
        la ou les valeur(s) unique(s) nécessaire(s) de la ou les colonne(s) pour répondre à la question qui a été posée.Notez que 'OM' \
        signifie 'Orange Money'.Si la question ne nécessite pas d'info supplémentaire sur le RAG, donnez en retour qu'elle n'a pas besoin d'infos sup. \
        Donner en sortie une ou des colonnes (s'il y en a dans le RAG) avec quelque(s) de sa ou ses valeurs uniques qui permet(tent) \
        de répondre à la question : {requete}.Par exemple, [ service : 'By_SENELEC' pour les paiement SENELEC, segment_marche : ['JEUNES', 'ENTRANT'] pour une question sur le marché des JEUNES et des ENTRANT,  \
        ca_cr_commune_90 : 'DAKAR' pour une question sur la commune de Dakar sur les 90 derniers jours ,offer_name = [ 'PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO'], etc pour une question\
        les souscriptions aux offres 'PASS 2,5GO', 'ILLIMIX JOUR 500F', 'PASS 150 MO', etc] ou bien pas besoins d'informations \
        supplémentaires s'il n'en a pas. Si la question n'est pas spécifique à une valeur d'une variable, cela veut dire qu'on a \
        besoin de tout les infos de la variables, dans ce cas ce n'est pas la peine de retourner des valeurs uniques de la variables"

    result = qa_chain(query)
    data = f"{result_table}\n\n{result['result']}"

    path = f"{user_dir}_infos_RAG.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

    print("OK")

    #print("Réponse :", result["result"])
    #print("Source :", result["source_documents"][0])
    return result["result"]

def Agent_SQL_Gem (requete, user_dir):
    
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDRMK4upPL-nEIXd8Nurjgcy3IZyTYoGK0"
    genai.configure(api_key = os.environ["GOOGLE_API_KEY"])
    
    result_table = Agent_analyst_RAG_Gemini(requete)

    print(result_table)

    path = f"{user_dir}_output_tab.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(result_table)
    
    result_table = Extract_tables(path)

    data = json.loads(result_table) 
    list_tab = []
    path = f"{user_dir}_desc_tab_output.txt"
    for key, value in data.items():
        info_tab = {
            "nom_table" : key,
            "description" : value,
            "colonnes" : caracteristique[key]
        }
        list_tab.append(info_tab)
    result_table = list_tab

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{list_tab}")


    info_sup = Agent_info_supp_Gem(result_table, requete, user_dir)
    print(info_sup)


    # Sélectionner le modèle Gemini multimodal
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"Vous allez recupérer uniquement la ou les table(s) de données suivante(s): {result_table}, " \
            "et les informations supplémentaires (s'il y en a) :"f" {info_sup} " \
            "pour produire une requete SQL valide qui permet de répondre à cette question métier :"f" {requete}.Ses infos supplémentaires" \
            "sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut" \
            "etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous" \
            "permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une requete SQL " \
            "valide et correcte par rapport à nos données.Sur la clause WHERE de la requete, utilisez la ou les valeur(s) donnée " \
            "par les infos supplémentaire si la question posée spécifie une ou des éléments d'une ou des colonnes fournies (Par exemple : donne " \
            "moi le CA des clients Mass-market? donne moi le parc des takers de l'offre PASS NUIT 5 GO?, etc). Si la question concerne tout les " \
            "éléments d'une colonne fournie dans les infos supplémentaires, ne met jamais cette colonne sur la clause WHERE car elle se sera unutile.  " \
            "Notez qu'à chaque fois que vous recevez une question qui parle de 'data',sachez que cela fait référence à internet. Autrement " \
            "dit, dans les données,'data' veut dire 'internet' ou 'pass internet' (Par exemple : offres data signifie offres de type 'pass internet')." \
            "Abonnés, bénéficaires, etc font référence aux clients.'OM' signifie 'Orange Money'." \
            " Analyse bien la ou les tables reçue(s) pour voir est ce que l'information de la question posée est dans une ou plusieurs " \
            "colonnes. Regarde bien aussi est-ce-que vous avez reçu une ou plusieurs table(s)." \
            "Parfois, vous pouvez recevoir plusieurs tables, et que chaque table peut répondre à la question posée. Dans ce cas, prenez " \
            "la table qui donne le plus d'information pour répondre à la question.Si vous avez reçu " \
            "une seule table, produit une requete SQL correcte avec cette table seulement en prenant que les colonnes qui contiennent" \
            "l'information de la réponse sur la table. S'il y a 2 " \
            "ou plusieurs tables reçues et que la réponse se trouve sur les différents tables, analyse les colonnes de chaque table, puis étudier les dépendances " \
            "entre les tables, c'est à dire les colonnes qui permet de faire la liaison entre les tables." \
            "Après cela, produit une requete SQL correcte avec les colonnes qui contiennent l'information sur les différentes " \
            "tables reçues en faisant des jointures pour répondre à cette question métier : " \
            f"{requete}"". Si la question est trop vaste, (par exemple 'Je veux le chiffre d'affaire', 'Quelle formule tarifaire " \
            "génère le plus de trafic 4G sur le mois d'avril', ect) vous essayerez toujours de répondre en donnant une " \
            "requete SQL qui donne les informations les plus récentes. Dans cet exemple, vous donneriez le chiffre d'affaire ou " \
            "formule tarifaire de l'année en cours (YEAR(CURRENT_DATE()))), Si vous avez une requete qui nécessite une condition et si la condition " \
            "doit se faire avec des caractères ou chaines de caractères (Ex : Région de Dakar), sur " \
            "la clause WHERE, Utilise LIKE plutot égal (=) par exemple '%Dakar%', '%DAKAR' ou '%kar%', etc .Référez vous aussi sur les infos supplémentaires données ci-dessus " \
            "Exemple de question : Quelle commune a généré le plus de CA data pour JAMONO NEW S’COOL? Sur la clause WHERE vous pouvez mettre " \
            "par exemple variable LIKE '%JAMONO' ou variable LIKE '%NEW S’C%', etc. Appliquez ses exemples dans ses genres cas." \
            "En résumé n'utilise jamais égal dans une clause WHERE avec caractère ou chaine de caractères, utilise toujours LIKE avec une partie du groupe de mot." \
            "N'utilisez jamais tout les mots donnés sur la question sur la condition de la requete (par exemple : variable LIKE '%JAMONO NEW S’COOL%' comme dans l'exemple précédent)" \
            "Sur les questions concernant les parc des takers, si 'maxit' ou 'digital' n'est pas renseigné sur la question, vous " \
            "ne devez pas interroger les tables suffixées 'maxit'.Attention ne fait jamais une requete pour supprimer, " \
            "pour modifier ou pour mettre à jour ou pour insérer dans la base. votre but est de sélectionner, alors " \
            "mettez seulement des requetes SQL qui permet de faire la sélection. Sélectionnez toujours les colonnes 'year' et 'month' " \
            "sur la requete et utilisez les memes nom de colonne pour les alias (exemple : as year et as month). Sachez que le mois est toujours sous format numérique." \
            "Si l'année n'est pas spécifiée ou renseignée sur la question, filtrez toujours sur l'année en cours (YEAR(CURRENT_DATE()))) ou le max des années pour ne pas " \
            "retourner les données de tout les années.Autrement dit, fait toujours un filtre de l'année en cours si l'année n'est " \
            "pas renseigné sur la question, utilise toujours des 'Alias' par exemple 'nom_table.nom_colonne' " \
            "pour éviter d'avoir des erreurs d'exécution provoqué par un nom de colonne.Notez que les colonnes avec comme nom 'day' représente les " \
            "jours et ont pour format numérique (yyyymmdd).Gardez cela en mémoire, il vous servira pour les questions sur les données quotidiennes.A " \
            "Notez que 'Airtime' signifie recharge à partir de ton crédit.Par rapport au question de segmentation, " \
            "analyse bien la question pour donner en retour une réponse claire qui permet de définir bien les différents segments(ou clusters) demandés." \
            "Sur la requete qui sera produite, ne met jamais une limite à moins que la requete vous " \
            "l'oblige à le faire. Par exemple, vous pouvez avoir comme requete : Donnez le top 10 des chiffres d'affaires des régions ou Quelle commune " \
            "a la balance moyenne SARGAL la plus élevée.Dans ses genres de question,vous pouvez utilisez la clause LIMIT dans la requete." \
            "La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une requete SQL valide qui peut etre exécutée " \
            "sur n'importe quelle version de HiveSQL sans erreur. Pour les questions sur la corrélation, n'utilise jamais la fonction " \
            "d'aggrégation 'CORR()' car cela n'a pas marché. l'erreur dit que cette fonction n'est pas supportée, essaie plutot de " \
            "le calculer en appliquant la formule de la corrélation.Autrement dit, n'utilise jamais une fonction SQL obsolète dans " \
            "la requete.Ne met jamais de requete avec une sous requete sur la clause WHERE." \
            "Par exemple,  WHERE year = (SELECT ....), Utilise plutot JOIN ON à place. " \
            \
            "Sur la requete SQL, ordonnez toujours le résultat du plus récents au plus anciens ou du plus grand au plus petit." \
            "Analysez la question pour voir, est ce que vous devez de faire une requete simple,d'aggrégation,de jointure ou combiné." \
            \
            "Sur les questions sur les souscriptions des offres, Notez que nous avons des catégories ou types d'offres (Pass Internet, illimix, illiflex, bundles, Mixel, International, NC) avec leur " \
        "formule tarifaire ou commerciale (JAMONO NEW S'COOL, JAMONO ALLO, JAMONO PRO, JAMONO MAX, AUTRES) et leurs segments recharges (Mass-Market, High, Middle, S0, super high) " \
        "et marché (JEUNES, ENTRANT, KIRENE AVEC ORANGE, AUTRES,MILIEU DE MARCHE,HAUT DE MARCHE, TRES HAUT DE MARCHE ) correspondant. Chaque type ou catégorie d'offres contient " \
        "plusieurs offres de souscriptions. Notez qu'aussi les offres data font références aux offres de type PASS INTERNET." \
        \
        "Si vous recevez une question sur le chiffre d'affaire tout court (Donne moi le Ca, quel est l'évolution du CA, etc) sans que l'utilisateur précise le chiffre " \
        "d'affaire, sachez que l'utilisateur veut simplement le chiffre d'affaire total qui n'est rien d'autre que le chiffre d'affaire des recharges " \
        "(ca recharges). Le chiffre d'affaire seulement fait référence au chiffres d'affaires des recharges(ca_recharges) ou CA total." \
        \
        "Produit en retour une requete SQL qui est claire, structurée, correcte et exécutable sous Hive ,prenez seulement tout les " \
            "colonnes qui contiennent l'information de la question.Utilisez toujours des alias dans la requete et ajouter les colonnes year" \
            \
            "et month dans la sélection.La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une " \
            "requete SQL valide sous Hive. Voici quelques genres d'exemples de requete SQL pour la sortie : " \
            "SELECT c.year as year,c.month as month,c.id, c.nom, c.ville, c.date_vente FROM clients as c WHERE c.ville LIKE " \
            "'Dakar' AND c.year = YEAR(CURRENT_DATE());" \
            "SELECT v.year as year,v.month as month,v.id, v.nom, v.ville, MONTH(v.date_vente), SUM(v.montant) FROM " \
            "ventes v WHERE v.year = YEAR(CURRENT_DATE()) GROUP BY MONTH(v.date_vente) ORDER BY MONTH(v.date_vente);" \
            "SELECT clients.year as year, clients.month as month, clients.id_client, clients.nom, clients.ville, " \
            "ventes.date_vente ventes.montant FROM clients JOIN ventes ON clients.id_client = ventes.id_client WHERE clients.year = 2024; " \
            "SELECT rcdm.year AS year, rcdm.month AS month,(AVG(rcdm.volumes * rcdm.ca_data) - AVG(rcdm.volumes) * AVG(rcdm.ca_data)) / " \
            "(STDDEV_POP(rcdm.volumes) * STDDEV_POP(rcdm.ca_data)) AS correlation_volume_revenu FROM reporting_ca_data_monthly AS rcdm JOIN " \
            "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS max_year_subquery ON rcdm.year=max_year_subquery.max_year " \
            "GROUP BY rcdm.year, rcdm.month ORDER BY rcdm.year DESC, rcdm.month DESC;" \
            "SELECT T1.year AS year,T1.month AS month,SUM(T1.ca_data) AS revenu FROM reporting_ca_data_monthly AS T1 JOIN " \
            "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS T2 ON T1.year = T2.max_year GROUP BY " \
            "T1.year, T1.month ORDER BY T1.year DESC, T1.month DESC;" \
            \
            "NB : N'oubliez pas de faire la filtre sur les années, si l'année n'est pas précisé sur la question ({requete}), faite un filtre " \
            "sur l'année en cours (par exemple year = YEAR(CURRENT_DATE())). Le YEAR(CURRENT_DATE())) est obligatoire si l'année n'est pas " \
            "spécifié sur la question. Si vous n'avez pas reçu de tables, n'invente pas de tables et ne " \
            "produit pas de requete, dites juste à l'utilisateur de reformuler sa question pour qu'il soit beaucoup plus clair.Utilisez que les " \
            "colonnes qui vous ont été fourni sur la ou les tables et n'essayez pas d'ajouter ou d'inventer une colonne meme si les informations " \
            "de la ou des table(s) sont insuffisant",

    response = model.generate_content(
        contents= prompt
        #generation_config={"max_output_tokens": 600}
    )

    path = f"{user_dir}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    sql_query = Extract_sql(path)

    print(sql_query)

    return sql_query
    #return response.text




#### RAG + Crewai
def SQL_Agent(requete, user_dir):
    
    ####### Recherche dans le RAG
    result_table = Agent_analyst_RAG_Gemini(requete)

    #check = Agent_reflex_pattern(requete,result_table)
    #result_table = check

    #### Recherche d'infos supplémentaire
    infos = Agent_info_supp_Gem(result_table, requete, user_dir)
    data = f"{result_table}\n\n{infos}"

    Gemini = LLM(
        model='gemini/gemini-2.0-flash',

        #model='gemini/gemini-1.5-flash',

        #model = 'gemini/gemini-2.0-flash-lite',
        #model='gemini/gemini-2.5-flash-preview-04-17',
        
        temperature = 0.1,
        #api_key = api,
        #max_tokens= 32000
        #api_key= "AIzaSyBxwZpwLsAl3YDWBfnsXd35djouNV3lX3E"
        api_key= api_k
    )

    agent_sql = Agent(
        role = "Spécialiste en Production de requete SQL sous Hive",
        goal = "Mise en place d'une requete SQL en utilisant la sortie (la ou les table(s)) de l'agent 'Analyste base de données'.",
        backstory = "Vous etes un expert et professionnel en SQL avec plus de 30 ans d'expérience. Vous traduisez " \
                " les questions métiers en une requete SQL pour recupérer des informations sur la base donnée de votre entreprise.",
        verbose = True,
        reasoning = True,
        #llm= Gemini,
        llm= Gemini,
        max_rpm = 10,
        max_iter= 20,
        #llm = gpt4,
        #llm = groq_llama,
        
        #memory = memory_request
    )



    task_resquest_db = Task(
        name = "Tache SQL",
        agent = agent_sql,
        #tools= [sql_tools],

        description = "Vous allez recupérer les informations suivante pour produire une requete SQL valide qui permet de répondre à \
                      cette question métier, {requete}.: {data}. \n\n"
                    # et les informations supplémentaires données en sortie par la tache 'Infos supplémentaire'(s'il y en a) .Ses infos supplémentaires 
                    # sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut 
                    # etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous
                    # permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une valide et correcte par rapport à nos données.
                      #"Ses infos supplémentaires" \
                      #"sont utiles pour les clauses WHERE pour éviter d'avoir des résultats vide après l'exécution de la requete.Ce qui peut" \
                      #"etre due à une mauvaise compréhension des éléments uniques d'une ou des colonnes de la base. Ses infos supplémentaires vous" \
                      #"permettent de savoir certain(s) valeur(s)unique(s) de certain(s) colonne(s) pour pouvoir prodruire une requete SQL " \
                      #"valide et correcte par rapport à nos données.Sur lA clause WHERE de la requete, utilisez toujours la ou les valeur(s) par les infos supplémentaire." \
                      "Notez qu'à chaque fois que vous recevez une question qui parle de 'data',sachez que cela fait référence à internet. Autrement " \
                      "dit, dans les données,'data' veut dire 'internet' ou 'pass internet' (Par exemple : offre data signifie offre pass internet)." \
                      "Abonnés, bénéficaires, etc font référence aux clients.'OM' signifie 'Orange Money'."
                      #"Et aussi le nombre de clients (ou abonnées,etc) est différent au nombre de souscription"
                      " Analyse bien la ou les tables reçue(s) pour voir est ce que l'information de la question posée est dans une ou plusieurs " \
                      "colonnes. Regarde bien aussi est-ce-que vous avez reçu une ou plusieurs table(s)." \
                      "Parfois, vous pouvez recevoir plusieurs tables, et que chaque table peut répondre à la question posée. Dans ce cas, prenez " \
                      "la table qui donne le plus d'information pour répondre à la question(Privilégiez toujours les tables commençant par 'reporting', " \
                      "ou 'monthly' à moins que la question se refére sur les informations quotidiennes ('daily'))."
                      "Si vous avez reçu une seule table, produit une requete SQL correcte"
                      " avec cette table seulement en prenant que les colonnes qui contiennent " \
                      "l'information de la réponse sur la table. S'il y a 2 "
                      "ou plusieurs tables reçues et que la réponse se trouve sur les différents tables, analyse les colonnes de chaque table, puis étudier les dépendances "
                      "entre les tables, c'est à dire les colonnes qui permet de faire la liaison entre les tables."
                      "Après cela, produit une requete SQL correcte avec les colonnes qui contiennent l'information sur les différentes " \
                      "tables reçues pour répondre à cette question métier : "
                      "{requete}. Si la question est trop vaste, (par exemple 'Je veux le chiffre d'affaire', 'Quelle formule tarifaire " \
                      "génère le plus de trafic 4G sur le mois d'avril', ect) vous essayerez toujours de répondre en donnant une " \
                      "requete SQL qui donne les informations les plus récentes. Dans cet exemple, vous donneriez le chiffre d'affaire ou " \
                      "formule tarifaire  de l'année en cours (2025), Si la question " \
                      "est posée en 2025, vous donnez une requete qui donne les information de 2025, de meme que si c'est en 2026, etc. " \
                      "Si vous avez une requete qui nécessite une condition et si la condition "
                      "doit se faire avec des caractères ou chaines de caractères (Ex : Région de Dakar), sur "
                      "la clause WHERE, Utilise LIKE plutot égal (=) par exemple '%Dakar%', '%DAKAR' ou '%kar%', etc ." \
                      "Exemple de question : Quelle commune a généré le plus de CA data pour JAMONO NEW S’COOL? Sur la clause WHERE vous pouvez mettre " \
                      "par exemple variable LIKE '%JAMONO' ou variable LIKE '%NEW S’C%', etc. Appliquez ses exemples dans ses genres cas." \
                      "En résumé n'utilise jamais égal dans une clause WHERE avec caractère ou chaine de caractères, utilise toujours LIKE avec une partie du groupe de mot."
                      "N'utilisez jamais tout les mots donnez sur la question sur la condition de la requete (par exemple : variable LIKE '%JAMONO NEW S’COOL%' comme dans l'exemple précédent)"
                      "Sur les questions concernant les parc des takers, si 'maxit' ou 'digital' n'est pas renseigné dans la question, vous ne devez pas interroger les tables suffixées 'maxit'."
                      "Attention ne fait jamais une requete pour supprimer, "
                      "pour modifier ou pour mettre à jour ou pour insérer dans la base. votre but est de sélectionner, alors "
                      "mettez seulement des requetes SQL qui permet de faire la sélection. Sélectionnez toujours les colonnes 'year' et 'month' " \
                      "sur la requete et utilisez les memes nom de colonne pour les alias (exemple : as year et as month). Sachez que le mois est toujours sous format numérique." \
                      "Si l'année n'est pas spécifiée ou renseignée sur la question, filtrez toujours sur l'année en cours ou le max des années pour ne pas retourner les données de tout les " \
                      "années.Autrement dit, fait toujours un filtre de l'année(2025 toujours si l'année n'est pas renseigné sur la question) sur la requete et utilise toujours des 'Alias' par exemple 'nom_table.nom_colonne' " \
                      "pour éviter d'avoir des erreurs d'exécution provoqué par un nom de colonne.A chaque fois que vous recevez une question qui parle de "
                      "'data',sachez que cela fait référence à internet. Autrement dit, dans les données, 'data' veut dire 'internet' ou 'pass internet'."
                     "Notez que 'Airtime' signifie recharge à partir de ton crédit."
                      "Par rapport au question de segmentation, analyse bien la question pour donner en retour une réponse claire qui permet de définir bien les différents segments(ou clusters) demandés." \
                      "Sur la requete qui sera produite, ne met jamais une limite à moins que la requete vous " \
                      "l'oblige à le faire. Par exemple, vous pouvez avoir comme requete : Donnez le top 10 des chiffres d'affaires des régions ou Quelle commune "
                      "a la balance moyenne SARGAL la plus élevée.Dans ses genres de question,vous pouvez utilisez la clause LIMIT dans la requete." \
                      "La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une requete SQL valide qui peut etre exécutée " \
                      "sur n'importe quelle version de HiveSQL sans erreur. Pour les questions sur la corrélation, n'utilise jamais la fonction " \
                      "d'aggrégation 'CORR()' car cela n'a pas marché. l'erreur dit que cette fonction n'est pas supportée, essaie plutot de " \
                      "le calculer en appliquant la formule de la corrélation.Autrement dit, n'utilise jamais une fonction obsolète dans la requete." \
                      "Ne met jamais de requete avec une sous requete sur la clause WHERE. Par exemple,  WHERE year = (SELECT ....), Utilise plutot " \
                      
                      #"sélectionner tout les colonnes reçues, si sélectionner tout sinon prenez deux colonnes (ou "
                      #"plus) qui ont plus de sens par rapport à la question {requete}. "
                      "JOIN ON à place.Sur la requete SQL, ordonnez toujours le résultat du plus récents au plus anciens ou du plus grand au plus petit."
                      "Analysez la question et les données fournies pour voir, est ce que vous devez de faire une requete simple,d'aggrégation,de jointure ou combiné" \
                      #"simple,d'aggrégation,de jointure ou combiné."
                      #f"Voici une petite historique des derniers messages \n:{context}\n",
                        ,
        #context = [task_analyst_db, task_sup],
        expected_output = "Une requete SQL uniquement qui est claire, structurée et correcte qui répond à la question:{requete}." \
                          "Ne prend pas de colonne unutile et aussi n'ajoute jamais des commentaires sur la requete, prenez seulement " \
                          "tout les colonnes qui contiennent l'information de la question et sélectionnez tout les colonnes qui ont " \
                          "été utilisées sur la requete.Utilisez toujours des alias dans la requete et ajouter les colonnes year " \
                          "et month dans la sélection.La requete doit etre exécutée sous 'Hive'.Alors, produit en retour une " \
                          "requete SQL valide sous Hive. Voici quelques genres d'exemples de requete SQL pour la sortie : "
                          "SELECT c.year as year,c.month as month,c.id, c.nom, c.ville, c.date_vente FROM clients as c WHERE c.ville LIKE 'Dakar';"
                          "SELECT v.year as year,v.month as month,v.id, v.nom, v.ville, MONTH(v.date_vente), SUM(v.montant) FROM "
                          "ventes v GROUP BY MONTH(v.date_vente) ORDER BY MONTH(v.date_vente);"
                          "SELECT clients.year as year, clients.month as month, clients.id_client, clients.nom, clients.ville, " \
                          "ventes.date_vente ventes.montant FROM clients JOIN ventes ON clients.id_client = ventes.id_client; " \
                          "SELECT rcdm.year AS year, rcdm.month AS month,(AVG(rcdm.volumes * rcdm.ca_data) - AVG(rcdm.volumes) * AVG(rcdm.ca_data)) / "
                          "(STDDEV_POP(rcdm.volumes) * STDDEV_POP(rcdm.ca_data)) AS correlation_volume_revenu FROM reporting_ca_data_monthly AS rcdm JOIN " \
                          "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS max_year_subquery ON rcdm.year=max_year_subquery.max_year " \
                          "GROUP BY rcdm.year, rcdm.month ORDER BY rcdm.year DESC, rcdm.month DESC;" \
                          "SELECT T1.year AS year,T1.month AS month,SUM(T1.ca_data) AS revenu FROM reporting_ca_data_monthly AS T1 JOIN "
                          "(SELECT MAX(year) AS max_year FROM reporting_ca_data_monthly) AS T2 ON T1.year = T2.max_year GROUP BY " \
                          "T1.year, T1.month ORDER BY T1.year DESC, T1.month DESC; " \
                          "NB : N'oubliez pas de faire la filtre sur les années, si l'année n'est pas précisé sur la question ({requete}), faite un filtre "
                        "sur le CURRENTYEAR (par exemple year = YEAR(CURRENT_DATE())). Si vous n'avez pas reçu de tables, n'invente pas de tables et ne "
                        "produit pas de requete, dites juste à l'utilisateur de reformuler sa question pour qu'il soit beaucoup plus clair." ,
        output_file = f"{user_dir}.txt"
    )

    systeme = Crew(
        agents= [agent_sql],
        tasks= [task_resquest_db],
        #memory= True,
        process=Process.sequential,
        #verbose= True
    )

    input = {
        #"table_desc" : table_desc,
        #"colonne_desc" : colonne_desc,
        "requete" : requete ,
        #"caracteristique" : caracteristique,
        "data" : data
        }

    try :
        #st.write("Veuillez patientez, votre requete est en cours d'exécution ...")
        system_multi_agent = systeme.kickoff(inputs=input)
        print(system_multi_agent.token_usage)
        path = f"{user_dir}.txt"
        #sql_query = extract_sql("result.txt")
        sql_query = Extract_sql(path)

    except Exception as e :
        st.error(f"Le modèle ne répond pas pour le moment, veuillez réessayez plus tard : {e}.")
        sql_query = None

    return sql_query



