# ============================================================
# TRADECLEANSE — NOTEBOOK 01 : Audit & Profiling Initial
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CELLULE 1 — Chargement multi-sources
# ============================================================
# CONSIGNE :
# Le dataset consolide 3 sources heterogenes :
#   - Bloomberg   : fichier CSV (colonnes marche : bid, ask, mid_price, price,
#                                volume_j, volatility_30d)
#   - Murex (SQL) : transactions internes (trade_id, dates, notional, quantity,
#                                          trader_id, asset_class)
#   - Refinitiv   : donnees fondamentales (credit_rating, sector,
#                                          counterparty_name, country_risk)
#
# TACHE :
#   1. Chargez tradecleanse_raw.csv avec les parametres d'import appropries.
#      Pensez a gerer : encodage, separateur, valeurs sentinelles connues,
#      types de colonnes.
#   2. Simulez les 3 sources en creant 3 sous-dataframes avec uniquement
#      les colonnes correspondant a chaque source.
#   3. Ajoutez une colonne "source" sur chacun avant consolidation.
#
# LIBRAIRIES SUGGÉREES : pandas, sqlite3 (pour simuler Murex en SQL)

# --- Votre code ici ---

# ============================================================
# CELLULE 1 — Chargement multi-sources
# ============================================================

# 1. Chargement du dataset principal
df = pd.read_csv(
    "data/tradecleanse_raw.csv",
    encoding="utf-8",
    sep=",",
    na_values=["N/A", "#N/A", "-", "null", "nd", "99999"]
)

# Conversion des dates
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")

print("Dataset chargé :", df.shape)


# 2. Simulation des sources

# Bloomberg (marché)
bloomberg_cols = ["isin", "price", "bid", "ask", "mid_price", "volume_j", "volatility_30d"]
df_bloomberg = df[bloomberg_cols].copy()
df_bloomberg["source"] = "bloomberg"

# Murex (transactions)
murex_cols = ["trade_id", "counterparty_id", "trade_date", "settlement_date",
              "asset_class", "notional_eur", "quantity", "trader_id"]
df_murex = df[murex_cols].copy()
df_murex["source"] = "murex"

# Refinitiv (fondamentaux)
refinitiv_cols = ["counterparty_name", "credit_rating", "sector", "country_risk", "default_flag"]
df_refinitiv = df[refinitiv_cols].copy()
df_refinitiv["source"] = "refinitiv"


print("Sources simulées :")
print("Bloomberg :", df_bloomberg.shape)
print("Murex :", df_murex.shape)
print("Refinitiv :", df_refinitiv.shape)


# ============================================================
# CELLULE 2 — Profiling initial
# ============================================================
# CONSIGNE :
# Produisez un rapport de profiling complet du dataset brut.
# Il doit contenir au minimum :
#   - Shape (nb lignes, nb colonnes)
#   - Types detectes par pandas
#   - Taux de valeurs manquantes par colonne (count + %)
#   - Statistiques descriptives (min, max, mean, std, quartiles)
#     pour toutes les colonnes numeriques
#   - Cardinalite (nb de valeurs uniques) pour les colonnes categorielles
#   - Distribution de chaque colonne categorielle (value_counts)
#   - Nombre de doublons exacts et doublons sur trade_id
#
# LIBRAIRIES SUGGÉREES : pandas (.describe, .isnull, .value_counts, .duplicated)

# --- Votre code ici ---

# ============================================================
# CELLULE 2 — Profiling initial
# ============================================================

print("\n==================== PROFILING DATASET ====================\n")

# 1. Shape
print("Shape du dataset :")
print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes\n")


# 2. Types de données
print("Types de colonnes :")
print(df.dtypes, "\n")


# 3. Valeurs manquantes
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct
}).sort_values(by="missing_pct", ascending=False)

print("Valeurs manquantes par colonne :")
print(missing_df, "\n")


# 4. Statistiques descriptives (numériques)
print("Statistiques descriptives (variables numériques) :")
print(df.describe(), "\n")


# 5. Cardinalité (variables catégorielles)
print("Cardinalité des variables catégorielles :")
for col in df.select_dtypes(include="object").columns:
    print(f"{col} : {df[col].nunique()} valeurs uniques")
print()


# 6. Distribution des variables catégorielles
print("Distribution des variables catégorielles :")
for col in df.select_dtypes(include="object").columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().head(10))


# 7. Doublons
duplicates_full = df.duplicated().sum()
duplicates_trade_id = df["trade_id"].duplicated().sum()

print("\nDoublons exacts :", duplicates_full)
print("Doublons sur trade_id :", duplicates_trade_id)

# ============================================================
# CELLULE 3 — Detection des anomalies
# ============================================================
# CONSIGNE :
# A partir du profiling, identifiez et quantifiez chaque anomalie.
# Pour chaque anomalie trouvee, vous devez indiquer :
#   - Le type (doublon / valeur manquante / outlier / incoherence / format...)
#   - La ou les colonnes concernees
#   - Le nombre de lignes impactees
#   - La criticite metier (impact sur le modele de risque)
#
# Construisez un dictionnaire ou DataFrame "anomalies_report" qui recense
# tout ce que vous avez trouve.
#
# RAPPEL DES COLONNES ET LEURS REGLES METIER :
#   trade_id          : doit etre unique
#   settlement_date   : doit etre >= trade_date (regle T+2 actions)
#   bid / ask         : bid doit toujours etre < ask
#   mid_price         : doit etre egal a (bid + ask) / 2
#   price             : doit se trouver dans la fourchette [bid, ask] +/- 0.5%
#   notional_eur      : doit etre positif (sauf position short documentee)
#   asset_class       : doit appartenir a {equity, bond, derivative, fx}
#   credit_rating     : valeurs valides AAA AA A BBB BB B CCC D
#   country_risk      : doit etre compris entre 0 et 100
#   volatility_30d    : doit etre > 0 et < 200
#   default_flag      : valeurs valides 0 ou 1 uniquement
#   credit_rating + default_flag : un emetteur note AAA/AA/A ne peut pas
#                                  avoir default_flag = 1 (contradiction)

# --- Votre code ici ---

# ============================================================
# CELLULE 3 — Detection des anomalies
# ============================================================

anomalies = []

# Conversion des colonnes mal typées
df["volatility_30d"] = pd.to_numeric(df["volatility_30d"], errors="coerce")


# 1. Doublons trade_id
dup_trade = df[df["trade_id"].duplicated()]
anomalies.append({
    "type": "doublon",
    "colonne": "trade_id",
    "nb_lignes": len(dup_trade),
    "criticite": "élevée — violation unicité clé métier"
})


# 2. settlement_date < trade_date
invalid_dates = df[df["settlement_date"] < df["trade_date"]]
anomalies.append({
    "type": "incoherence date",
    "colonne": "settlement_date",
    "nb_lignes": len(invalid_dates),
    "criticite": "élevée — violation règle T+2"
})


# 3. bid > ask
bid_ask_error = df[df["bid"] > df["ask"]]
anomalies.append({
    "type": "incoherence marché",
    "colonne": "bid/ask",
    "nb_lignes": len(bid_ask_error),
    "criticite": "critique — marché invalide"
})


# 4. mid_price incohérent
mid_calc = (df["bid"] + df["ask"]) / 2
mid_error = df[abs(df["mid_price"] - mid_calc) > 0.01 * mid_calc]

anomalies.append({
    "type": "incoherence prix",
    "colonne": "mid_price",
    "nb_lignes": len(mid_error),
    "criticite": "élevée — incohérence pricing"
})


# 5. price hors fourchette
price_error = df[
    (df["price"] < df["bid"] * 0.995) |
    (df["price"] > df["ask"] * 1.005)
]

anomalies.append({
    "type": "out of range",
    "colonne": "price",
    "nb_lignes": len(price_error),
    "criticite": "élevée — prix incohérent"
})


# 6. notional négatif
neg_notional = df[df["notional_eur"] < 0]
anomalies.append({
    "type": "valeur négative",
    "colonne": "notional_eur",
    "nb_lignes": len(neg_notional),
    "criticite": "moyenne — possible short mais suspect"
})


# 7. asset_class invalide
valid_assets = ["equity", "bond", "derivative", "fx"]
invalid_asset = df[~df["asset_class"].str.lower().isin(valid_assets)]

anomalies.append({
    "type": "valeur invalide",
    "colonne": "asset_class",
    "nb_lignes": len(invalid_asset),
    "criticite": "moyenne — référentiel non respecté"
})


# 8. credit_rating invalide
valid_rating = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
invalid_rating = df[~df["credit_rating"].isin(valid_rating)]

anomalies.append({
    "type": "valeur invalide",
    "colonne": "credit_rating",
    "nb_lignes": len(invalid_rating),
    "criticite": "élevée — impact modèle risque"
})


# 9. country_risk hors range
risk_error = df[(df["country_risk"] < 0) | (df["country_risk"] > 100)]

anomalies.append({
    "type": "out of range",
    "colonne": "country_risk",
    "nb_lignes": len(risk_error),
    "criticite": "élevée"
})


# 10. volatility hors range
vol_error = df[(df["volatility_30d"] <= 0) | (df["volatility_30d"] > 200)]

anomalies.append({
    "type": "outlier",
    "colonne": "volatility_30d",
    "nb_lignes": len(vol_error),
    "criticite": "élevée"
})


# 11. default_flag invalide
default_error = df[~df["default_flag"].isin([0, 1])]

anomalies.append({
    "type": "valeur invalide",
    "colonne": "default_flag",
    "nb_lignes": len(default_error),
    "criticite": "critique"
})


# 12. contradiction rating + défaut
contradiction = df[
    (df["credit_rating"].isin(["AAA", "AA", "A"])) &
    (df["default_flag"] == 1)
]

anomalies.append({
    "type": "contradiction",
    "colonne": "credit_rating + default_flag",
    "nb_lignes": len(contradiction),
    "criticite": "critique — incohérence risque"
})


# Résultat final
anomalies_report = pd.DataFrame(anomalies)

print("\n==================== ANOMALIES DETECTEES ====================\n")
print(anomalies_report)

# ============================================================
# CELLULE 4 — Visualisations
# ============================================================
# CONSIGNE :
# Produisez au minimum 4 graphiques illustrant les anomalies detectees :
#   - Taux de valeurs manquantes par colonne (barh)
#   - Distribution de asset_class (toutes variantes)
#   - Scatter bid vs ask (mettre en evidence les inversions)
#   - Distribution du delai settlement - trade_date (histogramme)
#
# Sauvegardez le tout dans un seul fichier : 01_profiling_report.png

# --- Votre code ici ---

# ============================================================
# CELLULE 4 — Visualisations
# ============================================================

import matplotlib.pyplot as plt

# 1. Missing values
missing_pct = (df.isnull().sum() / len(df)) * 100

plt.figure()
missing_pct.sort_values().plot(kind="barh")
plt.title("Taux de valeurs manquantes (%)")


# 2. Distribution asset_class
plt.figure()
df["asset_class"].value_counts().plot(kind="bar")
plt.title("Distribution asset_class")


# 3. Scatter bid vs ask
plt.figure()
plt.scatter(df["bid"], df["ask"], alpha=0.3)
plt.xlabel("bid")
plt.ylabel("ask")
plt.title("Bid vs Ask (détection anomalies)")


# 4. Délai settlement - trade_date
df["delay"] = (df["settlement_date"] - df["trade_date"]).dt.days

plt.figure()
df["delay"].hist(bins=50)
plt.title("Distribution délai settlement")


# Sauvegarde (IMPORTANT)
plt.savefig("01_profiling_report.png")

print("\nGraphiques sauvegardés dans 01_profiling_report.png")
