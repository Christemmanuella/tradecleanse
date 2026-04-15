# ============================================================
# TRADECLEANSE — NOTEBOOK 02 : Pipeline de Nettoyage Complet
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================
#
# CONTRAINTES OBLIGATOIRES :
#   - Ne jamais modifier tradecleanse_raw.csv
#   - Toujours travailler sur une copie : df = pd.read_csv(...).copy()
#   - Chaque etape doit etre loggee : nb lignes avant / apres / supprimees
#   - Chaque decision doit etre justifiee en commentaire (raison METIER)
#   - Le dataset final doit etre sauvegarde dans : tradecleanse_clean.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging (ne pas modifier)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('tradecleanse_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CHARGEMENT (ne pas modifier)
# ============================================================
df_raw = pd.read_csv('data/tradecleanse_raw.csv', low_memory=False)
df = df_raw.copy()
logger.info(f"Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================
# ETAPE 1 — Remplacement des valeurs sentinelles
# ============================================================
# CONSIGNE :
# Identifiez et remplacez TOUTES les valeurs sentinelles par NaN.
# Une sentinelle est une valeur utilisee a la place d'un NaN reel :
# textuelles (#N/A, N/A, #VALUE!, -, nd, null...) ET numeriques
# (ex: 99999 utilise comme code "donnee manquante" sur country_risk).
#
# ATTENTION : certaines colonnes sont en type "object" a cause des
# sentinelles textuelles melangees a des valeurs numeriques.
# Pensez a gerer le cast des colonnes apres nettoyage.
#
# Loggez le nb de NaN total avant et apres.

before = len(df)
# --- Votre code ici ---

# Comptage des NaN avant traitement
nan_before = df.isna().sum().sum()

# Liste des valeurs sentinelles (métier + erreurs Excel/API)
sentinels = ["#N/A", "N/A", "#VALUE!", "-", "nd", "null"]

# Remplacement des sentinelles textuelles par NaN
df.replace(sentinels, np.nan, inplace=True)

# Cas particulier : 99999 utilisé comme code "valeur manquante" pour country_risk
df.loc[df["country_risk"] == 99999, "country_risk"] = np.nan

# Conversion de volatility_30d en numérique (car mélange texte/nombre)
df["volatility_30d"] = pd.to_numeric(df["volatility_30d"], errors="coerce")

# Comptage des NaN après traitement
nan_after = df.isna().sum().sum()

# Log des résultats
logger.info(f"NaN avant : {nan_before}")
logger.info(f"NaN après : {nan_after}")
logger.info(f"NaN ajoutés : {nan_after - nan_before}")

# ============================================================
# JUSTIFICATION METIER :
# Les valeurs sentinelles (#N/A, #VALUE!, etc.) sont utilisées
# par certaines sources (Excel, APIs financières) pour représenter
# des données manquantes.
#
# Leur conversion en NaN permet :
# - un traitement cohérent avec pandas
# - d'éviter des erreurs de calcul
# - d'assurer la qualité des analyses de risque
#
# Le code 99999 est une convention métier pour "valeur inconnue"
# sur le country_risk, il doit donc être converti en NaN.
# ============================================================
logger.info(f"[Sentinelles] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 2 — Suppression des doublons
# ============================================================
# CONSIGNE :
# Supprimez les doublons sur la cle metier trade_id.
# Justifiez dans un commentaire : pourquoi garder "first" ou "last" ?
# Dans le contexte Murex, quel enregistrement est le plus fiable ?
#
# Loggez : nb de doublons exacts, nb de doublons sur trade_id, shape finale.

before = len(df)
# --- Votre code ici ---
# Comptage des doublons exacts
duplicates_exact = df.duplicated().sum()

# Comptage des doublons métier (trade_id)
duplicates_trade_id = df.duplicated(subset=["trade_id"]).sum()

logger.info(f"Doublons exacts : {duplicates_exact}")
logger.info(f"Doublons sur trade_id : {duplicates_trade_id}")

# Suppression des doublons métier
# On garde le premier enregistrement (logique métier Murex)
df = df.drop_duplicates(subset=["trade_id"], keep="first")

# ============================================================
# JUSTIFICATION METIER :
# Dans un système de trading (ex: Murex), chaque trade_id
# doit être UNIQUE.
#
# Les doublons proviennent généralement :
# - de multiples exports
# - de flux dupliqués
# - de corrections tardives
#
# On garde "first" car :
# - le premier enregistrement correspond à la capture initiale
# - les suivants peuvent être des duplications techniques
#
# Supprimer ces doublons garantit :
# - unicité de la clé métier
# - fiabilité des agrégations
# ============================================================

after = len(df)

logger.info(f"[Doublons] {before} -> {after} lignes (supprimés : {before - after})")
logger.info(f"[Doublons] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 3 — Conversion et normalisation des types
# ============================================================
# CONSIGNE :
# Convertissez chaque colonne vers son type pandas correct :
#   - trade_date, settlement_date : datetime (attention aux formats mixtes)
#   - bid, ask, mid_price, price, notional_eur,
#     quantity, volume_j, volatility_30d, country_risk : numerique
#   - asset_class, credit_rating, sector : chaine minuscule + strip
#
# Utilisez errors='coerce' pour les conversions — les valeurs non
# convertibles deviendront NaN (vous les traiterez a l'etape 8).
# Loggez le nb de valeurs devenues NaN par conversion rate.

before = len(df)
# --- Votre code ici ---
# Sauvegarde des NaN avant conversion
nan_before = df.isna().sum()

# ============================================================
# Conversion des dates
# ============================================================
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")

# ============================================================
# Conversion des colonnes numériques
# ============================================================
numeric_cols = [
    "bid", "ask", "mid_price", "price",
    "notional_eur", "quantity", "volume_j",
    "volatility_30d", "country_risk"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# Normalisation des colonnes catégorielles
# ============================================================
cat_cols = ["asset_class", "credit_rating", "sector"]

for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# ============================================================
# Analyse des conversions → combien de NaN ajoutés
# ============================================================
nan_after = df.isna().sum()

nan_diff = (nan_after - nan_before)
nan_diff = nan_diff[nan_diff > 0]

logger.info("NaN ajoutés par conversion :")
logger.info(f"\n{nan_diff}")

# ============================================================
# JUSTIFICATION METIER :
#
# - errors='coerce' permet de transformer les valeurs invalides
#   (ex: '#VALUE!', texte, erreurs Excel) en NaN
#
# - Cela garantit :
#   ✔ des calculs fiables
#   ✔ aucune erreur bloquante
#   ✔ un dataset exploitable pour le risk modeling
#
# - Normalisation des catégories :
#   évite les doublons logiques (ex: 'EQ', 'equity', 'Equity')
# ============================================================

after = len(df)
logger.info(f"[Types] {before} -> {after} lignes")
logger.info(f"[Types] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 4 — Normalisation du referentiel asset_class
# ============================================================
# CONSIGNE :
# La colonne asset_class contient de nombreuses variantes pour les
# 4 valeurs valides : equity, bond, derivative, fx.
# Construisez un dictionnaire de mapping exhaustif et appliquez-le.
# Toute valeur non mappee doit devenir NaN.
#
# Verifiez apres correction que seules les 4 valeurs existent.

before = len(df)
# --- Votre code ici ---
# ============================================================
# Mapping métier des asset_class
# ============================================================

mapping_asset = {
    # EQUITY
    "equity": "equity",
    "eq": "equity",

    # BOND
    "bond": "bond",
    "fixed income": "bond",
    "bond ": "bond",

    # FX
    "fx": "fx",
    "foreign exchange": "fx",

    # DERIVATIVE
    "derivative": "derivative",
    "deriv": "derivative"
}

# Application du mapping
df["asset_class"] = df["asset_class"].map(mapping_asset)

# Comptage des valeurs devenues NaN (non reconnues)
invalid_asset = df["asset_class"].isna().sum()

logger.info(f"Valeurs asset_class invalides (non mappées) : {invalid_asset}")

# Vérification finale
unique_values = df["asset_class"].dropna().unique()
logger.info(f"Valeurs finales asset_class : {unique_values}")

# ============================================================
# JUSTIFICATION METIER :
#
# Les classes d'actifs doivent être normalisées car :
# - utilisées dans les modèles de risque
# - utilisées pour les agrégations réglementaires
#
# Les variantes (EQ, BOND, DERIV...) proviennent :
# - de différentes sources (Bloomberg, Murex, Refinitiv)
#
# Toute valeur non reconnue devient NaN pour éviter
# une mauvaise classification métier.
# ============================================================

after = len(df)
logger.info(f"[asset_class] {before} -> {after} lignes")
logger.info(f"[asset_class] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 5 — Incoherences structurelles financieres
# ============================================================
# CONSIGNE :
# Corrigez les 6 types d'incoherences metier suivants.
# Pour chacun, loggez le nb de lignes concernees ET justifiez
# la correction choisie (NaN ? Recalcul ? Valeur absolue ?).
#
# 5a. settlement_date < trade_date
#     Regle : le reglement intervient toujours apres le trade (T+2).
#     -> Quelle valeur mettre a la place ?
#
# 5b. bid > ask
#     Regle : la fourchette est toujours bid < ask.
#     -> Que faire des deux colonnes concernees ?
#
# 5c. mid_price incoherent avec (bid + ask) / 2
#     Regle : mid = (bid + ask) / 2, tolerance 1%.
#     -> Comment le recalculer ?
#
# 5d. price en dehors de la fourchette [bid * 0.995, ask * 1.005]
#     Regle : le prix d'execution ne peut pas etre hors fourchette.
#     -> Quelle valeur de substitution choisir ?
#
# 5e. notional_eur negatif
#     Regle : le notionnel est toujours positif pour une transaction standard.
#     -> Comment corriger sans perdre l'information ?
#
# 5f. credit_rating AAA/AA/A avec default_flag = 1
#     Regle : une contrepartie en defaut ne peut pas etre notee investissement.
#     -> Que faire du rating ? Que faire du flag ?

before = len(df)
# --- Votre code ici ---
# ============================================================
# 5a. settlement_date < trade_date
# ============================================================
mask_date = df["settlement_date"] < df["trade_date"]
logger.info(f"Incohérences dates : {mask_date.sum()}")
df.loc[mask_date, "settlement_date"] = df.loc[mask_date, "trade_date"] + pd.Timedelta(days=2)

# ============================================================
# 5b. bid > ask
# ============================================================
mask_bid_ask = df["bid"] > df["ask"]
logger.info(f"Incohérences bid/ask : {mask_bid_ask.sum()}")
df.loc[mask_bid_ask, ["bid", "ask"]] = df.loc[mask_bid_ask, ["ask", "bid"]].values

# ============================================================
# 5c. mid_price incohérent
# ============================================================
mid_calc = (df["bid"] + df["ask"]) / 2
mask_mid = abs(df["mid_price"] - mid_calc) > (0.01 * mid_calc)

logger.info(f"Incohérences mid_price : {mask_mid.sum()}")
df.loc[mask_mid, "mid_price"] = mid_calc

# ============================================================
# 5d. price hors fourchette
# ============================================================
lower = df["bid"] * 0.995
upper = df["ask"] * 1.005

mask_price = (df["price"] < lower) | (df["price"] > upper)
logger.info(f"Prix hors fourchette : {mask_price.sum()}")
df.loc[mask_price, "price"] = df.loc[mask_price, "mid_price"]

# ============================================================
# 5e. notional_eur négatif
# ============================================================
mask_notional = df["notional_eur"] < 0
logger.info(f"Notional négatif : {mask_notional.sum()}")
df.loc[mask_notional, "notional_eur"] = df.loc[mask_notional, "notional_eur"].abs()

# ============================================================
# 5f. incohérence rating / défaut
# ============================================================
mask_rating = df["credit_rating"].isin(["aaa", "aa", "a"]) & (df["default_flag"] == 1)
logger.info(f"Incohérence rating/défaut : {mask_rating.sum()}")
df.loc[mask_rating, "credit_rating"] = np.nan

# ============================================================
# JUSTIFICATION METIER :
#
# - settlement_date : corrigée en T+2 (standard marché)
# - bid/ask inversés : erreur de flux → correction logique
# - mid_price : recalcul obligatoire (pricing)
# - price : remplacé par mid_price (valeur neutre marché)
# - notional : abs() conserve le montant économique
# - rating vs défaut : priorité au défaut (événement critique)
# ============================================================

after = len(df)
logger.info(f"[Incoherences financieres] {before} -> {after} lignes")
logger.info(f"[Incoherences financieres] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 6 — Regles metier (valeurs hors plage valide)
# ============================================================
# CONSIGNE :
# Appliquez les regles metier suivantes colonne par colonne.
# Justifiez pour chaque regle si vous mettez NaN ou supprimez la ligne.
#
#   country_risk   : doit etre dans [0, 100]
#   volatility_30d : doit etre dans [0.1, 200]
#   default_flag   : doit etre 0 ou 1
#   quantity       : doit etre > 0

before = len(df)
# --- Votre code ici ---
# ============================================================
# country_risk doit être entre 0 et 100
# ============================================================
mask_country = (df["country_risk"] < 0) | (df["country_risk"] > 100)
logger.info(f"country_risk hors plage : {mask_country.sum()}")

# NaN (donnée incohérente)
df.loc[mask_country, "country_risk"] = np.nan

# ============================================================
# volatility_30d doit être entre 0.1 et 200
# ============================================================
mask_vol = (df["volatility_30d"] < 0.1) | (df["volatility_30d"] > 200)
logger.info(f"volatility_30d hors plage : {mask_vol.sum()}")

df.loc[mask_vol, "volatility_30d"] = np.nan

# ============================================================
# default_flag doit être 0 ou 1
# ============================================================
mask_default = ~df["default_flag"].isin([0, 1])
logger.info(f"default_flag invalide : {mask_default.sum()}")

df.loc[mask_default, "default_flag"] = np.nan

# ============================================================
# quantity doit être > 0
# ============================================================
mask_qty = df["quantity"] <= 0
logger.info(f"quantity invalide : {mask_qty.sum()}")
df = df[~mask_qty]

# ============================================================
# JUSTIFICATION METIER :
#
# - country_risk : score normalisé 0–100 → sinon incohérent
# - volatility : ne peut pas être <=0 ou extrême → NaN
# - default_flag : variable binaire réglementaire
# - quantity <= 0 : transaction invalide → suppression
# ============================================================

after = len(df)
logger.info(f"[Regles metier] {before} -> {after} lignes (supprimés : {before - after})")
logger.info(f"[Regles metier] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 7 — Detection et traitement des outliers
# ============================================================
# CONSIGNE :
# Appliquez la methode IQR sur : notional_eur, volatility_30d, volume_j
#
# Pour chaque colonne :
#   1. Calculez Q1, Q3, IQR, lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR
#   2. Comptez et affichez le nb d'outliers detectes
#   3. Choisissez une strategie (suppression / winsorisation / flaggage)
#      et justifiez-la en commentaire avec une raison METIER
#   4. Produisez un boxplot pour chaque colonne (avant traitement)
#
# Appliquez ensuite Isolation Forest sur les colonnes :
# [price, volume_j, volatility_30d, notional_eur]
# pour detecter les anomalies multivariees.
# Ajoutez une colonne "is_anomaly_multivariate" (0/1).
# Ne supprimez PAS ces lignes — le Risk Officer doit les examiner.
#
# LIBRAIRIES : scipy.stats, sklearn.ensemble.IsolationForest

before = len(df)
# --- Votre code ici ---
from sklearn.ensemble import IsolationForest

# ============================================================
# Détection des outliers avec IQR
# ============================================================

cols_outliers = ["notional_eur", "volatility_30d", "volume_j"]

for col in cols_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask_outliers = (df[col] < lower) | (df[col] > upper)
    
    logger.info(f"{col} - outliers détectés : {mask_outliers.sum()}")

    # Boxplot (avant traitement)
    plt.figure()
    df.boxplot(column=col)
    plt.title(f"Boxplot {col}")
    plt.show()

    # ============================================================
    # STRATEGIE : winsorisation
    # (on limite les valeurs extrêmes sans supprimer)
    # ============================================================
    df.loc[df[col] < lower, col] = lower
    df.loc[df[col] > upper, col] = upper


# ============================================================
# Isolation Forest (anomalies multivariées)
# ============================================================

features = ["price", "volume_j", "volatility_30d", "notional_eur"]

# On drop les NaN pour le modèle (sinon erreur)
df_if = df[features].dropna()

iso = IsolationForest(contamination=0.02, random_state=42)
preds = iso.fit_predict(df_if)

# -1 = anomalie, 1 = normal → on transforme en 0/1
df.loc[df_if.index, "is_anomaly_multivariate"] = (preds == -1).astype(int)

# Remplir les NaN (lignes non évaluées)
df["is_anomaly_multivariate"] = df["is_anomaly_multivariate"].fillna(0)

logger.info(f"Anomalies multivariées détectées : {(df['is_anomaly_multivariate'] == 1).sum()}")

# ============================================================
# JUSTIFICATION METIER :
#
# - IQR : détecte les valeurs extrêmes univariées
# - Winsorisation : préférable à suppression en finance
#   (on conserve l'information économique)
#
# - Isolation Forest :
#   détecte des anomalies complexes (prix + volume + volatilité)
#
# - On NE SUPPRIME PAS ces anomalies :
#   elles doivent être analysées par le Risk Officer
# ============================================================

after = len(df)
logger.info(f"[Outliers] {before} -> {after} lignes")
logger.info(f"[Outliers] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 8 — Traitement des valeurs manquantes
# ============================================================
# CONSIGNE :
# Definissez une strategie par colonne. Regle generale :
#   < 20% NaN  : imputer (mediane pour numerique, mode pour categoriel)
#                + creer une colonne flag "colonne_was_missing" (0/1)
#   20%-70% NaN: imputer + flag (idem)
#   > 70% NaN  : supprimer la colonne
#
# Cas particuliers a justifier :
#   - settlement_date : quelle strategie pour les NaT ?
#   - credit_rating   : imputer le mode est-il pertinent pour un rating ?
#   - trade_id        : que faire si un trade_id est NaN ?
#
# Loggez la strategie choisie et le taux de NaN avant/apres pour chaque colonne.

before = len(df)
# --- Votre code ici ---
# ============================================================
# Analyse du taux de NaN par colonne
# ============================================================

nan_pct = df.isna().mean()

for col in df.columns:
    pct = nan_pct[col]

    # ========================================================
    # CAS > 70% → suppression colonne
    # ========================================================
    if pct > 0.7:
        logger.info(f"{col} supprimée (>70% NaN : {pct:.2%})")
        df.drop(columns=[col], inplace=True)

    # ========================================================
    # CAS entre 0% et 70% → imputation + flag
    # ========================================================
    elif pct > 0:
        logger.info(f"{col} imputée ({pct:.2%} NaN)")

        # Création du flag
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

        # NUMERIQUE → médiane
        if df[col].dtype in ["float64", "int64"]:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

        # CATEGORIEL → mode
        else:
            mode_value = df[col].mode(dropna=True)
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna("unknown")

# ============================================================
# CAS SPECIFIQUES METIER
# ============================================================

# settlement_date → si manquant, on applique T+2
mask_settle = df["settlement_date"].isna()
logger.info(f"settlement_date NaT corrigés : {mask_settle.sum()}")

df.loc[mask_settle, "settlement_date"] = df.loc[mask_settle, "trade_date"] + pd.Timedelta(days=2)

# credit_rating → déjà sensible → garder NaN aurait été possible
# MAIS ici on impute pour permettre le modeling (avec flag déjà créé)

# trade_id → ne doit JAMAIS être NaN
missing_trade_id = df["trade_id"].isna().sum()
logger.info(f"trade_id manquants : {missing_trade_id}")

# suppression si jamais ça arrive (clé métier critique)
df = df[df["trade_id"].notna()]

# ============================================================
# JUSTIFICATION METIER :
#
# - Flags permettent au modèle de savoir qu’une donnée était manquante
# - Médiane robuste aux outliers → préférable en finance
# - Mode pour catégories → valeur la plus probable
#
# - settlement_date : reconstruction logique T+2
# - trade_id : clé métier → jamais imputée
# ============================================================

after = len(df)
logger.info(f"[Valeurs manquantes] {before} -> {after} lignes")
logger.info(f"[Valeurs manquantes] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 9 — Pseudonymisation RGPD / BCBS 239
# ============================================================
# CONSIGNE :
# Identifiez toutes les colonnes contenant des donnees PII
# (Personally Identifiable Information) ou des donnees sensibles.
#
# Pour chaque colonne PII :
#   1. Creez une colonne "colonne_hash" avec un hash SHA-256 irreversible
#   2. Supprimez la colonne originale
#
# Le salt doit etre lu depuis une variable d'environnement :
#   salt = os.environ.get('CLEANSE_SALT', 'default_salt_dev')
# Ne jamais hardcoder le salt dans le code.
#
# Justifiez dans un commentaire quelles colonnes sont des PII
# et pourquoi (reference a l'article RGPD correspondant).
#
# LIBRAIRIE : hashlib

before = len(df)
# --- Votre code ici ---
import hashlib

# Récupération du salt (sécurité)
salt = os.environ.get('CLEANSE_SALT', 'default_salt_dev')

# Colonnes PII identifiées
pii_columns = ["counterparty_name", "counterparty_id", "trader_id"]

for col in pii_columns:
    if col in df.columns:
        logger.info(f"Pseudonymisation de la colonne : {col}")

        # Création du hash SHA-256
        df[f"{col}_hash"] = df[col].astype(str).apply(
            lambda x: hashlib.sha256((x + salt).encode()).hexdigest()
        )

        # Suppression colonne originale
        df.drop(columns=[col], inplace=True)

# ============================================================
# JUSTIFICATION METIER :
#
# - RGPD (article 4 + 32) → protection des données personnelles
# - Hash SHA-256 → irréversible → conforme sécurité
# - Salt → protège contre attaques dictionnaire
#
# - counterparty_name → identification entreprise
# - trader_id → donnée interne sensible
# - counterparty_id → identifiant client
# ============================================================

after = len(df)
logger.info(f"[Pseudonymisation RGPD] {before} -> {after} lignes")
logger.info(f"[Pseudonymisation RGPD] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 10 — Rapport de qualite final
# ============================================================
# CONSIGNE :
# Produisez un rapport avant/apres comparant :
#   - Nb de lignes et colonnes
#   - Taux de completude global
#   - Nb de doublons restants
#   - Recapitulatif de chaque etape (nb lignes supprimees / modifiees)
#
# Calculez le Data Quality Score (DQS) selon la formule :
#   DQS = (completude * 0.6 + unicite * 0.4) * 100
#   ou completude = 1 - taux_nan_global
#   et unicite    = 1 - taux_doublons
#
# Sauvegardez le dataset nettoye.

# --- Votre code ici ---


df.to_csv('data/tradecleanse_clean.csv', index=False)
logger.info(f"Dataset nettoye sauvegarde : tradecleanse_clean.csv")
print(f"Shape finale : {df.shape}")
