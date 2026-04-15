# ============================================================
# TRADECLEANSE — NOTEBOOK 04 : Bonus Expert
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================
#
# Ce notebook contient 3 bonus independants.
# Chaque bonus vaut +1 point au-dela de 20.
# Lisez attentivement chaque consigne avant de coder.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df_raw   = pd.read_csv('data/tradecleanse_raw.csv',   low_memory=False)
df_clean = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)
df_clean['trade_date'] = pd.to_datetime(df_clean['trade_date'], errors='coerce')

# ============================================================
# BONUS 1 — Detection de Wash Trading (+1 pt)
# ============================================================
#
# Le wash trading est une forme de manipulation de marche consistant
# a acheter et vendre le meme instrument a soi-meme pour gonfler
# artificiellement les volumes.
#
# Contexte reglementaire : interdit par l'article 12 du Reglement
# europeen MAR (Market Abuse Regulation).
#
# TACHE :
# Detectez dans le dataset les paires de transactions suspectes
# repondant aux criteres suivants SIMULTANEMENT :
#   - Meme ISIN (meme instrument)
#   - Meme trader (trader_id_hash)
#   - Meme date de trade
#   - Quantites quasi-identiques (ecart < 5%)
#   - Prix quasi-identiques (ecart < 0.1%)
#
# LIVRABLE :
#   - Un DataFrame "wt_suspects" listant toutes les paires detectees
#     avec : trade_id_1, trade_id_2, isin, trader_hash,
#            trade_date, delta_price_%, delta_qty_%
#   - Un court commentaire expliquant pourquoi ces criteres
#     caracterisent un wash trading
#   - Sauvegarde dans wash_trading_suspects.csv
#
# ATTENTION : vous travaillez sur df_clean (trader_id est pseudonymise).

# --- Votre code ici ---
# BONUS 1 — Detection Wash Trading

df = df_clean.copy()

# On trie pour faciliter la comparaison
df = df.sort_values(by=["isin", "trader_id_hash", "trade_date"])

suspects = []

for i in range(len(df)-1):
    row1 = df.iloc[i]
    row2 = df.iloc[i+1]

    # Conditions
    if (
        row1["isin"] == row2["isin"] and
        row1["trader_id_hash"] == row2["trader_id_hash"] and
        row1["trade_date"] == row2["trade_date"]
    ):
        # Calcul écarts
        delta_price = abs(row1["price"] - row2["price"]) / row1["price"]
        delta_qty   = abs(row1["quantity"] - row2["quantity"]) / row1["quantity"]

        if delta_price < 0.001 and delta_qty < 0.05:
            suspects.append({
                "trade_id_1": row1["trade_id"],
                "trade_id_2": row2["trade_id"],
                "isin": row1["isin"],
                "trader_hash": row1["trader_id_hash"],
                "trade_date": row1["trade_date"],
                "delta_price_%": delta_price * 100,
                "delta_qty_%": delta_qty * 100
            })

wt_suspects = pd.DataFrame(suspects)

print(f"Nombre de cas suspects : {len(wt_suspects)}")

wt_suspects.to_csv("data/wash_trading_suspects.csv", index=False)

# ============================================================
# BONUS 2 — Data Drift Monitoring (+1 pt)
# ============================================================
#
# Le data drift designe le phenomene par lequel la distribution
# statistique des donnees evolue dans le temps, rendant un modele
# ML entraine sur des donnees passees moins performant sur des
# donnees recentes.
#
# En finance : un changement de regime de volatilite (ex: crise),
# une variation de politique monetaire ou un choc de marche peuvent
# provoquer un drift significatif.
#
# TACHE :
# Divisez le dataset en deux periodes :
#   - Periode 1 (early) : premiers 90 jours
#   - Periode 2 (late)  : derniers 90 jours
#
# Pour chaque variable numerique cle (price, volatility_30d,
# notional_eur, volume_j, country_risk) :
#   1. Appliquez le test de Kolmogorov-Smirnov (scipy.stats.ks_2samp)
#   2. Si p-value < 0.05 : flaguer comme "drift detecte"
#   3. Produisez un graphique avec les distributions early vs late
#      pour chaque variable
#
# LIVRABLE :
#   - Un tableau recapitulatif : variable | KS stat | p-value | drift O/N
#   - Le graphique sauvegarde dans 04_drift_monitor.png
#   - drift_report.csv
#
# LIBRAIRIE : from scipy.stats import ks_2samp

# --- Votre code ici ---
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# ================================
# 1. Création des périodes
# ================================

df_clean = df_clean.sort_values("trade_date")

date_min = df_clean["trade_date"].min()
date_max = df_clean["trade_date"].max()

early = df_clean[df_clean["trade_date"] <= date_min + pd.Timedelta(days=90)]
late  = df_clean[df_clean["trade_date"] >= date_max - pd.Timedelta(days=90)]

print(f"Période early : {early.shape}")
print(f"Période late  : {late.shape}")

# ================================
# 2. Variables à analyser
# ================================

variables = ["price", "volatility_30d", "notional_eur", "volume_j", "country_risk"]

results = []

plt.figure(figsize=(15, 10))

# ================================
# 3. Test KS + Graphiques
# ================================

for i, col in enumerate(variables, 1):

    early_data = early[col].dropna()
    late_data  = late[col].dropna()

    ks_stat, p_value = ks_2samp(early_data, late_data)

    drift = "OUI" if p_value < 0.05 else "NON"

    results.append([col, ks_stat, p_value, drift])

    # Graphique
    plt.subplot(3, 2, i)
    plt.hist(early_data, bins=30, alpha=0.5, label="Early")
    plt.hist(late_data, bins=30, alpha=0.5, label="Late")
    plt.title(f"{col} | Drift: {drift}")
    plt.legend()

# ================================
# 4. Sauvegarde graphique
# ================================

plt.tight_layout()
plt.savefig("data/04_drift_monitor.png")
plt.show()

# ================================
# 5. Rapport
# ================================

drift_df = pd.DataFrame(results, columns=["variable", "ks_stat", "p_value", "drift"])

print("\n===== RAPPORT DRIFT =====")
print(drift_df)

drift_df.to_csv("data/drift_report.csv", index=False)
print("\nFichier sauvegardé : data/drift_report.csv")

# ============================================================
# BONUS 3 — Impact du nettoyage sur le modele ML (+1 pt)
# ============================================================
#
# L'argument ultime pour justifier le data cleansing aupres
# d'un Risk Officer ou d'un CTO est de montrer QUANTITATIVEMENT
# que le nettoyage ameliore les performances du modele.
#
# TACHE :
# Entrainez un modele Random Forest pour predire default_flag.
# Faites-le UNE FOIS sur df_raw et UNE FOIS sur df_clean.
# Comparez les metriques sur le jeu de test.
#
# Colonnes features a utiliser (disponibles dans les deux datasets) :
#   price, quantity, bid, ask, mid_price,
#   volume_j, volatility_30d, country_risk
#
# Etapes :
#   1. Preparez X et y pour chaque dataset
#      (gerer les NaN restants avec fillna ou imputation simple)
#   2. Split train/test 80/20 avec stratify=y et random_state=42
#   3. Entrainement : RandomForestClassifier(n_estimators=150,
#                     max_depth=6, random_state=42)
#   4. Metriques : AUC-ROC, precision, rappel, F1 sur la classe 1
#   5. Tracez les deux courbes ROC sur le meme graphique
#
# LIVRABLE :
#   - Tableau comparatif : Dataset | AUC-ROC | Precision | Rappel | F1
#   - Graphique 04_roc_comparison.png
#   - model_comparison.csv
#   - 3-5 phrases analysant le resultat :
#     * Le nettoyage ameliore-t-il le modele ? De combien ?
#     * Si le gain est faible, quelle en est la raison probable ?
#     * Que faudrait-il faire pour ameliorer davantage ?
#
# LIBRAIRIES : sklearn.ensemble, sklearn.metrics, sklearn.model_selection

# --- Votre code ici ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve

# ================================
# 1. Features
# ================================

features = ["price", "quantity", "bid", "ask", "mid_price",
            "volume_j", "volatility_30d", "country_risk"]

target = "default_flag"

# ================================
# 2. Préparation datasets 
# ================================

def prepare_data(df):
    X = df[features].copy()
    y = df[target].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # remplissage des NaN avec la médiane
    X = X.fillna(X.median())

    return X, y

X_raw, y_raw = prepare_data(df_raw)
X_clean, y_clean = prepare_data(df_clean)

# ================================
# 3. Split
# ================================

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
)

# ================================
# 4. Modèle
# ================================

model_raw = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model_clean = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

model_raw.fit(Xr_train, yr_train)
model_clean.fit(Xc_train, yc_train)

# ================================
# 5. Prédictions
# ================================

yr_pred = model_raw.predict(Xr_test)
yc_pred = model_clean.predict(Xc_test)

yr_proba = model_raw.predict_proba(Xr_test)[:, 1]
yc_proba = model_clean.predict_proba(Xc_test)[:, 1]

# ================================
# 6. Metrics
# ================================

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

metrics_raw = compute_metrics(yr_test, yr_pred, yr_proba)
metrics_clean = compute_metrics(yc_test, yc_pred, yc_proba)

# ================================
# 7. Tableau comparatif
# ================================

comparison = pd.DataFrame([
    ["RAW", metrics_raw["AUC"], metrics_raw["Precision"], metrics_raw["Recall"], metrics_raw["F1"]],
    ["CLEAN", metrics_clean["AUC"], metrics_clean["Precision"], metrics_clean["Recall"], metrics_clean["F1"]],
], columns=["Dataset", "AUC-ROC", "Precision", "Recall", "F1"])

print("\n===== COMPARAISON MODELE =====")
print(comparison)

comparison.to_csv("data/model_comparison.csv", index=False)

# ================================
# 8. Courbe ROC
# ================================

fpr_raw, tpr_raw, _ = roc_curve(yr_test, yr_proba)
fpr_clean, tpr_clean, _ = roc_curve(yc_test, yc_proba)

plt.figure()
plt.plot(fpr_raw, tpr_raw, label="RAW")
plt.plot(fpr_clean, tpr_clean, label="CLEAN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("data/04_roc_comparison.png")
plt.show()

print("\nFichiers générés :")
print("- data/model_comparison.csv")
print("- data/04_roc_comparison.png")