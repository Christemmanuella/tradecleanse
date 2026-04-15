# ============================================================
# TRADECLEANSE — NOTEBOOK 03 : Validation du Dataset Nettoye
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================
#
# CONSIGNE GENERALE :
# Ce notebook valide que votre pipeline a correctement nettoye le dataset.
# Vous devez implementer au minimum 14 tests de validation (expectations).
#
# Deux approches possibles :
#   A) Utiliser la librairie Great Expectations (recommande en entreprise)
#      pip install great_expectations
#      Documentation : https://docs.greatexpectations.io
#
#   B) Implementer vos propres tests avec pandas + assertions Python
#      (acceptable si vous documentez clairement chaque test)
#
# Pour chaque test, affichez clairement : [PASS] ou [FAIL] + le detail.
# A la fin, affichez un score : X/14 tests passes.
# ============================================================

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Chargement du dataset nettoye
df = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)
print(f"Dataset nettoye charge : {df.shape[0]} lignes x {df.shape[1]} colonnes\n")

# ============================================================
# FONCTION LOG (AJOUT)
# ============================================================

results = []

def log_test(name, condition, detail=""):
    if condition:
        print(f"[PASS] {name}")
        results.append((name, "PASS", detail))
    else:
        print(f"[FAIL] {name} ❌ {detail}")
        results.append((name, "FAIL", detail))

# ============================================================
# EXPECTATION 1 — Unicite de trade_id
# ============================================================

duplicates = df["trade_id"].duplicated().sum()
log_test("Unicité trade_id", duplicates == 0, f"{duplicates} doublons")

# ============================================================
# EXPECTATION 2 — Colonnes obligatoires non nulles
# ============================================================

cols = ["trade_id", "isin", "trade_date", "asset_class", "price", "quantity", "default_flag"]
missing = df[cols].isna().sum().sum()
log_test("Colonnes obligatoires non nulles", missing == 0, f"{missing} NaN")

# ============================================================
# EXPECTATION 3 — settlement_date >= trade_date
# ============================================================

invalid_dates = (df["settlement_date"] < df["trade_date"]).sum()
log_test("Dates cohérentes", invalid_dates == 0, f"{invalid_dates} erreurs")

# ============================================================
# EXPECTATION 4 — bid < ask sur toutes les lignes
# ============================================================

invalid_spread = (df["bid"] >= df["ask"]).sum()
log_test("bid < ask", invalid_spread == 0, f"{invalid_spread} erreurs")

# ============================================================
# EXPECTATION 5 — price dans la fourchette
# ============================================================

invalid_price = ((df["price"] < df["bid"] * 0.995) | 
                 (df["price"] > df["ask"] * 1.005)).sum()
log_test("Price dans fourchette", invalid_price == 0, f"{invalid_price} erreurs")

# ============================================================
# EXPECTATION 6 — mid_price coherent
# ============================================================

mid_calc = (df["bid"] + df["ask"]) / 2
diff = abs(df["mid_price"] - mid_calc) / mid_calc
invalid_mid = (diff > 0.01).sum()
log_test("Mid price cohérent", invalid_mid == 0, f"{invalid_mid} erreurs")

# ============================================================
# EXPECTATION 7 — asset_class valide
# ============================================================

valid_classes = ["equity", "bond", "derivative", "fx"]
invalid_asset = (~df["asset_class"].isin(valid_classes)).sum()
log_test("Asset class valide", invalid_asset == 0, f"{invalid_asset} erreurs")

# ============================================================
# EXPECTATION 8 — Pas de contradiction rating + defaut
# ============================================================

invalid_rating = df[
    (df["credit_rating"].isin(["aaa", "aa", "a"])) & (df["default_flag"] == 1)
].shape[0]
log_test("Pas de contradiction rating/défaut", invalid_rating == 0, f"{invalid_rating} erreurs")

# ============================================================
# EXPECTATION 9 — notional_eur > 0
# ============================================================

invalid_notional = (df["notional_eur"] <= 0).sum()
log_test("Notional positif", invalid_notional == 0, f"{invalid_notional} erreurs")

# ============================================================
# EXPECTATION 10 — country_risk dans [0, 100]
# ============================================================

invalid_risk = ((df["country_risk"] < 0) | (df["country_risk"] > 100)).sum()
log_test("Country risk valide", invalid_risk == 0, f"{invalid_risk} erreurs")

# ============================================================
# EXPECTATION 11 — Format ISIN valide
# ============================================================

isin_pattern = r'^[A-Z]{2}[A-Z0-9]{10}$'
invalid_isin = (~df["isin"].astype(str).str.match(isin_pattern)).sum()
log_test("ISIN valide", invalid_isin == 0, f"{invalid_isin} erreurs")

# ============================================================
# EXPECTATION 12 — volatility_30d dans [0.1, 200]
# ============================================================

invalid_vol = ((df["volatility_30d"] < 0.1) | (df["volatility_30d"] > 200)).sum()
log_test("Volatility valide", invalid_vol == 0, f"{invalid_vol} erreurs")

# ============================================================
# EXPECTATION 13 — Completude globale > 90%
# ============================================================

completeness = 1 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
log_test("Complétude > 90%", completeness > 0.9, f"{round(completeness*100,2)}%")

# ============================================================
# EXPECTATION 14 — Absence de PII en clair
# ============================================================

pii_columns = ["counterparty_name", "counterparty_id", "trader_id"]
pii_present = any(col in df.columns for col in pii_columns)
log_test("Pas de PII en clair", not pii_present)

# ============================================================
# SCORE FINAL
# ============================================================

passed = sum(1 for r in results if r[1] == "PASS")
total = len(results)

print("\n================ RESULTATS =================")
print(f"Score : {passed}/{total} expectations passees")

report = pd.DataFrame(results, columns=["test", "status", "detail"])
report.to_csv("data/ge_validation_report.csv", index=False)

print("Rapport exporte : data/ge_validation_report.csv")