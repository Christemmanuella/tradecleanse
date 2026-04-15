# TRADECLEANSE — Rapport de Qualité de Données
## QuantAxis Capital — Audit, Nettoyage et Certification


# 1. Contexte du projet

QuantAxis Capital est un hedge fund exploitant des stratégies de trading algorithmique basées sur des modèles de machine learning. Un dataset consolidé provenant de plusieurs sources (Bloomberg, Murex, Refinitiv) a été utilisé pour entraîner un modèle de scoring de risque de contrepartie.

Cependant, des résultats incohérents ont été observés (prédictions de risque nul pour des contreparties ayant fait défaut), indiquant des problèmes de qualité de données.

L'objectif de ce projet est donc de :
- Auditer le dataset brut
- Nettoyer les données selon des règles métier
- Garantir la qualité et la traçabilité des transformations
- Valider le dataset final avant utilisation en machine learning

---

# 2. Description du dataset

Le dataset contient environ 8 950 lignes et 20 colonnes représentant des transactions financières.

### Identification des transactions
- **trade_id** : identifiant unique de chaque transaction  
- **isin** : code ISIN permettant d’identifier l’instrument financier  

### Informations sur la contrepartie
- **counterparty_id** : identifiant interne de la contrepartie  
- **counterparty_name** : nom de la contrepartie (donnée sensible – PII)  
- **credit_rating** : notation de crédit de la contrepartie  
- **default_flag** : indicateur de défaut (0 = non, 1 = oui)  

### Informations temporelles
- **trade_date** : date à laquelle la transaction a été effectuée  
- **settlement_date** : date de règlement de la transaction  

### Caractéristiques financières
- **asset_class** : classe d’actif (equity, bond, derivative, fx)  
- **notional_eur** : montant notionnel de la transaction  
- **price** : prix d’exécution  
- **quantity** : quantité échangée  

### Données de marché
- **bid / ask** : fourchette de prix du marché  
- **mid_price** : prix médian théorique calculé  
- **volume_j** : volume journalier de l’instrument  
- **volatility_30d** : volatilité sur 30 jours  

### Informations de risque
- **country_risk** : indicateur de risque pays  
- **sector** : secteur d’activité  

### Données internes
- **trader_id** : identifiant du trader (donnée sensible – PII)



---

# 3. Profiling initial

Une analyse exploratoire a été réalisée sur le dataset brut.

## Résultats principaux :

- Nombre de lignes : 8950
- Nombre de colonnes : 20
- Nombre de NaN initial : 2501
- Présence de valeurs sentinelles : "N/A", "#N/A", "-", "99999", "0.0"
- Incohérences de type (colonnes numériques en string)
- Variabilité importante sur certaines variables (notional_eur notamment)

Des visualisations ont été produites :
- Boxplots (notional, volume, volatility)
- Distribution du délai de settlement

---

# 4. Anomalies détectées

Le dataset contient plusieurs anomalies de nature différente.

## 4.1 Anomalies classiques

- 200 doublons exacts sur `trade_id`
- Valeurs sentinelles remplacées par NaN
- Incohérences de casse sur `asset_class`
- Valeurs manquantes sur plusieurs colonnes

## 4.2 Anomalies métier

- settlement_date < trade_date (80 cas)
- bid > ask (120 cas)
- mid_price incohérent (297 cas)
- price hors fourchette (150 cas)
- notional_eur négatif (40 cas)

## 4.3 Anomalies avancées

- credit_rating élevé avec default_flag = 1 (96 cas)
- outliers importants sur notional_eur (1018 cas)
- anomalies multivariées détectées (151 cas)

---

# 5. Pipeline de nettoyage

Le pipeline de nettoyage a été structuré en plusieurs étapes.

## 5.1 Remplacement des valeurs sentinelles

Toutes les valeurs sentinelles ont été remplacées par NaN afin d'uniformiser le traitement.

## 5.2 Suppression des doublons

Suppression des doublons sur `trade_id` en conservant la première occurrence (considérée comme la plus fiable dans un contexte Murex).

Résultat :
- 200 lignes supprimées

## 5.3 Conversion des types

- Dates converties en datetime
- Variables numériques converties avec `errors='coerce'`
- Variables catégorielles normalisées (lowercase + strip)

## 5.4 Normalisation de asset_class

Mapping vers les 4 valeurs autorisées :
- equity
- bond
- derivative
- fx

945 valeurs invalides corrigées.

## 5.5 Correction des incohérences métier

- settlement_date corrigée si antérieure à trade_date
- bid et ask échangés si incohérents
- mid_price recalculé
- price ajusté dans la fourchette bid/ask
- notional rendu positif
- correction des contradictions rating/défaut

## 5.6 Règles métier

- country_risk borné entre 0 et 100
- volatility contrôlée
- quantity > 0

## 5.7 Détection des outliers

- Méthode IQR appliquée
- 1018 outliers détectés sur notional
- Isolation Forest utilisé pour anomalies multivariées

## 5.8 Gestion des valeurs manquantes

Stratégie appliquée :
- <20% : imputation (médiane/mode)
- création de flags de missing

## 5.9 Pseudonymisation

Colonnes sensibles hashées :
- counterparty_name
- counterparty_id
- trader_id

---

# 6. Rapport de qualité

## Résultats :

- Lignes initiales : 8950
- Lignes finales : 8750
- Colonnes finales : 25
- Taux de complétude : 100%
- Doublons restants : 0

## Data Quality Score

DQS = 100 / 100

---

# 7. Validation du dataset

14 tests ont été implémentés :

- unicité de trade_id
- cohérence des dates
- bid < ask
- price dans la fourchette
- mid_price cohérent
- asset_class valide
- pas de contradiction rating/défaut
- notional positif
- country_risk valide
- format ISIN valide
- volatility valide
- complétude > 90%
- absence de PII

Résultat :

Score : 14/14 tests passés

---

# 8. Bonus

## 8.1 Wash trading

7 cas suspects détectés selon :
- même ISIN
- même trader
- même date
- prix et quantités quasi identiques

## 8.2 Data drift

Aucun drift détecté (p-value > 0.05 pour toutes les variables)

## 8.3 Impact ML

Comparaison modèle Random Forest :

- dataset brut vs nettoyé
- courbes ROC similaires
- amélioration marginale

Conclusion :
Le nettoyage stabilise les données mais le gain ML dépend de la qualité des features.

---

# 9. Conclusion

Le pipeline de nettoyage a permis de :

- Corriger toutes les anomalies identifiées
- Garantir la cohérence métier des données
- Améliorer la qualité globale du dataset
- Fournir un dataset prêt pour un usage en production

Le dataset final est conforme aux exigences de qualité et peut être utilisé en toute confiance pour l’entraînement de modèles de machine learning.

---