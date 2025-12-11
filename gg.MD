# 📘 GRAND GUIDE : MARKET TREND & EXTERNAL FACTORS ANALYSIS  
*Version approfondie avec XGBoost*

---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)

Les marchés évoluent rapidement sous l’effet de facteurs macroéconomiques, technologiques et politiques.  
Analyser les tendances devient essentiel pour anticiper les risques et orienter les décisions stratégiques.

**Objectif :**  
- Étudier l’évolution du marché  
- Comprendre les facteurs externes  
- Construire un modèle prédictif basé sur XGBoost

---

## 2. Le Code Python (Laboratoire)

Ce script illustre comment charger, nettoyer, analyser et **modéliser** les tendances du marché.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- PHASE 1 : IMPORTATION DES DONNÉES ---
df = pd.DataFrame({
    "annee": np.arange(2015, 2026),
    "croissance_PIB": [3.2, 3.1, 3.0, 2.8, 2.5, 1.9, 1.2, 0.5, -1.0, 2.0, 2.7],
    "inflation":       [1.5, 1.7, 1.8, 2.0, 2.3, 3.0, 5.2, 6.7, 4.5, 3.1, 2.4],
    "croissance_secteur": [4.1, 4.3, 4.0, 3.9, 3.7, 3.0, 2.0, 1.2, -0.8, 1.5, 2.6]
})

# --- PHASE 2 : NETTOYAGE ---
df_clean = df.fillna(method="ffill")

# --- PHASE 3 : EXPLORATION ---
print("Statistiques descriptives :")
print(df_clean.describe())

plt.plot(df_clean["annee"], df_clean["croissance_secteur"])
plt.title("Évolution du secteur")
plt.grid(True)
plt.show()

# --- PHASE 4 : SPLIT DES DONNÉES ---
X = df_clean[["croissance_PIB", "inflation"]]
y = df_clean["croissance_secteur"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (XGBOOST) ---
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- PHASE 6 : ÉVALUATION ---
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.3f}")
print(f"R² Score : {r2:.3f}")

# Importance des features
plt.bar(X.columns, model.feature_importances_)
plt.title("Importance des variables pour XGBoost")
plt.show()
```

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Le Problème
Les données économiques et sectorielles contiennent souvent :
- des valeurs manquantes  
- des fluctuations irrégulières  
- des anomalies dues à des crises

### Stratégie adoptée
L’approche `ffill` conserve la cohérence temporelle, indispensable pour une analyse économique réaliste.

---

## 4. Analyse Exploratoire

Cette étape dévoile les tendances importantes :
- Baisse progressive de la croissance du PIB  
- Inflation élevée → pression sur la demande  
- Forte corrélation entre PIB et croissance du secteur

`.describe()` aide à détecter :
- la volatilité  
- les années extrêmes  
- les cycles du marché  

---

## 5. Méthodologie : Split & Prédiction

Comme dans les projets ML :
- On sépare les données pour **généraliser** (80% / 20%).  
- On utilise des variables explicatives macroéconomiques pour prédire la croissance du secteur.

---

## 6. FOCUS THÉORIQUE : L’Algorithme XGBoost 🚀

Pourquoi XGBoost ?

### A. Gradient Boosting Optimisé  
XGBoost construit des arbres **séquentiellement**, chaque arbre corrige les erreurs du précédent.

### B. Avantages face aux modèles économiques classiques  
- Gère les non-linéarités  
- Résistant au surapprentissage  
- Capable d’extraire les tendances cachées  
- Très performant même avec peu de données  

### C. Feature Importance  
Le modèle révèle quelles variables affectent le plus la tendance sectorielle :
- inflation  
- croissance du PIB  

Cela imite un raisonnement d’économiste, mais avec une précision algorithmique.

---

## 7. Analyse des Résultats

### MAE (Mean Absolute Error)
Mesure l’erreur moyenne du modèle → plus c’est faible, meilleur est le modèle.

### R² Score
Mesure la capacité explicative du modèle :
- 1.0 = prédiction parfaite  
- 0 = pas mieux qu’une moyenne  

### Importance des variables
L’Inflation peut dominer l’explication si le secteur est sensible au pouvoir d’achat.  
Le PIB explique souvent la demande globale.

---

## Conclusion du Projet

L’ajout de **XGBoost** montre comment l’intelligence artificielle peut renforcer l’analyse économique.  
Cette approche permet d’anticiper les variations du marché et de comprendre quels facteurs externes influencent réellement le secteur.

