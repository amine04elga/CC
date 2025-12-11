"""
═══════════════════════════════════════════════════════════════════
📊 GRAND GUIDE : ANALYSE DES TENDANCES DE MARCHÉ & FACTEURS EXTERNES
═══════════════════════════════════════════════════════════════════

Ce script génère un rapport complet d'analyse de données financières
dans le style pédagogique d'un Data Scientist expert.

Auteur : Analyse automatisée
Objectif : Prédire les tendances de marché en fonction de facteurs externes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, classification_report, confusion_matrix)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration esthétique
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*70)
print("📘 ANATOMIE D'UN PROJET DATA SCIENCE : ANALYSE DE MARCHÉ")
print("="*70)
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 : CONTEXTE MÉTIER ET MISSION
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("1️⃣  CONTEXTE MÉTIER : LA MISSION")
print("─" * 70)
print()
print("📌 LE PROBLÈME (BUSINESS CASE)")
print("   Dans le monde de la finance et du trading, les décisions d'investissement")
print("   reposent sur la compréhension des tendances de marché et des facteurs")
print("   externes (économiques, politiques, sociaux).")
print()
print("   🎯 Objectif : Créer un modèle prédictif pour anticiper les mouvements")
print("      de marché en analysant des indicateurs externes.")
print()
print("   ⚠️  L'Enjeu Critique : ")
print("      • Faux Positif (prédire une hausse qui n'arrive pas) → Perte financière")
print("      • Faux Négatif (manquer une opportunité) → Manque à gagner")
print("      → L'IA doit optimiser le ratio risque/rendement")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 2 : ACQUISITION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("2️⃣  ACQUISITION & CHARGEMENT DES DONNÉES")
print("─" * 70)
print()

try:
    import kagglehub
    print("📥 Téléchargement du dataset depuis Kaggle...")
    path = kagglehub.dataset_download("kundanbedmutha/market-trend-and-external-factors-dataset")
    print(f"✅ Dataset téléchargé dans : {path}")
    
    # Recherche du fichier CSV
    import os
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if csv_files:
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        print(f"✅ Fichier chargé : {csv_files[0]}")
    else:
        raise FileNotFoundError("Aucun fichier CSV trouvé")
        
except Exception as e:
    print(f"⚠️  Erreur lors du téléchargement : {e}")
    print("📝 Génération de données synthétiques pour démonstration...")
    
    # Création de données synthétiques réalistes
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'Market_Index': np.cumsum(np.random.randn(n_samples) * 2 + 0.05) + 1000,
        'GDP_Growth': np.random.uniform(1.5, 4.5, n_samples),
        'Inflation_Rate': np.random.uniform(1.0, 5.0, n_samples),
        'Interest_Rate': np.random.uniform(0.5, 3.5, n_samples),
        'Unemployment_Rate': np.random.uniform(3.0, 8.0, n_samples),
        'Consumer_Confidence': np.random.uniform(80, 120, n_samples),
        'Oil_Price': np.random.uniform(40, 100, n_samples),
        'Gold_Price': np.random.uniform(1500, 2000, n_samples),
        'USD_Exchange_Rate': np.random.uniform(0.85, 1.15, n_samples),
        'Market_Volatility': np.random.uniform(10, 40, n_samples),
    })
    
    # Création d'une variable cible : Tendance du marché (1=Hausse, 0=Baisse)
    df['Market_Trend'] = (df['Market_Index'].pct_change() > 0).astype(int)
    df.loc[0, 'Market_Trend'] = 1  # Première valeur

print()
print(f"📊 Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 3 : EXPLORATION INITIALE (FIRST LOOK)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("3️⃣  EXPLORATION INITIALE : PREMIER REGARD SUR LES DONNÉES")
print("─" * 70)
print()
print("📋 Aperçu des premières lignes :")
print(df.head())
print()
print("🔍 Informations sur les types de données :")
print(df.info())
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 4 : DATA WRANGLING (NETTOYAGE)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("4️⃣  DATA WRANGLING : NETTOYAGE ET PRÉPARATION")
print("─" * 70)
print()

# Simulation de données manquantes (réalisme)
df_dirty = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:5]:  # Corrompre 5 colonnes
    mask = np.random.rand(len(df)) < 0.03  # 3% de valeurs manquantes
    df_dirty.loc[mask, col] = np.nan

print(f"⚠️  Valeurs manquantes introduites (simulation de la réalité) :")
missing = df_dirty.isnull().sum()
print(missing[missing > 0])
print()

# Séparation features/target
# Identifier automatiquement la variable cible
if 'Market_Trend' in df_dirty.columns:
    target_col = 'Market_Trend'
    problem_type = 'classification'
elif 'Market_Index' in df_dirty.columns:
    target_col = 'Market_Index'
    problem_type = 'regression'
else:
    # Prendre la dernière colonne numérique
    numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[-1]
    problem_type = 'regression'

# Exclure les colonnes de date
date_cols = df_dirty.select_dtypes(include=['datetime64', 'object']).columns
X = df_dirty.drop(columns=[target_col] + list(date_cols))
y = df_dirty[target_col]

print(f"🎯 Variable cible identifiée : {target_col}")
print(f"📊 Type de problème : {problem_type.upper()}")
print(f"📐 Features sélectionnées : {X.shape[1]} variables")
print(f"   → {list(X.columns)}")
print()

# Imputation des valeurs manquantes
print("🔧 Stratégie d'imputation : Moyenne (mean)")
print("   ┌─ fit() : Calcul de la moyenne sur les données disponibles")
print("   └─ transform() : Remplissage des trous avec cette moyenne")
print()

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print(f"✅ Nettoyage terminé : 0 valeurs manquantes restantes")
print()

# ⚠️ AVERTISSEMENT DATA LEAKAGE
print("💡 COIN DE L'EXPERT : Data Leakage")
print("   Dans ce script pédagogique, nous avons imputé AVANT de séparer Train/Test.")
print("   En production, c'est une ERREUR subtile :")
print("   → La moyenne calculée inclut des informations du Test Set")
print("   → Risque de sur-optimisme dans les performances")
print()
print("   ✓ Bonne pratique : fit() sur Train uniquement, transform() sur Train ET Test")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 5 : ANALYSE EXPLORATOIRE (EDA)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("5️⃣  ANALYSE EXPLORATOIRE : PROFILAGE DES DONNÉES")
print("─" * 70)
print()

print("📊 Statistiques descriptives (5 premières features) :")
print(X_clean.iloc[:, :5].describe().round(2))
print()

print("🔍 DÉCRYPTAGE DE .describe() :")
print("   • Mean vs 50% (Médiane) : Si Mean >> Médiane → Distribution asymétrique")
print("   • Std (Écart-type) : Mesure de dispersion (std ≈ 0 → variable inutile)")
print("   • Min/Max : Détection des valeurs aberrantes potentielles")
print()

# Matrice de corrélation
print("🌡️  Analyse de la multicollinéarité...")
corr_matrix = X_clean.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.9)
high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                   for x, y in zip(*high_corr) if x != y and x < y]

if high_corr_pairs:
    print("⚠️  Variables fortement corrélées détectées (>0.9) :")
    for var1, var2, corr in high_corr_pairs[:3]:
        print(f"   • {var1} ↔ {var2} : {corr:.3f}")
    print()
    print("   💡 Impact : Redondance d'information (acceptable pour Random Forest)")
else:
    print("✅ Pas de multicollinéarité excessive détectée")
print()

# Visualisation de la distribution de la cible
plt.figure(figsize=(10, 4))
if problem_type == 'classification':
    plt.subplot(1, 2, 1)
    y.value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    plt.title('Distribution de la Variable Cible')
    plt.xlabel(target_col)
    plt.ylabel('Fréquence')
    plt.xticks(rotation=0)
else:
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
    plt.title('Distribution de la Variable Cible')
    plt.xlabel(target_col)
    plt.ylabel('Fréquence')

# Heatmap de corrélation (top 8 features)
plt.subplot(1, 2, 2)
top_features = X_clean.columns[:8]
sns.heatmap(X_clean[top_features].corr(), annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Matrice de Corrélation (8 premières features)')
plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
print("📈 Graphiques sauvegardés : eda_analysis.png")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 6 : PROTOCOLE EXPÉRIMENTAL (SPLIT)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("6️⃣  PROTOCOLE EXPÉRIMENTAL : TRAIN/TEST SPLIT")
print("─" * 70)
print()

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

print("📐 Principe : La Garantie de Généralisation")
print("   Le but du ML n'est pas de MÉMORISER le passé, mais de GÉNÉRALISER au futur.")
print()
print(f"✂️  Séparation effectuée :")
print(f"   • Train Set : {X_train.shape[0]} échantillons (80%) → Apprentissage")
print(f"   • Test Set  : {X_test.shape[0]} échantillons (20%) → Évaluation")
print()
print("🔐 random_state=42 → Reproductibilité scientifique garantie")
print("   (Deux exécutions = résultats identiques)")
print()

# Standardisation (optionnelle mais recommandée)
print("⚖️  Standardisation des features (mean=0, std=1)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Mise à l'échelle terminée")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 7 : MODÉLISATION (RANDOM FOREST)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("7️⃣  INTELLIGENCE ARTIFICIELLE : RANDOM FOREST 🌲")
print("─" * 70)
print()

print("🧠 POURQUOI RANDOM FOREST ?")
print()
print("A. La Faiblesse de l'Individu (Arbre de Décision)")
print("   Un arbre unique → Haute variance → Apprend le bruit")
print()
print("B. La Force du Groupe (Bagging)")
print("   1. Bootstrapping : Chaque arbre voit un échantillon différent")
print("   2. Feature Randomness : À chaque nœud, √n_features aléatoires")
print("   → Diversité maximale des opinions")
print()
print("C. Le Consensus (Vote)")
print("   100 arbres votent → Les erreurs s'annulent → Le signal émerge")
print()

if problem_type == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

print("🏗️  Construction du modèle...")
print(f"   • n_estimators=100 (100 arbres)")
print(f"   • max_depth=10 (profondeur max par arbre)")
print()

print("🚀 Entraînement en cours...")
model.fit(X_train_scaled, y_train)
print("✅ Modèle entraîné avec succès !")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 8 : ÉVALUATION (L'HEURE DE VÉRITÉ)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("8️⃣  AUDIT DE PERFORMANCE : L'HEURE DE VÉRITÉ")
print("─" * 70)
print()

y_pred = model.predict(X_test_scaled)

if problem_type == 'classification':
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 ACCURACY GLOBALE : {accuracy*100:.2f}%")
    print()
    
    print("📊 RAPPORT DÉTAILLÉ (Classification Report) :")
    print(classification_report(y_test, y_pred, digits=3))
    print()
    
    print("🔍 DÉCRYPTAGE DES MÉTRIQUES :")
    print("   • Precision : Qualité de l'alarme (TP / (TP + FP))")
    print("   • Recall : Puissance du filet (TP / (TP + FN))")
    print("   • F1-Score : Moyenne harmonique (2 × Precision × Recall / (P + R))")
    print()
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Nombre'})
    plt.title('Matrice de Confusion : Réalité vs IA', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("📈 Matrice de confusion sauvegardée : confusion_matrix.png")
    print()
    
    print("📋 ANALYSE DE LA MATRICE DE CONFUSION :")
    print(f"   • Vrais Positifs (TP)  : {cm[1, 1]} ✅")
    print(f"   • Vrais Négatifs (TN)  : {cm[0, 0]} ✅")
    print(f"   • Faux Positifs (FP)   : {cm[0, 1]} ⚠️  (Erreur Type I)")
    print(f"   • Faux Négatifs (FN)   : {cm[1, 0]} ⚠️  (Erreur Type II)")
    
else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📏 MÉTRIQUES DE RÉGRESSION :")
    print(f"   • R² Score (Coefficient de détermination) : {r2:.4f}")
    print(f"   • RMSE (Root Mean Squared Error)          : {rmse:.4f}")
    print(f"   • MAE (Mean Absolute Error)               : {mae:.4f}")
    print()
    
    print("🔍 INTERPRÉTATION :")
    print(f"   • R² = {r2:.2%} → Le modèle explique {r2:.1%} de la variance")
    if r2 > 0.7:
        print("     ✅ Excellente performance !")
    elif r2 > 0.5:
        print("     ✓ Performance acceptable")
    else:
        print("     ⚠️  Performance à améliorer")
    print(f"   • RMSE = {rmse:.2f} → Erreur moyenne de prédiction")
    print()
    
    # Graphique Prédictions vs Réalité
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='#4ECDC4', edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Prédiction parfaite')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.title('Prédictions vs Réalité')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    plt.xlabel('Résidus (Erreur)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Erreurs')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_analysis.png', dpi=150, bbox_inches='tight')
    print("📈 Graphiques sauvegardés : regression_analysis.png")

print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 9 : IMPORTANCE DES FEATURES
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("9️⃣  INTERPRÉTABILITÉ : QUELLES VARIABLES COMPTENT ?")
print("─" * 70)
print()

feature_importance = pd.DataFrame({
    'Feature': X_clean.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("🔝 Top 10 des features les plus importantes :")
print(feature_importance.head(10).to_string(index=False))
print()

# Visualisation
plt.figure(figsize=(10, 6))
top_n = min(15, len(feature_importance))
sns.barplot(data=feature_importance.head(top_n), y='Feature', x='Importance', 
            palette='viridis')
plt.title(f'Top {top_n} Features par Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Relative')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("📈 Graphique d'importance sauvegardé : feature_importance.png")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 10 : CONCLUSION ET RECOMMANDATIONS
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("🎓 CONCLUSION : SYNTHÈSE DU PROJET")
print("=" * 70)
print()

print("📝 CE QUE NOUS AVONS APPRIS :")
print()
print("1. CONTEXTE MÉTIER")
print("   → Comprendre le problème avant de coder est crucial")
print("   → Les coûts d'erreur ne sont pas symétriques en finance")
print()
print("2. DATA WRANGLING")
print("   → Les données réelles sont toujours sales (NaN, outliers)")
print("   → Attention au Data Leakage lors de l'imputation")
print()
print("3. EDA (EXPLORATION)")
print("   → .describe() révèle distribution et outliers")
print("   → La corrélation ≠ causalité (mais aide à détecter la redondance)")
print()
print("4. MODÉLISATION")
print("   → Random Forest = robuste, interprétable, peu de tuning")
print("   → Le vote de 100 arbres annule le bruit individuel")
print()
print("5. ÉVALUATION")
if problem_type == 'classification':
    print("   → Accuracy seule est trompeuse (classes déséquilibrées)")
    print("   → Recall est critique pour minimiser les faux négatifs")
else:
    print("   → R² mesure la qualité d'ajustement")
    print("   → RMSE donne l'erreur moyenne en unités réelles")
print()

print("🚀 RECOMMANDATIONS POUR ALLER PLUS LOIN :")
print("   • Tester d'autres algorithmes (XGBoost, LightGBM)")
print("   • Optimiser les hyperparamètres (GridSearchCV)")
print("   • Feature Engineering : créer de nouvelles variables")
print("   • Cross-Validation : valider sur plusieurs folds")
print("   • Déploiement : API Flask/FastAPI pour la production")
print()

print("=" * 70)
print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
print("=" * 70)
print()
print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("👨‍💻 Généré par : Script d'Analyse Automatisée")
print()
print("📂 Fichiers générés :")
print("   • eda_analysis.png")
if problem_type == 'classification':
    print("   • confusion_matrix.png")
else:
    print("   • regression_analysis.png")
print("   • feature_importance.png")
print()
print("🙏 Merci d'avoir utilisé ce guide pédagogique !")
print("="*70)
