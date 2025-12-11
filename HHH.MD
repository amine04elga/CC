"""
═══════════════════════════════════════════════════════════════════
📊 GRAND GUIDE : ANALYSE DES TENDANCES DE MARCHÉ & FACTEURS EXTERNES
═══════════════════════════════════════════════════════════════════

Ce script génère un rapport complet d'analyse de données financières
dans le style pédagogique d'un Data Scientist expert.

Auteur : Analyse automatisée
Objectif : Prédire les tendances de marché avec XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
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
print("   🎯 Objectif : Créer un Assistant IA pour anticiper les mouvements de marché")
print("      en analysant des indicateurs externes (PIB, inflation, taux d'intérêt).")
print()
print("   ⚠️  L'Enjeu Critique : La matrice des coûts d'erreur")
print("      • Faux Positif (prédire hausse → investir → baisse réelle)")
print("        → Perte financière directe + coûts de transaction")
print("      • Faux Négatif (prédire baisse → ne pas investir → hausse réelle)")
print("        → Manque à gagner (opportunité ratée)")
print()
print("   💡 Stratégie : Optimiser le ratio risque/rendement avec une IA robuste")
print("      qui minimise les fausses alertes tout en capturant les vraies opportunités.")
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
    
    # Simulation de facteurs économiques
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    market_base = 1000
    market_trend = np.cumsum(np.random.randn(n_samples) * 2 + 0.05)
    
    df = pd.DataFrame({
        'Date': dates,
        'Market_Index': market_base + market_trend,
        'GDP_Growth': np.random.uniform(1.5, 4.5, n_samples) + np.sin(np.arange(n_samples)/50) * 0.5,
        'Inflation_Rate': np.random.uniform(1.0, 5.0, n_samples) + np.cos(np.arange(n_samples)/60) * 0.3,
        'Interest_Rate': np.random.uniform(0.5, 3.5, n_samples),
        'Unemployment_Rate': np.random.uniform(3.0, 8.0, n_samples),
        'Consumer_Confidence': np.random.uniform(80, 120, n_samples),
        'Oil_Price': np.random.uniform(40, 100, n_samples) + np.sin(np.arange(n_samples)/30) * 10,
        'Gold_Price': np.random.uniform(1500, 2000, n_samples),
        'USD_Exchange_Rate': np.random.uniform(0.85, 1.15, n_samples),
        'Market_Volatility': np.random.uniform(10, 40, n_samples),
        'Trade_Volume': np.random.uniform(1e6, 1e7, n_samples),
    })
    
    # Création d'une variable cible : Tendance du marché (1=Hausse, 0=Baisse)
    df['Price_Change'] = df['Market_Index'].pct_change() * 100
    df['Market_Trend'] = (df['Price_Change'] > 0).astype(int)
    df.loc[0, 'Market_Trend'] = 1  # Première valeur
    df.loc[0, 'Price_Change'] = 0

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
exclude_target = [col for col in numeric_cols if 'trend' not in col.lower() and 'target' not in col.lower()]

for col in exclude_target[:6]:  # Corrompre 6 colonnes
    mask = np.random.rand(len(df)) < 0.04  # 4% de valeurs manquantes
    df_dirty.loc[mask, col] = np.nan

print(f"⚠️  Valeurs manquantes introduites (simulation de la réalité) :")
missing = df_dirty.isnull().sum()
missing_display = missing[missing > 0]
if len(missing_display) > 0:
    print(missing_display)
else:
    print("   Aucune valeur manquante dans ce dataset")
print()

# Séparation features/target
# Identifier automatiquement la variable cible
if 'Market_Trend' in df_dirty.columns:
    target_col = 'Market_Trend'
    problem_type = 'classification'
elif 'Trend' in df_dirty.columns:
    target_col = 'Trend'
    problem_type = 'classification'
elif 'Market_Index' in df_dirty.columns:
    target_col = 'Market_Index'
    problem_type = 'regression'
else:
    # Prendre la dernière colonne numérique
    numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[-1]
    problem_type = 'regression'

# Exclure les colonnes de date et variables dérivées
date_cols = df_dirty.select_dtypes(include=['datetime64', 'object']).columns
exclude_cols = list(date_cols) + [target_col]
if 'Price_Change' in df_dirty.columns and target_col != 'Price_Change':
    exclude_cols.append('Price_Change')

X = df_dirty.drop(columns=exclude_cols, errors='ignore')
y = df_dirty[target_col]

print(f"🎯 Variable cible identifiée : {target_col}")
print(f"📊 Type de problème : {problem_type.upper()}")
print(f"📐 Features sélectionnées : {X.shape[1]} variables")
print(f"   → {list(X.columns)[:10]}")  # Afficher les 10 premières
if len(X.columns) > 10:
    print(f"   ... et {len(X.columns) - 10} autres")
print()

# Imputation des valeurs manquantes
print("🔧 Stratégie d'imputation : Moyenne (mean)")
print("   Mécanisme de SimpleImputer :")
print("   ┌─ fit() : Calcul μ (moyenne) sur chaque colonne des données disponibles")
print("   │         Ex: Colonne 'GDP_Growth' → μ = 2.85%")
print("   └─ transform() : Pour chaque NaN détecté → injection de μ")
print()

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print(f"✅ Nettoyage terminé : {X_clean.isnull().sum().sum()} valeurs manquantes restantes")
print()

# ⚠️ AVERTISSEMENT DATA LEAKAGE
print("💡 COIN DE L'EXPERT : Data Leakage")
print("   ⚠️  Dans ce script pédagogique, nous avons imputé AVANT de séparer Train/Test.")
print("   En production rigoureuse, c'est une ERREUR subtile :")
print()
print("   ❌ Mauvaise pratique (ici) :")
print("      1. fit(toutes_les_données)  ← La moyenne inclut le Test Set")
print("      2. transform(toutes_les_données)")
print("      3. train_test_split()")
print()
print("   ✅ Bonne pratique industrielle :")
print("      1. train_test_split()  ← Séparer d'abord")
print("      2. fit(X_train)  ← Calculer μ uniquement sur Train")
print("      3. transform(X_train) et transform(X_test)  ← Appliquer le même μ")
print()
print("   💰 Impact financier : Le Data Leakage peut entraîner un sur-optimisme")
print("      de +2 à +5% en accuracy, ce qui fausse les décisions d'investissement.")
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
print("   • count : Nombre d'observations (devrait être constant après imputation)")
print("   • mean vs 50% (médiane) :")
print("     → Si mean >> médiane : Distribution asymétrique (skewed right)")
print("     → Exemple : Si mean=3.2 et 50%=2.8 pour GDP → Pics exceptionnels tirés vers le haut")
print("   • std (Écart-type) : Mesure de volatilité")
print("     → std élevé = variable instable (ex: Oil_Price)")
print("     → std ≈ 0 = variable quasi-constante (inutile pour le ML)")
print("   • min/max : Détection d'outliers potentiels")
print("     → Exemple : Interest_Rate à 15% serait suspect")
print()

# Matrice de corrélation
print("🌡️  Analyse de la multicollinéarité...")
corr_matrix = X_clean.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.9)
high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                   for x, y in zip(*high_corr) if x != y and x < y]

if high_corr_pairs:
    print("⚠️  Variables fortement corrélées détectées (|r| > 0.9) :")
    for var1, var2, corr in high_corr_pairs[:3]:
        print(f"   • {var1} ↔ {var2} : {corr:.3f}")
    print()
    print("   💡 Le problème de la redondance :")
    print("      Géométriquement : Deux variables corrélées portent la même information")
    print("      Impact ML :")
    print("      → XGBoost : TOLÉRANT (robuste à la multicollinéarité)")
    print("      → Régression Linéaire : PROBLÉMATIQUE (coefficients instables)")
else:
    print("✅ Pas de multicollinéarité excessive détectée")
print()

# Analyse de la distribution de la cible
print("🎯 Analyse de la variable cible :")
if problem_type == 'classification':
    print(y.value_counts())
    balance_ratio = y.value_counts().min() / y.value_counts().max()
    print(f"   Balance ratio : {balance_ratio:.2f}")
    if balance_ratio < 0.5:
        print("   ⚠️  Classes déséquilibrées détectées ! Considérer SMOTE ou class_weight")
    else:
        print("   ✅ Classes relativement équilibrées")
else:
    print(f"   Moyenne : {y.mean():.2f}")
    print(f"   Std : {y.std():.2f}")
    print(f"   Min : {y.min():.2f} | Max : {y.max():.2f}")
print()

# Visualisation
fig = plt.figure(figsize=(14, 5))

# Distribution de la cible
ax1 = plt.subplot(1, 3, 1)
if problem_type == 'classification':
    counts = y.value_counts()
    ax1.bar(counts.index, counts.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax1.set_title('Distribution de la Variable Cible', fontsize=12, fontweight='bold')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Fréquence')
    ax1.set_xticks(counts.index)
else:
    ax1.hist(y, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution de la Variable Cible', fontsize=12, fontweight='bold')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Fréquence')

# Heatmap de corrélation (top 8 features)
ax2 = plt.subplot(1, 3, 2)
top_features = X_clean.columns[:min(8, len(X_clean.columns))]
sns.heatmap(X_clean[top_features].corr(), annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1, ax=ax2)
ax2.set_title('Matrice de Corrélation (Top Features)', fontsize=12, fontweight='bold')

# Boxplot pour détecter les outliers
ax3 = plt.subplot(1, 3, 3)
sample_features = X_clean.iloc[:, :4]
sample_features_scaled = (sample_features - sample_features.mean()) / sample_features.std()
ax3.boxplot([sample_features_scaled.iloc[:, i] for i in range(sample_features_scaled.shape[1])],
            labels=sample_features_scaled.columns)
ax3.set_title('Détection d\'Outliers (Standardisé)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Valeur (Z-score)')
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')

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
    X_clean, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None
)

print("📐 Le Concept : La Garantie de Généralisation")
print("   Le but du ML n'est PAS de MÉMORISER le passé (overfitting),")
print("   mais de GÉNÉRALISER vers le futur (prédictions robustes).")
print()
print("   🧪 Analogie : Un étudiant qui mémorise les réponses d'un examen")
print("      vs un étudiant qui comprend les concepts.")
print()
print(f"✂️  Séparation effectuée :")
print(f"   • Train Set : {X_train.shape[0]} échantillons (80%) → Apprentissage")
print(f"   • Test Set  : {X_test.shape[0]} échantillons (20%) → Évaluation")
if problem_type == 'classification':
    print(f"   • Stratification activée : maintient les proportions de classes")
print()
print("🔐 Les Paramètres sous le Capot :")
print("   • test_size=0.2 : Le ratio 80/20 (Principe de Pareto)")
print("     → 80% pour capturer la complexité des patterns")
print("     → 20% pour une évaluation statistiquement significative")
print("   • random_state=42 : Reproductibilité scientifique")
print("     → Deux exécutions = résultats identiques")
print("     → Partage avec un collègue au Japon = mêmes patients dans Test Set")
print()

# Standardisation (critique pour certains algos, optionnelle pour XGBoost)
print("⚖️  Standardisation des features...")
print("   Note : XGBoost est invariant à l'échelle (tree-based),")
print("   mais on standardise pour cohérence avec d'autres algos potentiels.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Transformation : mean=0, std=1 pour chaque feature")
print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 7 : MODÉLISATION (XGBOOST)
# ═══════════════════════════════════════════════════════════════════

print("─" * 70)
print("7️⃣  INTELLIGENCE ARTIFICIELLE : XGBOOST 🚀")
print("─" * 70)
print()

print("🧠 POURQUOI XGBOOST POUR LES MARCHÉS FINANCIERS ?")
print()
print("A. Le Champion des Compétitions Kaggle")
print("   XGBoost (eXtreme Gradient Boosting) domine les compétitions de Data Science")
print("   depuis 2014. Il a révolutionné le ML grâce à :")
print("   • Vitesse d'exécution : 10x plus rapide que les autres Gradient Boosting")
print("   • Gestion native des valeurs manquantes")
print("   • Régularisation intégrée (L1/L2) → Réduit l'overfitting")
print()
print("B. La Mécanique : Boosting Séquentiel")
print("   Contrairement au Random Forest (Bagging = arbres en parallèle),")
print("   XGBoost utilise le Boosting = arbres en SÉQUENCE.")
print()
print("   Étape 1 : L'Arbre #1 fait des prédictions")
print("            → Il se trompe sur certains points")
print()
print("   Étape 2 : L'Arbre #2 se concentre sur ces erreurs")
print("            → Il apprend à corriger les faiblesses de l'Arbre #1")
print()
print("   Étape 3 : L'Arbre #3 corrige les erreurs restantes...")
print()
print("   Étape 100 : 100 arbres collaborent de manière séquentielle")
print()
print("   💡 Formule mathématique :")
print("      ŷ_final = ŷ_arbre1 + η·ŷ_arbre2 + η·ŷ_arbre3 + ... + η·ŷ_arbre100")
print("      où η (learning_rate) contrôle la contribution de chaque arbre")
print()
print("C. Les Hyperparamètres Critiques")
print("   • n_estimators : Nombre d'arbres (↑ précision, ↓ risque overfitting)")
print("   • max_depth : Profondeur max (↑ complexité, ↑ risque overfitting)")
print("   • learning_rate : Pas d'apprentissage (petit = robuste mais lent)")
print("   • subsample : % de données par arbre (ajoute randomness)")
print()

try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("⚠️  XGBoost n'est pas installé. Installation : pip install xgboost")
    print("📦 Utilisation de GradientBoosting (sklearn) comme alternative...")
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

if xgboost_available:
    if problem_type == 'classification':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    model_name = "XGBoost"
else:
    if problem_type == 'classification':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    model_name = "GradientBoosting"

print(f"🏗️  Construction du modèle {model_name}...")
print(f"   • n_estimators=100 (100 arbres séquentiels)")
print(f"   • max_depth=6 (profondeur limitée pour éviter l'overfitting)")
print(f"   • learning_rate=0.1 (contribution modérée de chaque arbre)")
print(f"   • subsample=0.8 (80% des données par arbre → diversité)")
print()

print("🚀 Entraînement en cours...")
print("   Phase 1 : Construction de l'arbre #1...")
model.fit(X_train_scaled, y_train)
print(f"   Phase 100 : Agrégation des 100 arbres → Prédiction finale")
print(f"✅ Modèle {model_name} entraîné avec succès !")
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
    
    # Prédictions probabilistes pour ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"📊 ROC-AUC SCORE : {roc_auc:.4f}")
        print("   → 0.5 = hasard | 1.0 = parfait | >0.75 = bon modèle")
        print()
    
    print("📋 RAPPORT DÉTAILLÉ (Classification Report) :")
    print(classification_report(y_test, y_pred, digits=3))
    print()
    
    print("🔍 DÉCRYPTAGE DES MÉTRIQUES (Application Finance) :")
    print()
    print("   1. PRECISION (Qualité de l'alarme)")
    print("      Formule : TP / (TP + FP)")
    print("      Question : Quand l'IA prédit 'Hausse', est-ce fiable ?")
    print("      → Precision basse = Beaucoup de fausses alertes")
    print("      💰 Impact : Coûts de transaction inutiles")
    print()
    print("   2. RECALL / SENSIBILITÉ (Puissance du filet)")
    print("      Formule : TP / (TP + FN)")
    print("      Question : Sur toutes les vraies hausses, combien sont détectées ?")
    print("      → Recall bas = Opportunités ratées")
    print("      💰 Impact : Manque à gagner")
    print()
    print("   3. F1-SCORE (Équilibre)")
    print("      Formule : 2 × (Precision × Recall) / (Precision + Recall)")
    print("      → Moyenne harmonique (pénalise les déséquilibres)")
    print("      → F1 élevé = Bon compromis entre qualité et couverture")
    print()
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    fig = plt.figure(figsize=(14, 5))
    
    # Matrice de confusion
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Nombre'}, ax=ax1)
    ax1.set_title('Matrice de Confusion : Réalité vs IA', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Vraie Classe', fontsize=11)
    ax1.set_xlabel('Classe Prédite', font
