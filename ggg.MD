import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from datetime import datetime
import warnings, os
warnings.filterwarnings('ignore')

print("=" * 60)
print("PROJET : Analyse des tendances de marché et facteurs externes")
print("=" * 60)
print(f"Date d'exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ========= PHASE 1 : DONNÉES =========
print("PHASE 1 - ACQUISITION DES DONNÉES")
print("-" * 60)

try:
    import kagglehub
    print("Téléchargement du dataset depuis Kaggle...")
    path = kagglehub.dataset_download(
        "kundanbedmutha/market-trend-and-external-factors-dataset"
    )
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("Aucun fichier CSV trouvé.")
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    print(f"Source : Dataset Kaggle ({csv_files[0]})")
except Exception as e:
    print(f"Impossible de récupérer Kaggle ({e}), génération de données synthétiques.")
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    market_base = 1000
    market_trend = np.cumsum(np.random.randn(n_samples) * 2 + 0.05)
    df = pd.DataFrame({
        "Date": dates,
        "Market_Index": market_base + market_trend,
        "GDP_Growth": np.random.uniform(1.5, 4.5, n_samples),
        "Inflation_Rate": np.random.uniform(1.0, 5.0, n_samples),
        "Interest_Rate": np.random.uniform(0.5, 3.5, n_samples),
        "Unemployment_Rate": np.random.uniform(3.0, 8.0, n_samples),
        "Consumer_Confidence": np.random.uniform(80, 120, n_samples),
        "Oil_Price": np.random.uniform(40, 100, n_samples),
        "Gold_Price": np.random.uniform(1500, 2000, n_samples),
        "USD_Exchange_Rate": np.random.uniform(0.85, 1.15, n_samples),
        "Market_Volatility": np.random.uniform(10, 40, n_samples),
        "Trade_Volume": np.random.uniform(1e6, 1e7, n_samples),
    })
    df["Price_Change"] = df["Market_Index"].pct_change() * 100
    df["Market_Trend"] = (df["Price_Change"] > 0).astype(int)
    df.loc[0, "Market_Trend"] = 1
    df.loc[0, "Price_Change"] = 0
    print("Source : données synthétiques (simulation).")

print(f"Dimensions du dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print()

# ========= PHASE 2 : EXPLORATION =========
print("PHASE 2 - EXPLORATION RAPIDE")
print("-" * 60)
print("Aperçu des premières lignes :")
print(df.head())
print()
print("Types de variables :")
print(df.dtypes)
print()

# ========= PHASE 3 : PRÉPARATION =========
print("PHASE 3 - NETTOYAGE ET PRÉPARATION")
print("-" * 60)

df_dirty = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols[:5]:
    mask = np.random.rand(len(df_dirty)) < 0.03
    df_dirty.loc[mask, col] = np.nan

print("Valeurs manquantes (après simulation) :")
na_counts = df_dirty.isnull().sum()
print(na_counts[na_counts > 0])
print()

target_col = "Market_Trend" if "Market_Trend" in df_dirty.columns else numeric_cols[-1]
problem_type = "classification" if target_col == "Market_Trend" else "regression"

date_cols = df_dirty.select_dtypes(include=["datetime64", "object"]).columns
exclude_cols = list(date_cols) + [target_col]
if "Price_Change" in df_dirty.columns and target_col != "Price_Change":
    exclude_cols.append("Price_Change")

X = df_dirty.drop(columns=exclude_cols, errors="ignore")
y = df_dirty[target_col]

print(f"Variable cible : {target_col}")
print(f"Type de problème : {problem_type}")
print(f"Nombre de variables explicatives : {X.shape[1]}")
print()

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)
print("Imputation des valeurs manquantes : moyenne par colonne.")
print(f"Valeurs manquantes restantes dans X : {X_clean.isnull().sum().sum()}")
print()

# ========= PHASE 4 : SPLIT & SCALING =========
print("PHASE 4 - TRAIN/TEST SPLIT ET STANDARDISATION")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if problem_type == "classification" else None,
)

print(f"Taille train : {X_train.shape[0]}  |  taille test : {X_test.shape[0]}")
print("random_state=42 pour reproductibilité.")
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Standardisation des features (mean=0, std=1).")
print()

# ========= PHASE 5 : MODÈLE =========
print("PHASE 5 - MODÉLISATION (XGBoost ou équivalent)")
print("-" * 60)

try:
    import xgboost as xgb

    if problem_type == "classification":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    model_name = "XGBoost"
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    if problem_type == "classification":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
    model_name = "GradientBoosting (fallback)"

print(f"Modèle utilisé : {model_name}")
model.fit(X_train_scaled, y_train)
print("Entraînement terminé.")
print()

# ========= PHASE 6 : RÉSULTATS =========
print("PHASE 6 - ÉVALUATION DU MODÈLE")
print("-" * 60)

if problem_type == "classification":
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (test) : {acc * 100:.2f} %")

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC : {roc:.4f}")
    print()
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
else:
    y_pred = model.predict(X_test_scaled)
    print("Problème de régression (métriques non détaillées ici).")

print()
print("PROJET TERMINÉ.")
print("=" * 60)
