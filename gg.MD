import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from datetime import datetime
import warnings, os
warnings.filterwarnings('ignore')

# ========== Génération ou chargement des données ==========
try:
    import kagglehub
    path = kagglehub.dataset_download("kundanbedmutha/market-trend-and-external-factors-dataset")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("Aucun CSV trouvé dans le dataset Kaggle.")
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    source = f"Dataset Kaggle : {csv_files[0]}"
except Exception as e:
    # Données synthétiques de secours
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
    source = f"Données synthétiques (erreur Kaggle : {e})"

# ========== Data wrangling ==========
df_dirty = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:6]:
    mask = np.random.rand(len(df_dirty)) < 0.04
    df_dirty.loc[mask, col] = np.nan

target_col = "Market_Trend" if "Market_Trend" in df_dirty.columns else numeric_cols[-1]
problem_type = "classification" if target_col == "Market_Trend" else "regression"

date_cols = df_dirty.select_dtypes(include=["datetime64", "object"]).columns
exclude_cols = list(date_cols) + [target_col]
if "Price_Change" in df_dirty.columns and target_col != "Price_Change":
    exclude_cols.append("Price_Change")

X = df_dirty.drop(columns=exclude_cols, errors="ignore")
y = df_dirty[target_col]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# ========== Split & scaling ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42,
    stratify=y if problem_type == "classification" else None
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Modélisation (XGBoost ou fallback) ==========
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

model.fit(X_train_scaled, y_train)

# ========== Évaluation ==========
if problem_type == "classification":
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None
    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
else:
    y_pred = model.predict(X_test_scaled)
    # (tu peux ajouter RMSE/R2 si tu passes en régression)

# ========== Graphiques simples ==========
sns.set_theme(style="whitegrid")
# Distribution de la cible
plt.figure()
if problem_type == "classification":
    y.value_counts().plot(kind="bar", color=["#FF6B6B", "#4ECDC4"])
    plt.xlabel(target_col)
    plt.ylabel("Fréquence")
    plt.title("Distribution de la variable cible")
else:
    plt.hist(y, bins=40, color="#4ECDC4", edgecolor="black")
    plt.xlabel(target_col)
    plt.ylabel("Fréquence")
    plt.title("Distribution de la variable cible")
plt.tight_layout()
plt.savefig("target_distribution.png", dpi=150)
plt.close()

# Feature importance (si disponible)
try:
    importances = model.feature_importances_
    fi = pd.DataFrame({"Feature": X_clean.columns, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi.head(10), x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 10 features – {model_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
except Exception:
    fi = None

# Confusion matrix plot
if problem_type == "classification":
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

# ========== Génération du compte rendu Markdown ==========
rapport = ""
rapport += "# Analyse des tendances de marché et facteurs externes\n\n"
rapport += f"_Rapport généré automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"

rapport += "## PHASE 1 – Contexte métier\n\n"
rapport += "- Domaine : finance / marché.\n"
rapport += "- Objectif : prédire la tendance du marché (hausse/baisse) à partir de facteurs externes.\n\n"

rapport += "## PHASE 2 – Données\n\n"
rapport += f"- Source : {source}\n"
rapport += f"- Observations : **{df.shape[0]}**\n"
rapport += f"- Variables : **{df.shape[1]}**\n"
rapport += "- Variable cible : `Market_Trend` (1 = hausse, 0 = baisse).\n\n"

rapport += "### Aperçu des données\n\n```
rapport += df.head().to_string()
rapport += "\n```\n\n"

rapport += "## PHASE 3 – Préparation des données\n\n"
rapport += "- Gestion de valeurs manquantes par imputation (moyenne).\n"
rapport += "- Exclusion des colonnes de date et dérivées (`Date`, `Price_Change`).\n"
rapport += "- Standardisation des variables explicatives.\n\n"

rapport += "## PHASE 4 – Modélisation\n\n"
rapport += f"- Modèle utilisé : **{model_name}**.\n"
rapport += "- Paramètres principaux : nombre d'arbres = 100, profondeur max ≈ 6.\n\n"

rapport += "## PHASE 5 – Résultats\n\n"
if problem_type == "classification":
    rapport += f"- Accuracy sur le jeu de test : **{acc*100:.2f}%**.\n"
    if roc_auc is not None:
        rapport += f"- ROC-AUC : **{roc_auc:.4f}**.\n\n"
    rapport += "### Rapport de classification\n\n```
    rapport += cls_report
    rapport += "\n```\n\n"
    rapport += "### Matrice de confusion\n\n```
    rapport += pd.DataFrame(cm, index=["Réel 0", "Réel 1"], columns=["Prédit 0", "Prédit 1"]).to_string()
    rapport += "\n```\n\n"
else:
    rapport += "- Mode régression non détaillé ici.\n\n"

if fi is not None:
    rapport += "### Principales variables explicatives\n\n```
    rapport += fi.head(10).to_string(index=False)
    rapport += "\n```\n\n"

rapport += "## PHASE 6 – Graphiques générés\n\n"
rapport += "- `target_distribution.png` : distribution de la cible.\n"
if problem_type == "classification":
    rapport += "- `confusion_matrix.png` : matrice de confusion.\n"
if fi is not None:
    rapport += "- `feature_importance.png` : importance des variables.\n\n"

rapport += "## Conclusion\n\n"
rapport += "- Le modèle fournit une première base pour l’aide à la décision sur les tendances de marché.\n"
rapport += "- Des améliorations sont possibles (tuning des hyperparamètres, nouvelles features, etc.).\n"

with open("rapport_marche_xgboost.md", "w", encoding="utf-8") as f:
    f.write(rapport)

print("✅ Rapport généré : rapport_marche_xgboost.md")
