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

sns.set_theme(style="whitegrid")

--- PHASE 1 : ACQUISITION DES DONNÉES ---
try:
import kagglehub
path = kagglehub.dataset_download(
"kundanbedmutha/market-trend-and-external-factors-dataset"
)
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
raise FileNotFoundError("Aucun fichier CSV trouvé.")
df = pd.read_csv(os.path.join(path, csv_files))
print(f"Source : Dataset Kaggle ({csv_files})")
except Exception as e:
print(f"Erreur Kaggle ({e}), génération de données synthétiques.")
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

print("Dimensions :", df.shape)
print(df.head())

--- PHASE 2 : DATA WRANGLING (NETTOYAGE) ---
df_dirty = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns

simulation de quelques NaN
for col in numeric_cols[:5]:
mask = np.random.rand(len(df_dirty)) < 0.03
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

--- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("\n--- Statistiques descriptives (5 premières features) ---")
print(X_clean.iloc[:, :5].describe())

plt.figure()
if problem_type == "classification":
y.value_counts().plot(kind="bar")
plt.xlabel(target_col)
plt.ylabel("Fréquence")
plt.title("Distribution de la cible")
else:
plt.hist(y, bins=40)
plt.xlabel(target_col)
plt.ylabel("Fréquence")
plt.title("Distribution de la cible")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()

--- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
X_clean,
y,
test_size=0.2,
random_state=42,
stratify=y if problem_type == "classification" else None
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

--- PHASE 5 : INTELLIGENCE ARTIFICIELLE (XGBoost / Gradient Boosting) ---
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
