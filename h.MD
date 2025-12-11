import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from datetime import datetime
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

rapport = ""
rapport += "# Analyse des tendances de marché et facteurs externes\n\n"
rapport += f"_Rapport généré automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"

# PHASE 1 : CONTEXTE
rapport += "## 1️⃣ Contexte métier et objectif\n\n"
rapport += (
    "Dans le monde de la finance et du trading, les décisions d'investissement "
    "reposent sur la compréhension des tendances de marché et des facteurs externes "
    "(économiques, politiques, sociaux).\n\n"
)
rapport += "- **Objectif** : construire un modèle prédictif pour anticiper les mouvements de marché.\n"
rapport += "- **Enjeux** : limiter les pertes (faux positifs) et le manque à gagner (faux négatifs).\n\n"

# PHASE 2 : ACQUISITION
rapport += "## 2️⃣ Acquisition et chargement des données\n\n"
rapport += "Les données proviennent du dataset Kaggle `market-trend-and-external-factors-dataset`.\n\n"
rapport += "```
rapport += "import kagglehub, os, pandas as pd\n"
rapport += "path = kagglehub.dataset_download('kundanbedmutha/market-trend-and-external-factors-dataset')\n"
rapport += "csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]\n"
rapport += "df = pd.read_csv(os.path.join(path, csv_files))\n"
rapport += "```\n\n"

try:
    import kagglehub
    path = kagglehub.dataset_download("kundanbedmutha/market-trend-and-external-factors-dataset")
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        rapport += f"- Fichier chargé : **{csv_files[0]}**\n\n"
    else:
        raise FileNotFoundError("Aucun fichier CSV trouvé")
except Exception as e:
    rapport += f"> ⚠️ Erreur lors du téléchargement réel : `{e}`\n\n"
    rapport += "> Des données synthétiques ont été générées pour la démonstration.\n\n"
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
    df['Market_Trend'] = (df['Market_Index'].pct_change() > 0).astype(int)
    df.loc[0, 'Market_Trend'] = 1

rapport += f"- Nombre de lignes : **{df.shape[0]}**\n"
rapport += f"- Nombre de colonnes : **{df.shape[1]}**\n\n"

rapport += "### Aperçu des premières lignes\n\n```
rapport += df.head().to_string()
rapport += "\n```\n\n"

rapport += "### Informations sur les types de données\n\n```
from io import StringIO
buf = StringIO()
df.info(buf=buf)
rapport += buf.getvalue()
rapport += "```\n\n"

# PHASE 4 : DATA WRANGLING
rapport += "## 4️⃣ Data wrangling : nettoyage et préparation\n\n"

df_dirty = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:5]:
    mask = np.random.rand(len(df)) < 0.03
    df_dirty.loc[mask, col] = np.nan

missing = df_dirty.isnull().sum()
missing_nonzero = missing[missing > 0]

rapport += "### Valeurs manquantes introduites (simulation)\n\n```
rapport += missing_nonzero.to_string()
rapport += "\n```\n\n"

if 'Market_Trend' in df_dirty.columns:
    target_col = 'Market_Trend'
    problem_type = 'classification'
elif 'Market_Index' in df_dirty.columns:
    target_col = 'Market_Index'
    problem_type = 'regression'
else:
    numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[-1]
    problem_type = 'regression'

date_cols = df_dirty.select_dtypes(include=['datetime64', 'object']).columns
X = df_dirty.drop(columns=[target_col] + list(date_cols))
y = df_dirty[target_col]

rapport += f"- Variable cible : **{target_col}**\n"
rapport += f"- Type de problème : **{problem_type.upper()}**\n"
rapport += f"- Nombre de variables explicatives : **{X.shape[1]}**\n\n"

rapport += "### Imputation des valeurs manquantes\n\n"
rapport += "- Stratégie : **moyenne (mean)** sur les variables numériques.\n\n"

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

rapport += "Après imputation, il ne reste plus de valeurs manquantes dans les features.\n\n"

rapport += "> ⚠️ Remarque importante : dans ce script pédagogique, l'imputation est faite avant le split Train/Test.\n"
rapport += "> En production, il faut **fit** l'imputeur sur le train uniquement pour éviter le data leakage.\n\n"

# PHASE 5 : EDA
rapport += "## 5️⃣ Analyse exploratoire (EDA)\n\n"
rapport += "### Statistiques descriptives (5 premières variables)\n\n```
rapport += X_clean.iloc[:, :5].describe().round(2).to_string()
rapport += "\n```\n\n"

rapport += "Interprétation rapide de `.describe()` :\n"
rapport += "- **mean vs 50% (médiane)** : forte différence → distribution asymétrique.\n"
rapport += "- **std** : mesure de dispersion (proche de 0 → variable peu informative).\n"
rapport += "- **min/max** : utiles pour repérer des valeurs aberrantes.\n\n"

corr_matrix = X_clean.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.9)
high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                   for x, y in zip(*high_corr) if x != y and x < y]

if high_corr_pairs:
    rapport += "### Multicollinéarité (corrélations > 0.9)\n\n```
    for var1, var2, corr in high_corr_pairs[:5]:
        rapport += f"{var1} ↔ {var2} : {corr:.3f}\n"
    rapport += "```\n\n"
else:
    rapport += "Aucune corrélation excessive (>0.9) détectée entre les variables numériques.\n\n"

plt.figure(figsize=(10, 4))
if problem_type == 'classification':
    y.value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    plt.title('Distribution de la variable cible')
    plt.xlabel(target_col)
    plt.ylabel('Fréquence')
else:
    plt.hist(y, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
    plt.title('Distribution de la variable cible')
    plt.xlabel(target_col)
    plt.ylabel('Fréquence')
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

rapport += "Un graphique de distribution de la cible a été sauvegardé sous `target_distribution.png`.\n\n"

# PHASE 6 : SPLIT
rapport += "## 6️⃣ Protocole expérimental : Train/Test split\n\n"

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

rapport += f"- Taille du Train set : **{X_train.shape[0]}** observations (80%)\n"
rapport += f"- Taille du Test set : **{X_test.shape[0]}** observations (20%)\n"
rapport += "- `random_state=42` utilisé pour la reproductibilité.\n\n"

rapport += "### Standardisation des variables\n\n"
rapport += "Les features sont mises à l'échelle (moyenne 0, écart-type 1) via `StandardScaler`.\n\n"

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PHASE 7 : MODELE
rapport += "## 7️⃣ Modélisation : Random Forest\n\n"
rapport += "Le modèle utilisé est une **Random Forest**, robuste et capable de capturer des relations non linéaires.\n\n"
rapport += "- Avantages : robustesse, peu de tuning, gestion naturelle des interactions et non-linéarités.\n"
rapport += "- Hyperparamètres principaux : `n_estimators=100`, `max_depth=10`.\n\n"

if problem_type == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

model.fit(X_train_scaled, y_train)

# PHASE 8 : ÉVALUATION
rapport += "## 8️⃣ Évaluation du modèle\n\n"

y_pred = model.predict(X_test_scaled)

if problem_type == 'classification':
    accuracy = accuracy_score(y_test, y_pred)
    rapport += f"- **Accuracy globale** : **{accuracy*100:.2f}%**\n\n"
    rapport += "### Rapport de classification\n\n```
    rapport += classification_report(y_test, y_pred, digits=3)
    rapport += "\n```\n\n"

    cm = confusion_matrix(y_test, y_pred)
    rapport += "### Matrice de confusion\n\n```
    rapport += pd.DataFrame(cm,
                            index=["Vrai 0", "Vrai 1"],
                            columns=["Prédit 0", "Prédit 1"]).to_string()
    rapport += "\n```\n\n"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Nombre'})
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    rapport += "Une matrice de confusion a été sauvegardée sous `confusion_matrix.png`.\n\n"
    rapport += "Interprétation :\n"
    rapport += "- **Faux positifs (FP)** : prédictions de hausse qui n'ont pas lieu.\n"
    rapport += "- **Faux négatifs (FN)** : opportunités manquées (hausse non prédite).\n\n"

else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rapport += "### Métriques de régression\n\n"
    rapport += f"- **R²** : `{r2:.4f}`\n"
    rapport += f"- **RMSE** : `{rmse:.4f}`\n"
    rapport += f"- **MAE** : `{mae:.4f}`\n\n"

    rapport += "Interprétation :\n"
    rapport += "- R² mesure la part de variance expliquée par le modèle.\n"
    rapport += "- RMSE et MAE quantifient l'erreur moyenne en unités de la cible.\n\n"

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='#4ECDC4', edgecolors='black')
    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Prédiction parfaite')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Prédictions vs Réalité')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    plt.xlabel('Résidus (erreur)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des erreurs')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    rapport += "Des graphiques `regression_analysis.png` ont été générés (prédictions vs réalité, distribution des résidus).\n\n"

# PHASE 9 : IMPORTANCE DES FEATURES
rapport += "## 9️⃣ Interprétabilité : importance des variables\n\n"

feature_importance = pd.DataFrame({
    'Feature': X_clean.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

rapport += "### Top 10 des variables les plus importantes\n\n```
rapport += feature_importance.head(10).to_string(index=False)
rapport += "\n```\n\n"

plt.figure(figsize=(10, 6))
top_n = min(15, len(feature_importance))
sns.barplot(data=feature_importance.head(top_n), y='Feature', x='Importance', palette='viridis')
plt.title(f'Top {top_n} features par importance')
plt.xlabel('Importance relative')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

rapport += "Un graphique d'importance des variables a été sauvegardé sous `feature_importance.png`.\n\n"

# PHASE 10 : CONCLUSION
rapport += "## 🔟 Conclusion et recommandations\n\n"
rapport += "### Synthèse du projet\n\n"
rapport += "1. **Contexte métier** :\n"
rapport += "   - Compréhension des enjeux de prédiction des tendances de marché.\n"
rapport += "2. **Données** :\n"
rapport += "   - Jeu de données multi-facteurs (macroéconomie, marchés, devises...).\n"
rapport += "3. **Préparation** :\n"
rapport += "   - Nettoyage, gestion des valeurs manquantes, standardisation des variables.\n"
rapport += "4. **Modélisation** :\n"
rapport += "   - Utilisation d'une Random Forest pour capturer des relations complexes.\n"
rapport += "5. **Évaluation** :\n"
rapport += "   - Analyse des métriques principales et des erreurs.\n\n"

if problem_type == 'classification':
    rapport += "- Pour la **classification**, il est crucial d'équilibrer précision et rappel,\n"
    rapport += "  surtout lorsque les coûts des faux positifs et faux négatifs sont asymétriques.\n\n"
else:
    rapport += "- Pour la **régression**, l'amélioration du R² et la réduction du RMSE passent\n"
    rapport += "  par un meilleur feature engineering et éventuellement des modèles plus puissants.\n\n"

rapport += "### Pistes d'amélioration\n\n"
rapport += "- Tester d'autres algorithmes (XGBoost, LightGBM, réseaux neuronaux).\n"
rapport += "- Optimiser les hyperparamètres (GridSearchCV ou RandomizedSearchCV).\n"
rapport += "- Ajouter ou transformer des variables (feature engineering avancé).\n"
rapport += "- Mettre en place une validation croisée robuste.\n"
rapport += "- Envisager un déploiement (API Flask/FastAPI, dashboard Streamlit).\n\n"

rapport += "---\n\n"
rapport += "Rapport généré automatiquement par un script Python d'analyse de données.\n"

with open("rapport_marche.md", "w", encoding="utf-8") as f:
    f.write(rapport)

print("✅ Rapport Markdown généré : rapport_marche.md")
print("   À déposer tel quel sur GitHub avec les images PNG générées.")
