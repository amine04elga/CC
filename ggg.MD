print("PHASE 5 - MODÉLISATION (XGBoost ou équivalent)")
print("-"*60)

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
