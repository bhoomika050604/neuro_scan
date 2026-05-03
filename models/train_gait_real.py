"""
Train Gait Model on uploaded gait.csv
Dataset structure (tab-separated):
  col1 = subject_id
  col2 = GROUP: 'control'=healthy, 'park'=PD, 'hunt'=Huntington, 'subjects'=ALS
  col3 = AGE(YRS)
  col4 = HEIGHT(meters)
  col5 = Weight(kg)
  col6 = gender (m/f)
  col7 = GaitSpeed(m/sec)
  col8 = Duration/Severity

Strategy: binary classify park vs control only (ignore hunt/ALS for PD prediction)
Run: python models/train_gait_real.py
Saves: models/gait_model.pkl
"""
import numpy as np
import pandas as pd
import pickle, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer

FEATURE_NAMES = ['age', 'height_m', 'weight_kg', 'gender_encoded', 'gait_speed_ms', 'severity']

def load_gait():
    df = pd.read_csv('datasets/gait.csv', sep='\t', header=0)
    df.columns = ['subject_id','group','age','height_m','weight_kg','gender','gait_speed_ms','severity']

    # Binary: park=1 (PD), control=0 (healthy)  — drop hunt/ALS/unknown rows
    df = df[df['group'].isin(['park', 'control'])].copy()
    df['label'] = (df['group'] == 'park').astype(int)

    # Encode gender
    df['gender_encoded'] = (df['gender'].str.strip().str.lower() == 'm').astype(int)

    # Replace MISSING with NaN
    for col in ['gait_speed_ms', 'weight_kg', 'height_m', 'age', 'severity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def train():
    df = load_gait()
    print(f"Loaded: {len(df)} samples  |  PD (park): {df['label'].sum()}  |  Control: {(df['label']==0).sum()}")
    print(df[FEATURE_NAMES + ['label']].describe())

    X_raw = df[FEATURE_NAMES].values
    y = df['label'].values

    # ── Pipeline with imputer (handles MISSING values) ────────────────────────
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ])
    gb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=42
        ))
    ])
    svm_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(C=5, kernel='rbf', probability=True, class_weight='balanced'))
    ])

    ensemble = VotingClassifier(
        estimators=[('rf', rf_pipe), ('gb', gb_pipe), ('svm', svm_pipe)],
        voting='soft', weights=[2, 2, 1]
    )

    # ── Small dataset → use LOO-CV + stratified k-fold ────────────────────────
    print("\nRunning Leave-One-Out CV (small dataset)...")
    loo = LeaveOneOut()
    loo_preds, loo_probs, loo_true = [], [], []
    for train_idx, test_idx in loo.split(X_raw, y):
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        ensemble.fit(X_tr, y_tr)
        loo_preds.append(ensemble.predict(X_te)[0])
        loo_probs.append(ensemble.predict_proba(X_te)[0][1])
        loo_true.append(y_te[0])

    loo_acc = accuracy_score(loo_true, loo_preds)
    loo_auc = roc_auc_score(loo_true, loo_probs)
    print(f"LOO Accuracy : {loo_acc:.4f}")
    print(f"LOO AUC-ROC  : {loo_auc:.4f}")
    print("\nLOO Classification Report:")
    print(classification_report(loo_true, loo_preds, target_names=['Control','PD']))

    # ── Train final model on all data ─────────────────────────────────────────
    print("Training final model on all data...")
    ensemble.fit(X_raw, y)

    # Compute feature importances from RF sub-pipeline
    rf_clf = ensemble.estimators_[0].named_steps['clf']
    importances = rf_clf.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.4f}")

    # Median values for imputation at inference time
    medians = {}
    for i, col in enumerate(FEATURE_NAMES):
        col_vals = X_raw[:, i]
        valid = col_vals[~np.isnan(col_vals)]
        medians[col] = float(np.median(valid)) if len(valid) > 0 else 0.0

    os.makedirs('models', exist_ok=True)
    save_obj = {
        'model': ensemble,
        'features': FEATURE_NAMES,
        'loo_accuracy': float(loo_acc),
        'loo_auc': float(loo_auc),
        'col_medians': medians
    }
    with open('models/gait_model.pkl', 'wb') as f:
        pickle.dump(save_obj, f)

    print(f"\nSaved → models/gait_model.pkl")
    print(f"LOO Accuracy: {loo_acc*100:.1f}%  |  AUC: {loo_auc:.3f}")
    return save_obj

if __name__ == '__main__':
    train()
