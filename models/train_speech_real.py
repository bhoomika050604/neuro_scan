"""
Train Speech Model on actual mPower/MDVP dataset
Files: datasets/speech_train.txt, datasets/speech_test.txt
Structure:
  col 1        = subject_id  (skip)
  cols 2-27    = 26 speech features
  col 28       = patient_id  (skip)
  col 29       = label: 0=healthy, 1=PD  (train only; test has no label)

Run: python models/train_speech_real.py
Saves: models/speech_model.pkl
"""
import numpy as np
import pandas as pd
import pickle, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# ── Feature names for 26 speech columns (cols 2–27) ───────────────────────────
FEATURE_NAMES = [
    'Jitter_local', 'Jitter_local_abs', 'Jitter_rap', 'Jitter_ppq5', 'Jitter_ddp',
    'Shimmer_local', 'Shimmer_local_db', 'Shimmer_apq3', 'Shimmer_apq5',
    'Shimmer_apq11', 'Shimmer_dda',
    'AC', 'NTH', 'HTN',
    'Median_pitch', 'Mean_pitch', 'SD_pitch', 'Min_pitch', 'Max_pitch',
    'Num_pulses', 'Num_periods',
    'Mean_period', 'SD_period',
    'Fraction_unvoiced', 'Num_breaks', 'Degree_breaks'
]

def load_data(path, has_label=True):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if has_label and len(parts) >= 29:
                features = [float(x) if x not in ('', 'MISSING') else np.nan
                            for x in parts[1:27]]
                label = int(float(parts[28].strip()))
                rows.append(features + [label])
            elif not has_label and len(parts) >= 27:
                features = [float(x) if x not in ('', 'MISSING') else np.nan
                            for x in parts[1:27]]
                rows.append(features)
    return rows

def train():
    print("Loading data...")
    train_rows = load_data('datasets/speech_train.txt', has_label=True)
    df = pd.DataFrame(train_rows, columns=FEATURE_NAMES + ['label'])

    print(f"Train samples: {len(df)}  |  PD: {df['label'].sum()}  |  Healthy: {(df['label']==0).sum()}")

    # Impute any NaN with column median
    for col in FEATURE_NAMES:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    X = df[FEATURE_NAMES].values
    y = df['label'].values

    # ── Models ────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=10, kernel='rbf', gamma='scale', probability=True, class_weight='balanced'))
    ])

    # Ensemble with soft voting
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft',
        weights=[2, 2, 1]
    )

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\nRunning 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc  = cross_val_score(ensemble, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    cv_auc  = cross_val_score(ensemble, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_f1   = cross_val_score(ensemble, X, y, cv=skf, scoring='f1', n_jobs=-1)

    print(f"CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"CV AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"CV F1       : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── Train on full training set ────────────────────────────────────────────
    print("\nTraining final model on full training set...")
    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    y_prob = ensemble.predict_proba(X)[:, 1]
    print("\nTraining set performance:")
    print(classification_report(y, y_pred, target_names=['Healthy', 'PD']))
    print(f"Train AUC-ROC: {roc_auc_score(y, y_prob):.4f}")

    # ── Feature importance from RF ─────────────────────────────────────────
    rf_fitted = ensemble.estimators_[0]
    importances = rf_fitted.feature_importances_
    top = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:10]
    print("\nTop 10 features (RF):")
    for name, imp in top:
        print(f"  {name:<30} {imp:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    save_obj = {
        'model': ensemble,
        'features': FEATURE_NAMES,
        'cv_accuracy': float(cv_acc.mean()),
        'cv_auc': float(cv_auc.mean()),
        'cv_f1': float(cv_f1.mean()),
        'col_medians': {col: float(df[col].median()) for col in FEATURE_NAMES}
    }
    with open('models/speech_model.pkl', 'wb') as f:
        pickle.dump(save_obj, f)

    print(f"\nSaved → models/speech_model.pkl")
    print(f"Reported accuracy: {cv_acc.mean()*100:.1f}%")
    return save_obj

if __name__ == '__main__':
    train()
