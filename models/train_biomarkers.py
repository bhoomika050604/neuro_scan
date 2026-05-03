"""
Train Biomarker Model on PPMI-style Dataset
Usage: python models/train_biomarkers.py
Saves: models/biomarker_model.pkl
"""
import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
except ImportError:
    print("Install: pip install scikit-learn pandas")
    sys.exit(1)

def train():
    dataset_path = 'datasets/biomarker_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Generating synthetic data first...")
        import subprocess
        subprocess.run(['python', 'datasets/generate_synthetic.py'])

    df = pd.read_csv(dataset_path)
    feature_cols = ['alpha_synuclein_ngmL','dj1_ngmL','nfl_pgmL',
                    'gfap_pgmL','uric_acid_mgdL','tnf_alpha_pgmL']
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline with imputer (handles missing biomarkers in real data)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced'))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("=== Biomarker Model ===")
    print(classification_report(y_test, y_pred, target_names=['Healthy','PD']))
    print(f"AUC-ROC: {auc:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    rf = model.named_steps['clf']
    importances = rf.feature_importances_
    print("\nFeature importances:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {col}: {imp:.3f}")

    os.makedirs('models', exist_ok=True)
    with open('models/biomarker_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    print("Saved: models/biomarker_model.pkl")

if __name__ == '__main__':
    train()
