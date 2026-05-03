"""
Train Clinical Model on PPMI/UPDRS-style Dataset
Usage: python models/train_clinical.py
Saves: models/clinical_model.pkl
"""
import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.pipeline import Pipeline
except ImportError:
    print("Install: pip install scikit-learn pandas")
    sys.exit(1)

def train():
    dataset_path = 'datasets/clinical_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Generating synthetic data first...")
        import subprocess
        subprocess.run(['python', 'datasets/generate_synthetic.py'])

    df = pd.read_csv(dataset_path)
    feature_cols = ['age','sex','tremor_0to4','rigidity_0to4',
                    'bradykinesia_0to4','updrs_total','anosmia','rem_disorder']
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("=== Clinical Model ===")
    print(classification_report(y_test, y_pred, target_names=['Healthy','PD']))
    print(f"AUC-ROC: {auc:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    os.makedirs('models', exist_ok=True)
    with open('models/clinical_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    print("Saved: models/clinical_model.pkl")

if __name__ == '__main__':
    train()
