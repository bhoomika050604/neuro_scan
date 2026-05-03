"""
Train MRI Model (handcrafted features → sklearn, or CNN for NIfTI)
Usage: python models/train_mri.py
Saves: models/mri_model.pkl

For full 3D CNN training, install:
  pip install torch torchvision nibabel nilearn
"""
import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.pipeline import Pipeline
except ImportError:
    print("Install: pip install scikit-learn pandas")
    sys.exit(1)

def generate_mri_features(n=500):
    """Simulate volumetric MRI feature extraction (replace with nibabel pipeline)."""
    np.random.seed(42)
    rows, labels = [], []
    for label in [0, 1]:
        for _ in range(n):
            if label == 0:  # healthy
                sn = np.random.uniform(480, 620)
                fa = np.random.uniform(0.38, 0.55)
                fw = np.random.uniform(0.05, 0.12)
                put = np.random.uniform(4200, 5200)
                asym = np.random.uniform(0.0, 0.06)
            else:  # PD
                sn = np.random.uniform(380, 480)
                fa = np.random.uniform(0.25, 0.38)
                fw = np.random.uniform(0.14, 0.25)
                put = np.random.uniform(3500, 4200)
                asym = np.random.uniform(0.08, 0.20)
            rows.append([sn, fa, fw, put, asym,
                         np.random.uniform(1.8, 2.8) if label else np.random.uniform(2.4, 3.0)])
            labels.append(label)
    return np.array(rows), np.array(labels)

def train():
    feature_cols = ['sn_volume_mm3','fa_sn','free_water','putamen_volume','signal_asymmetry','cortical_thickness']
    X, y = generate_mri_features()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, C=5, kernel='rbf'))])
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    model = VotingClassifier([('svm', svm), ('rf', rf)], voting='soft')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("=== MRI Model ===")
    print(classification_report(y_test, y_pred, target_names=['Healthy','PD']))
    print(f"AUC-ROC: {auc:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    os.makedirs('models', exist_ok=True)
    with open('models/mri_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    print("Saved: models/mri_model.pkl")

if __name__ == '__main__':
    train()

# ── REAL 3D CNN (PyTorch) ─────────────────────────────────────────────────────
# import torch, torch.nn as nn, nibabel as nib
# class MRI3DCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
#             nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
#             nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(4),
#         )
#         self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*64, 256), nn.ReLU(), nn.Linear(256, 2))
#     def forward(self, x): return self.fc(self.conv(x))
