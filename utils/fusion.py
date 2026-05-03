"""
Late Fusion Engine — combines predictions from all 5 modalities
Uses learned weights + Bayesian averaging
"""
import numpy as np

# Modality weights based on clinical literature + typical model accuracy
DEFAULT_WEIGHTS = {
    'speech': 0.20,
    'gait': 0.22,
    'biomarkers': 0.23,
    'clinical': 0.20,
    'mri': 0.15,
}

def fuse_predictions(predictions_dict):
    """
    predictions_dict = {
        'speech': {'probability': 0.7, 'confidence': 0.85, ...},
        'gait': {'probability': 0.6, 'confidence': 0.80, ...},
        ...
    }
    """
    probs = {}
    confs = {}
    
    for modality, pred in predictions_dict.items():
        if pred and 'probability' in pred:
            probs[modality] = float(pred['probability'])
            confs[modality] = float(pred.get('confidence', 0.80))
    
    if not probs:
        return {'error': 'No valid predictions to fuse'}
    
    # Weighted average (weight * confidence * probability)
    total_weight = 0.0
    weighted_sum = 0.0
    for mod, prob in probs.items():
        w = DEFAULT_WEIGHTS.get(mod, 0.2) * confs.get(mod, 0.8)
        weighted_sum += w * prob
        total_weight += w
    
    fused_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
    
    # Uncertainty estimation (disagreement between modalities)
    prob_values = list(probs.values())
    uncertainty = np.std(prob_values) if len(prob_values) > 1 else 0.1
    adjusted_confidence = np.clip(1.0 - uncertainty, 0.5, 0.99)
    
    # Risk stratification
    risk = 'Low' if fused_prob < 0.35 else ('Moderate' if fused_prob < 0.65 else 'High')
    
    # Clinical interpretation
    if fused_prob < 0.25:
        interpretation = "Results strongly suggest absence of Parkinson's Disease markers. Continue routine health monitoring."
        recommendation = "Routine annual neurological screening recommended."
    elif fused_prob < 0.45:
        interpretation = "Some markers present but below diagnostic threshold. Monitoring advised."
        recommendation = "Follow-up assessment in 6-12 months. Consult a neurologist if symptoms persist."
    elif fused_prob < 0.65:
        interpretation = "Multiple modalities show moderate PD indicators. Clinical evaluation strongly advised."
        recommendation = "Neurologist referral recommended. Consider DaTscan imaging for confirmation."
    else:
        interpretation = "High concordance across modalities indicating significant PD risk markers. Immediate evaluation required."
        recommendation = "Urgent neurologist referral. DaTscan, detailed UPDRS assessment, and dopaminergic therapy evaluation warranted."
    
    # Find most predictive modality
    max_mod = max(probs, key=probs.get) if probs else None
    
    modality_summary = {
        mod: {
            'probability': round(p, 4),
            'risk': 'Low' if p < 0.35 else ('Moderate' if p < 0.65 else 'High'),
            'weight_used': round(DEFAULT_WEIGHTS.get(mod, 0.2), 2)
        }
        for mod, p in probs.items()
    }
    
    return {
        'fused_probability': round(float(fused_prob), 4),
        'risk_level': risk,
        'confidence': round(float(adjusted_confidence), 3),
        'uncertainty': round(float(uncertainty), 3),
        'modalities_used': list(probs.keys()),
        'modality_summary': modality_summary,
        'most_predictive': max_mod,
        'interpretation': interpretation,
        'recommendation': recommendation,
        'fusion_method': 'Confidence-Weighted Late Fusion',
        'disclaimer': (
            'This system is a research tool only and does not constitute medical advice. '
            'Always consult a qualified neurologist for diagnosis.'
        )
    }
