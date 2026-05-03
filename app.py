from flask import Flask, render_template, request, jsonify
import os, json, numpy as np
from werkzeug.utils import secure_filename
from utils.speech_processor import analyze_speech
from utils.gait_processor import analyze_gait, analyze_gait_from_dict
from utils.biomarker_processor import analyze_biomarkers
from utils.clinical_processor import analyze_clinical
from utils.mri_processor import analyze_mri
from utils.fusion import fuse_predictions

app = Flask(__name__)

# ── Use os.path.join for Windows compatibility ──────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
SPEECH_DIR  = os.path.join(UPLOAD_DIR, 'speech')
GAIT_DIR    = os.path.join(UPLOAD_DIR, 'gait')
MRI_DIR     = os.path.join(UPLOAD_DIR, 'mri')

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

# Create upload dirs on startup
for d in [SPEECH_DIR, GAIT_DIR, MRI_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Index ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── Speech ──────────────────────────────────────────────────────
@app.route('/predict/speech', methods=['POST'])
def predict_speech():
    try:
        if 'file' in request.files:
            f = request.files['file']
            if f and f.filename:
                fname = secure_filename(f.filename)
                path  = os.path.join(SPEECH_DIR, fname)
                f.save(path)
                result = analyze_speech(path)
            else:
                return jsonify({'error': 'No valid file'}), 400
        elif request.json and 'audio_data' in request.json:
            import base64, tempfile
            audio_bytes = base64.b64decode(request.json['audio_data'])
            tmp_path = os.path.join(SPEECH_DIR, 'recorded.wav')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(audio_bytes)
            result = analyze_speech(tmp_path)
        else:
            return jsonify({'error': 'No audio provided'}), 400
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(),
                        'probability': 0.5, 'features': {}, 'key_findings': [str(e)],
                        'risk_level': 'Unknown', 'modality': 'speech'}), 200

# ── Gait ────────────────────────────────────────────────────────
@app.route('/predict/gait', methods=['POST'])
def predict_gait():
    try:
        if 'file' in request.files:
            f = request.files['file']
            if f and f.filename:
                fname = secure_filename(f.filename)
                path  = os.path.join(GAIT_DIR, fname)
                f.save(path)
                result = analyze_gait(path)
            else:
                return jsonify({'error': 'No valid file'}), 400
        elif request.json and 'video_data' in request.json:
            import base64
            video_bytes = base64.b64decode(request.json['video_data'])
            tmp_path = os.path.join(GAIT_DIR, 'recorded.webm')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(video_bytes)
            result = analyze_gait(tmp_path)
        else:
            return jsonify({'error': 'No video/file provided'}), 400
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(),
                        'probability': 0.5, 'features': {}, 'key_findings': [str(e)],
                        'risk_level': 'Unknown', 'modality': 'gait'}), 200

@app.route('/predict/gait_form', methods=['POST'])
def predict_gait_form():
    try:
        data   = request.get_json()
        result = analyze_gait_from_dict(data)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(),
                        'probability': 0.5, 'features': {}, 'key_findings': [str(e)],
                        'risk_level': 'Unknown', 'modality': 'gait'}), 200

# ── Biomarkers ──────────────────────────────────────────────────
@app.route('/predict/biomarkers', methods=['POST'])
def predict_biomarkers():
    try:
        data   = request.get_json()
        result = analyze_biomarkers(data)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(),
                        'probability': 0.5, 'features': {}, 'key_findings': [str(e)],
                        'risk_level': 'Unknown', 'modality': 'biomarkers'}), 200

# ── Clinical ────────────────────────────────────────────────────
@app.route('/predict/clinical', methods=['POST'])
def predict_clinical():
    try:
        data   = request.get_json()
        result = analyze_clinical(data)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(),
                        'probability': 0.5, 'features': {}, 'key_findings': [str(e)],
                        'risk_level': 'Unknown', 'modality': 'clinical'}), 200

# ── MRI ─────────────────────────────────────────────────────────
@app.route('/predict/mri', methods=['POST'])
def predict_mri():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No MRI file uploaded'}), 400
        f = request.files['file']
        if not f or not f.filename:
            return jsonify({'error': 'Empty file'}), 400
        fname  = secure_filename(f.filename)
        path   = os.path.join(MRI_DIR, fname)
        f.save(path)
        result = analyze_mri(path)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(),
                        'probability': 0.5, 'features': {}, 'key_findings': [str(e)],
                        'risk_level': 'Unknown', 'modality': 'mri'}), 200

# ── Fusion ───────────────────────────────────────────────────────
@app.route('/predict/fused', methods=['POST'])
def predict_fused():
    try:
        data   = request.get_json()
        result = fuse_predictions(data)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# ── Debug route — shows errors clearly ──────────────────────────
@app.route('/health')
def health():
    checks = {}
    model_dir = os.path.join(BASE_DIR, 'models')
    for m in ['speech_model','gait_model','mri_model','biomarker_model','clinical_model']:
        path = os.path.join(model_dir, f'{m}.pkl')
        checks[m] = 'OK' if os.path.exists(path) else 'MISSING'
    checks['upload_dirs'] = {
        'speech': os.path.exists(SPEECH_DIR),
        'gait':   os.path.exists(GAIT_DIR),
        'mri':    os.path.exists(MRI_DIR),
    }
    return jsonify(checks)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
