from flask import Flask, render_template, request, jsonify
import os
import sys
import time
import threading
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import requests as _requests

import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Dense, Flatten,
    Dropout, BatchNormalization, Input
)
from keras.applications.efficientnet import preprocess_input as eff_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

NUM_CLASSES = 43
MODEL_IMG_SIZES = {'cnn': 48, 'eff': 96, 'mob': 96}

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DRIVE_IDS = {
    'cnn': '1aL5YZVqdY7yOBhTCHBWccyV1zBBzzlZh',
    'eff': '1EmJIAoRSsyeM5btNgEviDv5TapNehJJm',
    'mob': '1SFwKU9pXo6rLtRb4UXWDolOph8AVg61f',
}
MODEL_FILENAMES = {
    'cnn': 'TSR_best.keras',
    'eff': 'EfficientNetB0_fixed.keras',
    'mob': 'MobileNetV2_fixed.keras',
}

AVAILABLE_MODELS    = ['cnn', 'eff', 'mob']
_model_cache        = {}
_load_locks         = {key: threading.Lock() for key in AVAILABLE_MODELS}
_predict_locks      = {key: threading.Lock() for key in AVAILABLE_MODELS}
_warmup_status      = {key: 'pending' for key in AVAILABLE_MODELS}
_download_file_lock = threading.Lock()

# Max seconds allowed for a single model download (rate-limited trickle = fail fast)
MAX_DOWNLOAD_SECONDS = 180


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


class EfficientNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return eff_preprocess(x * 255.0)
    def get_config(self):
        return super().get_config()

class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return mob_preprocess(x * 255.0)
    def get_config(self):
        return super().get_config()

CUSTOM_OBJECTS = {
    'EfficientNetPreprocess': EfficientNetPreprocess,
    'MobileNetPreprocess':    MobileNetPreprocess,
}


def build_cnn():
    model = Sequential([
        Input(shape=(48, 48, 3)),
        Conv2D(32, (3,3), padding='same', activation='relu'), BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'), BatchNormalization(),
        MaxPool2D((2,2)), Dropout(0.2),
        Conv2D(64, (3,3), padding='same', activation='relu'), BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'), BatchNormalization(),
        MaxPool2D((2,2)), Dropout(0.2),
        Conv2D(128, (3,3), padding='same', activation='relu'), BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'), BatchNormalization(),
        MaxPool2D((2,2)), Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'), BatchNormalization(), Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def _download_model(key, dest):
    """
    Download from Google Drive using usercontent URL.
    Enforces a hard wall-clock timeout so rate-limited/slow downloads
    fail fast and retry, instead of trickling forever.
    """
    file_id = DRIVE_IDS[key]
    MAX_RETRIES = 3
    DOWNLOAD_URL = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&authuser=0&confirm=t"
    )

    with _download_file_lock:
        if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 100:
            log(f"  '{key}' already on disk, skipping download.")
            return

        for attempt in range(1, MAX_RETRIES + 1):
            tmp = dest + '.tmp'
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)

                log(f"Downloading '{key}' (attempt {attempt}/{MAX_RETRIES}) ...")
                t_start = time.time()

                resp = _requests.get(
                    DOWNLOAD_URL,
                    stream=True,
                    timeout=(30, 60),        # 30s connect, 60s per-chunk read
                    headers={'User-Agent': 'Mozilla/5.0'},
                )
                resp.raise_for_status()

                total = 0
                with open(tmp, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)

                        # Hard wall-clock limit — kills rate-limited trickle downloads
                        elapsed = time.time() - t_start
                        if elapsed > MAX_DOWNLOAD_SECONDS:
                            raise RuntimeError(
                                f"Download exceeded {MAX_DOWNLOAD_SECONDS}s "
                                f"({total // 1024} KB downloaded so far) — "
                                "Google Drive may be rate-limiting. Retrying."
                            )

                if total < 1024 * 100:
                    raise RuntimeError(
                        f"File too small ({total} bytes) — "
                        "probably received an HTML error page instead of the model."
                    )

                os.replace(tmp, dest)
                elapsed = time.time() - t_start
                log(f"  ✓ '{key}' downloaded ({total // 1024} KB in {elapsed:.1f}s)")
                return

            except Exception as e:
                log(f"  ✗ Attempt {attempt} failed: {e}")
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                if attempt < MAX_RETRIES:
                    log(f"  Waiting 10s before retry ...")
                    time.sleep(10)   # brief pause between retries avoids instant re-rate-limit
                else:
                    raise RuntimeError(
                        f"Failed to download '{key}' after {MAX_RETRIES} attempts: {e}"
                    )


def get_model(key):
    if key in _model_cache:
        return _model_cache[key]
    with _load_locks[key]:
        if key in _model_cache:
            return _model_cache[key]
        _warmup_status[key] = 'loading'
        dest = os.path.join(MODEL_DIR, MODEL_FILENAMES[key])
        if not os.path.exists(dest) or os.path.getsize(dest) < 1024 * 100:
            _download_model(key, dest)
        log(f"Loading '{key}' ...")
        try:
            if key == 'cnn':
                model = build_cnn()
                model.load_weights(dest)
            else:
                model = load_model(dest, compile=False, custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            _warmup_status[key] = 'failed'
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to load '{key}': {e}")
        _model_cache[key] = model
        _warmup_status[key] = 'ready'
        log(f"  ✓ '{key}' ready")
        return model


def _warmup_all_models():
    log("[Warmup] Starting ...")
    for key in AVAILABLE_MODELS:
        try:
            get_model(key)
            log(f"[Warmup] '{key}' ✓")
        except Exception as e:
            _warmup_status[key] = 'failed'
            log(f"[Warmup] '{key}' FAILED: {e}")
    log("[Warmup] Done.")


threading.Thread(target=_warmup_all_models, daemon=True).start()


CLASSES = {
    0:'Speed limit (20km/h)',        1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',        3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',        5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',       9:'No passing',
    10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road',              13:'Yield',
    14:'Stop',                       15:'No vehicles',
    16:'Vehicle > 3.5 tons prohibited', 17:'No entry',
    18:'General caution',            19:'Dangerous curve left',
    20:'Dangerous curve right',      21:'Double curve',
    22:'Bumpy road',                 23:'Slippery road',
    24:'Road narrows on the right',  25:'Road work',
    26:'Traffic signals',            27:'Pedestrians',
    28:'Children crossing',          29:'Bicycles crossing',
    30:'Beware of ice/snow',         31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead',
    34:'Turn left ahead',            35:'Ahead only',
    36:'Go straight or right',       37:'Go straight or left',
    38:'Keep right',                 39:'Keep left',
    40:'Roundabout mandatory',       41:'End of no passing',
    42:'End no passing vehicle > 3.5 tons'
}


def preprocess_image(img_path, model_key):
    img_size = MODEL_IMG_SIZES[model_key]
    image = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route('/')
def index():
    return render_template('index.html', available_models=AVAILABLE_MODELS)


@app.route('/ready')
def ready():
    statuses   = dict(_warmup_status)
    all_ready  = all(s == 'ready'  for s in statuses.values())
    any_failed = any(s == 'failed' for s in statuses.values())

    if all_ready:
        code = 200
    elif any_failed:
        code = 503
    else:
        code = 202

    return jsonify({
        'all_ready':  all_ready,
        'any_failed': any_failed,
        'models':     statuses,
    }), code


@app.route('/predict', methods=['POST'])
def predict():
    file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        model_key = request.form.get('model', 'cnn')
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': 'Invalid model.'}), 400

        status = _warmup_status.get(model_key)
        if status == 'failed':
            return jsonify({'error': f"Model '{model_key.upper()}' failed to load. Try reloading the page."}), 503
        if status != 'ready':
            return jsonify({'error': f"Model '{model_key.upper()}' is not ready yet. Please wait."}), 503

        filename    = secure_filename(f.filename)
        unique_name = f"{threading.get_ident()}_{filename}"
        file_path   = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        f.save(file_path)

        model = get_model(model_key)
        X     = preprocess_image(file_path, model_key)

        with _predict_locks[model_key]:
            pred = model.predict(X, verbose=0)

        pred_class = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred) * 100)
        label      = CLASSES.get(pred_class, 'Unknown')

        log(f"[{model_key.upper()}] class={pred_class} | conf={confidence:.1f}% | label={label}")
        return jsonify({
            'result':     label,
            'confidence': f"{confidence:.1f}",
            'model':      model_key.upper(),
            'class_id':   pred_class,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)