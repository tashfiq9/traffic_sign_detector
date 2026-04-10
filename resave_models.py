"""
resave_models.py  —  run ONCE from F:\\project3\\
    python resave_models.py
"""
import os, json, zipfile, shutil, tempfile
import tensorflow as tf

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')

def strip_renorm_from_keras_file(src_path, dst_path):
    """
    A .keras file is a zip. This opens it, patches the config.json
    to remove renorm/renorm_clipping/renorm_momentum from every
    BatchNormalization layer, then writes a new .keras file.
    """
    RENORM_KEYS = {'renorm', 'renorm_clipping', 'renorm_momentum'}

    def strip_renorm(obj):
        """Recursively remove renorm keys from any dict."""
        if isinstance(obj, dict):
            # If this looks like a BatchNormalization config, strip the keys
            if obj.get('class_name') == 'BatchNormalization' and 'config' in obj:
                for k in RENORM_KEYS:
                    obj['config'].pop(k, None)
            # Also strip directly from any dict that has these keys
            for k in RENORM_KEYS:
                obj.pop(k, None)
            for v in obj.values():
                strip_renorm(v)
        elif isinstance(obj, list):
            for item in obj:
                strip_renorm(item)
        return obj

    tmp_dir = tempfile.mkdtemp()
    try:
        # Extract the zip
        with zipfile.ZipFile(src_path, 'r') as zin:
            zin.extractall(tmp_dir)

        # Patch config.json
        config_path = os.path.join(tmp_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config = strip_renorm(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        # Repack into new .keras zip
        with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname  = os.path.relpath(full_path, tmp_dir)
                    zout.write(full_path, arcname)

        print(f"  Patched config written to: {dst_path}")
    finally:
        shutil.rmtree(tmp_dir)


# ── EfficientNetB0 ────────────────────────────────────────────────────────────
old_eff     = os.path.join(TRAIN_DIR, 'EfficientNetB0_phase2_best.keras')
patched_eff = os.path.join(TRAIN_DIR, 'EfficientNetB0_patched.keras')
new_eff     = os.path.join(TRAIN_DIR, 'EfficientNetB0_fixed.keras')

print("=" * 60)
print("Patching EfficientNetB0 config ...")
strip_renorm_from_keras_file(old_eff, patched_eff)
print("Loading patched EfficientNetB0 ...")
eff_model = tf.keras.models.load_model(patched_eff, safe_mode=False)
eff_model.save(new_eff)
os.remove(patched_eff)
print(f"✓ Saved to: {new_eff}")

# ── MobileNetV2 ───────────────────────────────────────────────────────────────
old_mob     = os.path.join(TRAIN_DIR, 'MobileNetV2_phase2_best.keras')
patched_mob = os.path.join(TRAIN_DIR, 'MobileNetV2_patched.keras')
new_mob     = os.path.join(TRAIN_DIR, 'MobileNetV2_fixed.keras')

print("=" * 60)
print("Patching MobileNetV2 config ...")
strip_renorm_from_keras_file(old_mob, patched_mob)
print("Loading patched MobileNetV2 ...")
mob_model = tf.keras.models.load_model(patched_mob, safe_mode=False)
mob_model.save(new_mob)
os.remove(patched_mob)
print(f"✓ Saved to: {new_mob}")

print("=" * 60)
print("All done!")