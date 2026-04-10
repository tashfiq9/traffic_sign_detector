"""
resave_models.py  —  run ONCE from F:\\project\\
    python resave_models.py

Rebuilds EfficientNetB0 and MobileNetV2 using a Rescaling layer instead of
a Lambda layer (Lambda causes shape-inference errors in newer Keras).
Loads weights from the old files and re-saves clean new ones.
"""

import os
import tensorflow as tf
from keras.applications import EfficientNetB0, MobileNetV2
from keras.applications.efficientnet import preprocess_input as eff_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from keras.layers import (
    GlobalAveragePooling2D, BatchNormalization,
    Dropout, Dense, Rescaling
)

IMG_SIZE    = 48
NUM_CLASSES = 43
TRAIN_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')


# ─────────────────────────────────────────────────────────────────────────────
# Custom preprocessing layers (replace Lambda — no shape inference issues)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetPreprocess(tf.keras.layers.Layer):
    """Scales [0,1] → [0,255] then applies EfficientNet preprocessing."""
    def call(self, x):
        return eff_preprocess(x * 255.0)
    def get_config(self):
        return super().get_config()


class MobileNetPreprocess(tf.keras.layers.Layer):
    """Scales [0,1] → [0,255] then applies MobileNetV2 preprocessing."""
    def call(self, x):
        return mob_preprocess(x * 255.0)
    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture builders
# ─────────────────────────────────────────────────────────────────────────────

def build_efficientnet():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_layer')

    x = EfficientNetPreprocess(name='efficientnet_preprocess')(inputs)

    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None
    )
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    x = base(x, training=False)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization(name='head_bn1')(x)
    x = Dropout(0.4, name='head_drop1')(x)
    x = Dense(256, activation='relu', name='head_dense1')(x)
    x = BatchNormalization(name='head_bn2')(x)
    x = Dropout(0.3, name='head_drop2')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    return tf.keras.Model(inputs, outputs, name='EfficientNetB0_TSR')


def build_mobilenet():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_layer')

    x = MobileNetPreprocess(name='mobilenet_preprocess')(inputs)

    base = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=1.0
    )
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    x = base(x, training=False)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization(name='head_bn1')(x)
    x = Dropout(0.4, name='head_drop1')(x)
    x = Dense(256, activation='relu', name='head_dense1')(x)
    x = BatchNormalization(name='head_bn2')(x)
    x = Dropout(0.3, name='head_drop2')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    return tf.keras.Model(inputs, outputs, name='MobileNetV2_TSR')


# ─────────────────────────────────────────────────────────────────────────────
# Re-save EfficientNet
# ─────────────────────────────────────────────────────────────────────────────
old_eff = os.path.join(TRAIN_DIR, 'EfficientNetB0_phase2_best.keras')
new_eff = os.path.join(TRAIN_DIR, 'EfficientNetB0_fixed.keras')

print("=" * 60)
print("Rebuilding EfficientNetB0 ...")
eff_model = build_efficientnet()
print(f"Loading weights from: {old_eff}")
eff_model.load_weights(old_eff)
eff_model.save(new_eff)
print(f"✓ Saved to: {new_eff}")

# ─────────────────────────────────────────────────────────────────────────────
# Re-save MobileNetV2
# ─────────────────────────────────────────────────────────────────────────────
old_mob = os.path.join(TRAIN_DIR, 'MobileNetV2_phase2_best.keras')
new_mob = os.path.join(TRAIN_DIR, 'MobileNetV2_fixed.keras')

print("=" * 60)
print("Rebuilding MobileNetV2 ...")
mob_model = build_mobilenet()
print(f"Loading weights from: {old_mob}")
mob_model.load_weights(old_mob)
mob_model.save(new_mob)
print(f"✓ Saved to: {new_mob}")

print("=" * 60)
print("All done! Now run Traffic_app.py — all 3 models should load.")