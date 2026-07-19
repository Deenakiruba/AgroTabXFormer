"""
Model architectures for harvest-day prediction.

AgroTabXFormer and the baseline Transformer are identical except for the
pooling layer: attention pooling versus mean pooling. This isolates the
contribution of attention-based pooling over the growing-season sequence.

Written for Keras 3 (`keras.ops`). For Keras 2, replace ops.softmax /
ops.matmul / ops.sum with tf.nn.softmax / tf.matmul / tf.reduce_sum.
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, ops


class PositionalEncoding(layers.Layer):
    """Learned positional embedding added to a (batch, T, d) sequence."""

    def __init__(self, T, d, **kwargs):
        super().__init__(**kwargs)
        self.T, self.d = T, d

    def build(self, input_shape):
        self.pos = self.add_weight(
            shape=(1, self.T, self.d), initializer="glorot_uniform",
            trainable=True, name="pos_embedding")

    def call(self, x):
        return x + self.pos


class AttentionPooling(layers.Layer):
    """Learned attention-weighted pooling over the time axis."""

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1), initializer="glorot_uniform",
            trainable=True, name="attention_weights")

    def call(self, x):
        a = ops.softmax(ops.matmul(x, self.w), axis=1)   # (batch, T, 1)
        return ops.sum(x * a, axis=1)                     # (batch, d)


def transformer_block(x, d, heads, ff_dim, dropout):
    attn = layers.MultiHeadAttention(heads, d // heads, dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dense(d)(ff)
    return layers.LayerNormalization()(x + ff)


def _head(pooled, tab_in, dropout):
    t = layers.Dense(64, activation="gelu")(tab_in)
    x = layers.Concatenate()([pooled, t])
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(1)(x)


def agrotabxformer(n_features, T=170, C=4, d=96, heads=8, blocks=2, drop=0.1):
    """Proposed model: transformer encoder with attention pooling."""
    seq_in = keras.Input(shape=(T, C))
    tab_in = keras.Input(shape=(n_features,))
    z = layers.Dense(d)(seq_in)
    z = PositionalEncoding(T, d)(z)
    for _ in range(blocks):
        z = transformer_block(z, d, heads, d * 2, drop)
    z = AttentionPooling()(z)
    return keras.Model([seq_in, tab_in], _head(z, tab_in, drop))


def transformer(n_features, T=170, C=4, d=96, heads=8, blocks=2, drop=0.1):
    """Ablation: identical to agrotabxformer but with mean pooling."""
    seq_in = keras.Input(shape=(T, C))
    tab_in = keras.Input(shape=(n_features,))
    z = layers.Dense(d)(seq_in)
    z = PositionalEncoding(T, d)(z)
    for _ in range(blocks):
        z = transformer_block(z, d, heads, d * 2, drop)
    z = layers.GlobalAveragePooling1D()(z)
    return keras.Model([seq_in, tab_in], _head(z, tab_in, drop))


def lstm(n_features, T=170, C=4, units=64, drop=0.2):
    seq_in = keras.Input(shape=(T, C))
    tab_in = keras.Input(shape=(n_features,))
    z = layers.LSTM(units, return_sequences=True, dropout=drop)(seq_in)
    z = layers.LSTM(units, dropout=drop)(z)
    return keras.Model([seq_in, tab_in], _head(z, tab_in, drop))


def cnn(n_features, T=170, C=4, filters=64, drop=0.2):
    seq_in = keras.Input(shape=(T, C))
    tab_in = keras.Input(shape=(n_features,))
    z = layers.Conv1D(filters, 5, padding="causal", activation="gelu")(seq_in)
    z = layers.Conv1D(filters, 5, padding="causal", dilation_rate=2, activation="gelu")(z)
    z = layers.Conv1D(filters, 5, padding="causal", dilation_rate=4, activation="gelu")(z)
    z = layers.GlobalAveragePooling1D()(z)
    return keras.Model([seq_in, tab_in], _head(z, tab_in, drop))


def tabtransformer(n_features, d=96, n_tokens=8, heads=8, blocks=2, drop=0.1):
    """Tabular-only baseline: attention over projected feature tokens."""
    inp = keras.Input(shape=(n_features,))
    z = layers.Dense(d * n_tokens)(inp)
    z = layers.Reshape((n_tokens, d))(z)
    for _ in range(blocks):
        z = transformer_block(z, d, heads, d * 2, drop)
    z = AttentionPooling()(z)
    z = layers.Dense(64, activation="gelu")(z)
    z = layers.Dropout(drop)(z)
    return keras.Model(inp, layers.Dense(1)(z))


def callbacks():
    return [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True,
                                      monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-5),
    ]
