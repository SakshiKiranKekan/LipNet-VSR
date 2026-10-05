"""
LipNet Deep Learning Architecture in TensorFlow / Keras (FR-9, FR-10).
Reference: Assael et al., "LipNet: End-to-End Sentence-level Lipreading"
Architecture: 3D-CNN + SpatialDropout + MaxPool3D + BiGRU + CTC Loss
"""

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from typing import Optional, Tuple


def ctc_loss_func(y_true, y_pred, input_length, label_length):
    """
    TensorFlow CTC Loss calculation for variable length transcripts.
    """
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


class CTCLossLayer(layers.Layer):
    """Custom Keras layer wrapping CTC loss for end-to-end training."""
    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        y_true, y_pred, input_length, label_length = inputs
        loss = ctc_loss_func(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def build_lipnet_model(
    input_shape: Tuple[int, int, int, int] = (75, 46, 96, 1),
    vocab_size: int = 39,
    training_mode: bool = False
) -> tf.keras.Model:
    """
    Constructs the standard LipNet 3D-CNN + BiGRU architecture.
    
    Args:
        input_shape: (Frames, Height, Width, Channels) = (75, 46, 96, 1)
        vocab_size: Size of vocabulary including CTC blank token.
        training_mode: If True, adds CTC loss layer inputs for training.
        
    Returns:
        tf.keras.Model
    """
    # Video input: (Batch, 75, 46, 96, 1)
    video_input = layers.Input(shape=input_shape, name="video_input", dtype="float32")

    # Spatiotemporal Conv Block 1
    x = layers.ZeroPadding3D(padding=(1, 2, 2))(video_input)
    x = layers.Conv3D(32, kernel_size=(3, 5, 5), strides=(1, 1, 1), padding="valid", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout3D(0.5)(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="max1")(x)

    # Spatiotemporal Conv Block 2
    x = layers.ZeroPadding3D(padding=(1, 2, 2))(x)
    x = layers.Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), padding="valid", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout3D(0.5)(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="max2")(x)

    # Spatiotemporal Conv Block 3
    x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)
    x = layers.Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout3D(0.5)(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="max3")(x)

    # Flatten spatial features into time-distributed vector: (Batch, 75, H*W*C)
    # Output shape after 3 maxpools is (Batch, 75, 5, 12, 96) -> (Batch, 75, 5760)
    x = layers.TimeDistributed(layers.Flatten(), name="flatten_spatial")(x)

    # Sequence Processing with 2-Layer Bidirectional GRU
    x = layers.Bidirectional(
        layers.GRU(256, return_sequences=True, kernel_initializer="Orthogonal", name="gru1"),
        merge_mode="concat"
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Bidirectional(
        layers.GRU(256, return_sequences=True, kernel_initializer="Orthogonal", name="gru2"),
        merge_mode="concat"
    )(x)
    x = layers.Dropout(0.5)(x)

    # Dense projection to Vocabulary + Softmax
    y_pred = layers.Dense(vocab_size, activation="softmax", kernel_initializer="he_normal", name="dense_vocab")(x)

    if training_mode:
        labels = layers.Input(name="labels", shape=[None], dtype="int32")
        input_length = layers.Input(name="input_length", shape=[1], dtype="int32")
        label_length = layers.Input(name="label_length", shape=[1], dtype="int32")
        
        loss_out = CTCLossLayer()([labels, y_pred, input_length, label_length])
        model = models.Model(
            inputs=[video_input, labels, input_length, label_length],
            outputs=loss_out,
            name="LipNet_Training"
        )
    else:
        model = models.Model(inputs=video_input, outputs=y_pred, name="LipNet_Inference")

    return model
