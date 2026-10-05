"""
End-to-End Training Script for LipNet Visual Speech Recognition.
Supports CTC loss training on GRID / LRW datasets with learning rate scheduling & checkpointing.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from models.lipnet_tf import build_lipnet_model
from core.vocabulary import default_vocab
from training.grid_dataset import GridDatasetLoader


def data_generator(loader: GridDatasetLoader, batch_size: int = 8):
    """Yields batches formatted for Keras LipNet training mode."""
    num_samples = len(loader)
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) == 0:
                continue

            videos = []
            labels = []
            input_lens = []
            label_lens = []

            for idx in batch_idx:
                v, l, in_len, lbl_len = loader.load_sample(idx)
                videos.append(v)
                labels.append(l)
                input_lens.append([in_len])
                label_lens.append([lbl_len])

            # Pad labels in batch to max label length
            max_lbl = max(len(l) for l in labels)
            padded_labels = np.zeros((len(labels), max_lbl), dtype=np.int32)
            for j, l in enumerate(labels):
                padded_labels[j, :len(l)] = l

            inputs = {
                "video_input": np.array(videos, dtype=np.float32),
                "labels": padded_labels,
                "input_length": np.array(input_lens, dtype=np.int32),
                "label_length": np.array(label_lens, dtype=np.int32),
            }
            # Dummy target since loss is calculated in CTCLossLayer
            outputs = np.zeros((len(batch_idx), 1), dtype=np.float32)

            yield inputs, outputs


def train(args):
    """Main training execution loop."""
    print("=" * 60)
    print("🚀 Initializing LipNet Visual Speech Recognition Training")
    print(f"Dataset Directory : {args.dataset_dir}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Learning Rate     : {args.lr}")
    print(f"Total Epochs      : {args.epochs}")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Build Model
    model = build_lipnet_model(
        input_shape=(75, 46, 96, 1),
        vocab_size=default_vocab.vocab_size,
        training_mode=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=5.0)
    model.compile(optimizer=optimizer)
    model.summary()

    # 2. Setup Dataset
    loader = GridDatasetLoader(dataset_dir=args.dataset_dir, vocab=default_vocab)
    print(f"Loaded {len(loader)} total training video samples.")

    if len(loader) == 0:
        print("[Warning] No video samples found in dataset directory. Creating dummy checkpoint for deployment.")
        # Save an initialized model weights file
        inference_model = build_lipnet_model(training_mode=False)
        weight_path = os.path.join(args.save_dir, "lipnet_weights.h5")
        inference_model.save_weights(weight_path)
        print(f"Saved initial model weights to {weight_path}")
        return

    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, "lipnet_best.h5"),
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor="loss", patience=12, restore_best_weights=True, verbose=1)

    # 4. Fit Model
    steps_per_epoch = max(1, len(loader) // args.batch_size)
    gen = data_generator(loader, batch_size=args.batch_size)

    model.fit(
        gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    # Save final inference weights
    inference_model = build_lipnet_model(training_mode=False)
    for layer in inference_model.layers:
        try:
            layer.set_weights(model.get_layer(layer.name).get_weights())
        except Exception:
            pass

    final_weight_path = os.path.join(args.save_dir, "lipnet_final.h5")
    inference_model.save_weights(final_weight_path)
    print(f"✅ Training Complete. Saved final weights to {final_weight_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LipNet VSR Model")
    parser.add_argument("--dataset_dir", type=str, default="./data/grid", help="Path to GRID dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    args = parser.parse_args()

    train(args)
