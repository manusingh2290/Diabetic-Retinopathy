import argparse
import os
import math
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from utils import create_generators, compute_class_weights


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB4 for Diabetic Retinopathy Detection")
    parser.add_argument('--dataset', required=True, help='Path to dataset folder (train/val/test)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--save_dir', default='models', help='Directory to save trained models')
    return parser.parse_args()


# ---------------------------------------------------------
# Build Model Function
# ---------------------------------------------------------
def build_efficientnet(input_shape=(380, 380, 3), num_classes=5):
    print("🔧 Loading EfficientNetB4 base model (ImageNet pretrained)...")
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze most layers, fine-tune top 50
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n📂 Loading dataset from: {args.dataset}")
    train_loader, val_loader, test_loader = create_generators(
        args.dataset, target_size=(380, 380), batch_size=args.batch_size, num_classes=args.num_classes
    )

    print("\n🧮 Computing class weights...")
    class_weights = compute_class_weights(train_loader)

    print("\n✅ Building EfficientNetB4 model...")
    model = build_efficientnet(input_shape=(380, 380, 3), num_classes=args.num_classes)
    model.summary()

    # ---------------------------------------------------------
    # Optimizer and Loss (Focal Loss)
    # ---------------------------------------------------------
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-6
    )

    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(args.save_dir, "final2_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    csv_logger = callbacks.CSVLogger('training_log.csv', append=True)
    tensorboard = callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)

    # ---------------------------------------------------------
    # Train Model
    # ---------------------------------------------------------
    print("\n🚀 Starting 30-epoch full fine-tuning with class balancing...\n")

    history = model.fit(
        train_loader,
        epochs=args.epochs,
        validation_data=val_loader,
        class_weight=class_weights,
        callbacks=[checkpoint, csv_logger, tensorboard, reduce_lr],
        verbose=1
    )

    # ---------------------------------------------------------
    # Save Final Model
    # ---------------------------------------------------------
    final_path = os.path.join(args.save_dir, "final2.h5")
    model.save(final_path)
    print(f"\n✅ Training complete. Model saved at: {final_path}")


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
