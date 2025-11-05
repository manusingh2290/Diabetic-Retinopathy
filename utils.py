import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
import os

def create_generators(dataset_path, target_size=(380, 380), batch_size=16, num_classes=5):


    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.25,
        shear_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    
    
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    test_dir = os.path.join(dataset_path, "test")

    if not os.path.exists(val_dir):
        print(" Validation folder not found — splitting from training data (20%)")
        val_split = 0.2
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=42
        )
        val_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=42
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

    
    
    if os.path.exists(test_dir):
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    else:
        print(" Test folder not found — using validation data as test set.")
        test_generator = val_generator

    return train_generator, val_generator, test_generator


def compute_class_weights(train_generator):
    """
    Computes balanced class weights to handle class imbalance.
    """
    labels = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))
    print("\n Computed Class Weights:", class_weights)
    return class_weights
