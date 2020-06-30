import tensorflow as tf
import os
import json
import argparse
import math
import sys as _sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam

def _model_fn(params):

    model = Sequential([
        Reshape((50,50,3), input_shape=(7500,)),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Flatten(),
        Dense(units=256, activation='relu'),
        BatchNormalization(),
        Dropout(rate=0.5),
        Dense(units=128, activation='relu'),
        BatchNormalization(),
        Dropout(rate=0.5),
        Dense(units=16, activation='relu'),
        BatchNormalization(),
        Dropout(rate=0.5),
        Dense(units=params['num_classes'], activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def _read_and_decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example, 
        features = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })

    image = tf.io.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1./255)
    label = tf.cast(features['label'], tf.int32)

    return image, label

def _load_data(input_dir, input_filename, mini_batch_size):
    file_path = os.path.join(input_dir, input_filename)
    
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(_read_and_decode)
    dataset = dataset.batch(mini_batch_size)
    dataset = dataset.repeat()
    
    return dataset

def _load_training_data(training_dir, mini_batch_size):
    return _load_data(training_dir, 'images_train.tfrecords', mini_batch_size)
    
def _load_test_data(eval_dir, mini_batch_size):
    return _load_data(eval_dir, 'images_test.tfrecords', mini_batch_size)

def _get_steps_per_epoch(number_of_samples, mini_batch_size):
    return int(math.ceil(1. * number_of_samples / mini_batch_size))

def _parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--mini-batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train-size', type=int)
    parser.add_argument('--test-size', type=int)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    
    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()
    
    # Create the model
    model = _model_fn(params = {
        'num_classes': args.num_classes,
        'learning_rate': args.learning_rate
    })
    
    # Load data
    train_dataset = _load_training_data(args.train, args.mini_batch_size)
    test_dataset = _load_test_data(args.test, args.mini_batch_size)
    
    # Train model
    model.fit(
        train_dataset, epochs=args.epochs, validation_data=test_dataset,
        steps_per_epoch=_get_steps_per_epoch(args.train_size, args.mini_batch_size),
        validation_steps=_get_steps_per_epoch(args.test_size, args.mini_batch_size),
        verbose=2
    )
    
    if args.current_host == args.hosts[0]:
        model.save(os.path.join(args.sm_model_dir, '000000001'))