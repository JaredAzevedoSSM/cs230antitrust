"""
File: model.py

Authors: Jared Azevedo & Andres Suarez

Desc: implement baseline model (logistic regression)
"""
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd


def custom_crossentropy_b(y_true, y_pred):
    """
    Desc: returns a high value (penalty) in case of mismatches between the true and predicted values. 
    """
    return - tf.math.reduce_mean(tf.math.reduce_sum((y_true/(y_pred + 0.01))+(1-y_true)/(1-y_pred + 0.01)))


def custom_crossentropy_pos(y_true, y_pred):
    """
    Desc: Similar to binary crossentropy, but weight towards positive labels - GIVES GOOD RECALL
    """
    return -tf.math.reduce_mean(tf.math.reduce_sum((tf.math.log(2 + y_pred) * (1 + y_true)) + (tf.math.log(1 + y_pred) * y_true)))


def custom_true_label(y_true, y_pred):
    """
    Desc: Simply returns value of label when positive - GIVES GOOD ACCURACY
    """
    return y_true * y_pred


def custom_mse_pos(y_true, y_pred):
    """
    Desc: Mean squared error loss with extra weight towards positive labels - GIVES GREAT RECALL
    """
    return tf.math.reduce_mean(tf.square(y_true - y_pred + 0.5))


def custom_mse_neg(y_true, y_pred):
    """
    Desc: Mean squared error loss with extra weight towards positive labels - GIVES GOOD ACCURACY
    """
    return tf.math.reduce_mean(tf.square(y_true - y_pred - 0.5))


def split_dataset_from_csv(path_to_dataset, fraction_of_data_to_train):
    """
    Desc: divides the data in train, val, and test sets; splits each database between features and labels; 
    finally, converts the three dataset to type tensor
    """
    # Read the dataset from CSV
    compiled_data = pd.read_csv(path_to_dataset)

    # Normalize each column (min-max)
    # for col in compiled_data.columns:
    #     if col == "NEWID" or col == "PURCHASED":
    #         continue
    #     compiled_data[col] = (compiled_data[col] - compiled_data[col].min()) / (compiled_data[col].max() - compiled_data[col].min())

    # Split the dataset into train, vali, and test sets
    train_dataset = compiled_data.sample(frac=fraction_of_data_to_train, random_state=1)
    val_dataset = compiled_data.drop(train_dataset.index).sample(frac=0.5, random_state=1)
    test_dataset = compiled_data.drop(train_dataset.index).drop(val_dataset.index)

    # Split each subset into features and labels; note that we don't keep NEWID since that would be like keeping a name
    x_train, y_train = train_dataset.iloc[:, 2:], train_dataset.iloc[:, 0]
    x_val, y_val = val_dataset.iloc[:, 2:], val_dataset.iloc[:, 0]
    x_test, y_test = test_dataset.iloc[:, 2:], test_dataset.iloc[:, 0]    

    # Cast our subsets into tensors
    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_val, y_val = tf.convert_to_tensor(x_val, dtype=tf.float32), tf.convert_to_tensor(y_val, dtype=tf.float32)
    x_test, y_test = tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)

    return {'train':{'x': x_train,'y': y_train}, 'val':{'x': x_val, 'y': y_val}, 'test':{'x': x_test, 'y': y_test}}


def get_uncompiled_model(path_to_dataset, fraction_of_data_to_train):
    """
    Desc: create an uncompiled instance of our logistic regression model
    """
    # Split dataset into subsets
    split_dataset = split_dataset_from_csv(path_to_dataset, fraction_of_data_to_train)

    # Insert input, hidden layers, and output into our model
    inputs = keras.Input(shape=(split_dataset['train']['x'].shape[1]))
    x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = keras.layers.Dropout(0.25, name='dropout_1')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    x = keras.layers.Dropout(0.25, name='dropout_2')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_3')(x)
    x = keras.layers.Dropout(0.25, name='dropout_3')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_4')(x)
    x = keras.layers.Dropout(0.25, name='dropout_4')(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    # Create model instance
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model, split_dataset


def get_compiled_model(path_to_dataset, fraction_of_data_to_train):
    """
    Desc: compile the model
    """
    # Get uncompiled model
    model, split_dataset = get_uncompiled_model(path_to_dataset, fraction_of_data_to_train)

    # Compile the model; note that this is where to change abstract parts of the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss=custom_mse_pos,
        metrics=[keras.metrics.Accuracy(), keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()],
    )

    return model, split_dataset


def build_and_run_model(path_to_dataset, fraction_of_data_to_train=0.9, batch_sz=128, nepochs=25):
    """
    Desc: create and run the model on the provided dataset (given via a path)
    """
    # Get compiled model
    model, split_dataset = get_compiled_model(path_to_dataset, fraction_of_data_to_train)

    # Train model
    print("Fitting model on training data...")
    history = model.fit(
        split_dataset['train']['x'],
        split_dataset['train']['y'],
        batch_size = batch_sz,
        epochs = nepochs,
        validation_data=(split_dataset['val']['x'], split_dataset['val']['y']),
    )

    # Evaluate model
    print("Evaluating on test data...")
    results = model.evaluate(split_dataset['test']['x'], split_dataset['test']['y'], batch_size=batch_sz)
    # Add F1: \nTest F1: {(2 * results[3] * results[4]) / (results[3] + results[4])}
    print(f"""\nTest loss: {results[0]} \nTest accuracy: {results[1] * 100} \nTest AUROC: {results[2]} \nTest precision: {results[3] * 100} \nTest recall: {results[4] * 100}""")


def main(args):
    """
    Desc: build and test logistic regression model
    """
    path_to_dataset = args[0]

    # Change hyperparameters here
    build_and_run_model(path_to_dataset, 0.9, 128, 25)


if __name__ == '__main__':
    # Capture path to dataset as input argument from command line
    args = sys.argv[1:]

    # Check we have right number of args before proceeding
    if len(args) < 1:
        print("Please enter the path to a directory (ex: ../data/diary21/diary21_merged.csv)")
    else:
        # Run processing
        main(args)
