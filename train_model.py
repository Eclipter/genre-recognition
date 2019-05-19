import os
import pickle
from optparse import OptionParser
from sklearn.model_selection import train_test_split
import tensorflow as tf

from common import GENRES

REGULARIZATION_RATE = 0.01

BATCH_NORM_MOMENTUM = 0.9

DROPOUT_RATE = 0.5

POOL_SIZE = 2

SEED = 42
VAL_SIZE = 0.1
N_LAYERS = 3
CONV_KERNEL_SIZE = 5
CONV_FILTER_COUNT = 256
LSTM_UNITS_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 150


def train_model(train_data, model_path):
    x = train_data['x']
    y = train_data['y']
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=VAL_SIZE, random_state=SEED)

    print('Building model...')

    n_features = x_train.shape[2]
    input_shape = (None, n_features)
    model_input = tf.keras.layers.Input(input_shape, name='input')
    layer = model_input
    for i in range(N_LAYERS):
        layer = tf.keras.layers.Convolution1D(
            filters=CONV_FILTER_COUNT,
            kernel_size=CONV_KERNEL_SIZE,
            name='convolution_' + str(i + 1),
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE)
        )(layer)
        layer = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM)(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.MaxPooling1D(POOL_SIZE)(layer)
        layer = tf.keras.layers.Dropout(DROPOUT_RATE)(layer)

    layer = tf.keras.layers.LSTM(
        units=LSTM_UNITS_COUNT,
        kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION_RATE)
    )(layer)
    layer = tf.keras.layers.Dense(units=len(GENRES))(layer)
    layer = tf.keras.layers.Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = tf.keras.models.Model(model_input, model_output)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    print('Training...')
    model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH_COUNT,
        validation_data=(x_val, y_val), verbose=1, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                model_path, save_best_only=True, monitor='val_acc', verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
        ]
    )

    return model


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--data_path', dest='data_path',
                      default=os.path.join(os.path.dirname(__file__),
                                           'data/data.pkl'),
                      help='path to the data pickle', metavar='DATA_PATH')
    parser.add_option('-m', '--model_path', dest='model_path',
                      default=os.path.join(os.path.dirname(__file__),
                                           'models/model.h5'),
                      help='path to the output model HDF5 file', metavar='MODEL_PATH')
    options, args = parser.parse_args()

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    train_model(data, options.model_path)
