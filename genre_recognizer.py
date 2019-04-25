import operator

import numpy as np
import tensorflow as tf
from common import GENRES
from common import load_track, get_layer_output_function
import matplotlib.pyplot as plt


class GenreRecognizer:

    def __init__(self, model_path):
        model = tf.keras.models.load_model(model_path)
        self.pred_fun = get_layer_output_function(model, 'output_realtime')
        print('Loaded model.')

    def recognize(self, track_path):
        print('Loading song', track_path)
        (features, duration) = load_track(track_path)
        features = np.reshape(features, (1,) + features.shape)
        return self.pred_fun(features), duration


def main():
    track_path = "data/genres/rock/rock.00020.au"
    # track_path = "data/03.Pressure.mp3"
    recognizer = GenreRecognizer(model_path="models/model.h5")
    (predictions, duration) = recognizer.recognize(track_path)
    distribution = {genre_name: predictions[0][genre_index]
                    for (genre_index, genre_name) in enumerate(GENRES)}
    print("Distribution: " + str(distribution))
    print("Assumed genre is: " + max(distribution.items(), key=operator.itemgetter(1))[0])
    plt.pie(predictions[0], labels=GENRES, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.show()


if __name__ == '__main__':
    main()
