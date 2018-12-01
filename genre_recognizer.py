from common import load_track, get_layer_output_function
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_yaml, Model
from tensorflow.keras import backend as K
from common import GENRES
import operator


class GenreRecognizer():

    def __init__(self, model_path, weights_path):
        with open(model_path, 'r') as f:
            model = model_from_yaml(f.read())
        model.load_weights(weights_path)
        self.pred_fun = get_layer_output_function(model, 'output_realtime')
        print('Loaded model.')

    def recognize(self, track_path):
        print('Loading song', track_path)
        (features, duration) = load_track(track_path)
        features = np.reshape(features, (1,) + features.shape)
        return (self.pred_fun(features), duration)

    def get_genre_distribution_over_time(self, predictions, duration):
        '''
        Turns the matrix of predictions given by a model into a dict mapping
        time in the song to a music genre distribution.
        '''
        predictions = np.reshape(predictions, predictions.shape[1:])
        n_steps = predictions.shape[0]
        delta_t = duration / n_steps

        def get_genre_distribution(step):
            return {genre_name: float(predictions[step, genre_index])
                    for (genre_index, genre_name) in enumerate(GENRES)}

        return [((step + 1) * delta_t, get_genre_distribution(step))
                for step in range(n_steps)]


def main():
    track_path = "data/genres/rock/rock.00019.au"
    # track_path = "data/03.Pressure.mp3"
    recognizer = GenreRecognizer(model_path="models/custom/model.yaml", weights_path="models/custom/weights.h5")
    (predictions, duration) = recognizer.recognize(track_path)
    print(predictions)

    genre_distributions_over_time = recognizer.get_genre_distribution_over_time(
        predictions, duration)

    print("Genre Distributions over time: ")
    print(genre_distributions_over_time)
    distributions = [list(elem)[1] for elem in genre_distributions_over_time]
    distribution = {genre_name: sum(item[genre_name] for item in distributions) / len(distributions)
                    for (genre_index, genre_name) in enumerate(GENRES)}
    print("Distribution: " + str(distribution))
    print("Assumed genre is: " + max(distribution.items(), key=operator.itemgetter(1))[0])


if __name__ == '__main__':
    main()
