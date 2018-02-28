import numpy as np
from sklearn.preprocessing import minmax_scale
from keras.models import load_model
from src.util import get_save_path

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# Leaderboard trickery functions that normalise the models output..
def min_max_scale(predictions):
    for label in labels:
        print('Scaling {}...'.format(label))
        predictions[label] = minmax_scale(predictions[label])
    return predictions


def power_scale(predictions, power=1.4):
    for label in labels:
        print('Scaling {}...'.format(label))
        predictions[label] = predictions[label] ** power
    return predictions


# Write out the results
def write_results(model_instance, test_set, df_submission, trickery='power_scale', folds=10, custom_objects={}):
    print('Starting to write results...')

    print('Running ' + str(len(test_set)) + ' predictions...')

    if folds > 0:
        fold_predictions = []
        # Load the first fold so we don't have to reinitialise the model each time == no more OOM for cross-validation.
        model = load_model(get_save_path(model_instance, fold=0), custom_objects)
        # Run predictions on a per-fold basis.
        for i in range(folds):
            print('Running Fold ' + str(i) + ' predictions...')
            model.load_weights(get_save_path(model_instance, fold=i))
            pred = model.predict(test_set)
            fold_predictions.append(pred)

        predictions = np.ones(fold_predictions[0].shape)
        for fold in fold_predictions:
            predictions *= fold

        predictions **= (1. / folds)
    else:
        model = load_model(get_save_path(model_instance), custom_objects)
        predictions = model.predict(test_set)

    assert len(predictions) == len(test_set)

    if trickery:
        print('Scaling ' + str(len(predictions)) + ' predictions...')
        if trickery == 'power_scale':
            df_submission = power_scale(df_submission)
        elif trickery == 'min_max':
            df_submission = min_max_scale(predictions)

    print('Writing ' + str(len(predictions)) + ' predictions...')

    df_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = predictions

    df_submission.to_csv('submission.csv', index=False)
