import numpy as np
from sklearn.preprocessing import minmax_scale

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
def write_results(model, test_set, df_submission, trickery='power_scale'):
    print('Starting to write results...')

    print('Running ' + str(len(test_set)) + ' predictions...')

    if type(model) == list:
        fold_predictions = []
        for i in range(len(model)):
            print('Running Fold ' + str(i) + ' predictions...')

        predictions = np.ones(len(model))
        for fold in fold_predictions:
            predictions *= fold

        predictions **= (1. / len(model))
    else:
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
