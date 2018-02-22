def write_results(model, test_set, df_submission):
    print('Running ' + str(len(test_set)) + ' predictions...')

    predictions = model.predict(test_set)

    assert len(predictions) == len(test_set)

    print('Writing ' + str(len(predictions)) + ' predictions...')

    df_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = predictions
    df_submission.to_csv('submission.csv', index=False)
