import numpy as np
import pandas as pd

WRITE_COEFFICIENTS = True
WRITE_RESULTS = True


bi_gru_conc_pool = pd.read_csv('./results/best_submissions/BidirectionalGRUConcPool_0.9851.csv')
double_bi_gru_conc_pool = pd.read_csv('./results/best_submissions/DoubleBiGRUConcPool_0.9843.csv')
bi_gru_max_avg = pd.read_csv('./results/best_submissions/BidirectionalGRUMaxAvg_0.9841.csv')
gru_conc_pool = pd.read_csv('./results/best_submissions/GRUConcPool_0.9831.csv')
text_cnn = pd.read_csv('./results/best_submissions/TextCNN_0.9826.csv')
lstm_cnn = pd.read_csv('./results/best_submissions/LSTMCNN_0.9804.csv')
gru_attention = pd.read_csv('./results/best_submissions/BiGRUAttention_0.9721.csv')

# Models that have a low correlation with each other give us a greater boost when combined
# give a higher LB score.
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def correlation_coefficients():
    for label in labels:
        print(label)
        print(np.corrcoef([bi_gru_conc_pool[label].rank(pct=True),
                           double_bi_gru_conc_pool[label].rank(pct=True),
                           bi_gru_max_avg[label].rank(pct=True),
                           gru_conc_pool[label].rank(pct=True),
                           text_cnn[label].rank(pct=True),
                           lstm_cnn[label].rank(pct=True),
                           gru_attention[label].rank(pct=True)]))


def weighted_ensemble_predictions():
    submission = pd.DataFrame()
    submission['id'] = bi_gru_conc_pool['id']

    for label in labels:
        submission[label] = bi_gru_conc_pool[label].rank(pct=True) * 0.3 + \
                            double_bi_gru_conc_pool[label].rank(pct=True) * 0.15 + \
                            bi_gru_max_avg[label].rank(pct=True) * 0.15 + \
                            gru_conc_pool[label].rank(pct=True) * 0.15 + \
                            text_cnn[label].rank(pct=True) * 0.125 + \
                            lstm_cnn[label].rank(pct=True) * 0.1 + \
                            gru_attention[label].rank(pct=True) * 0.025

    submission.to_csv('weighted_average.csv', index=False)


def voting_ensemble_predictions():
    submission = pd.DataFrame()
    submission['id'] = bi_gru_conc_pool['id']

    for label in labels:
        best = bi_gru_conc_pool[label].rank(pct=True) * 3

        submission[label] = (best +
                             double_bi_gru_conc_pool[label].rank(pct=True) +
                             bi_gru_max_avg[label].rank(pct=True) +
                             gru_conc_pool[label].rank(pct=True) +
                             text_cnn[label].rank(pct=True) +
                             lstm_cnn[label].rank(pct=True) +
                             gru_attention[label].rank(pct=True)) / 9

    submission.to_csv('voting_average.csv', index=False)


if __name__ == "__main__":
    if WRITE_COEFFICIENTS:
        correlation_coefficients()
    if WRITE_RESULTS:
        weighted_ensemble_predictions()
        voting_ensemble_predictions()
