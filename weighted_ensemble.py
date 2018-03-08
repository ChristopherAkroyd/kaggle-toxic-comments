import numpy as np
import pandas as pd

bi_gru_conc_pool = pd.read_csv('./results/best_submissions/BidirectionalGRUConcPool_0.9851.csv')
bi_gru_max_avg = pd.read_csv('./results/best_submissions/BidirectionalGRUMaxAvg_0.9841.csv')
gru_conc_pool = pd.read_csv('./results/best_submissions/GRUConcPool_0.9831.csv')
text_cnn = pd.read_csv('./results/best_submissions/TextCNN_0.9826.csv')
gru_attention = pd.read_csv('./results/best_submissions/BiGRUAttention_0.9721.csv')

# Models that have a low correlation with each other give us a greater boost when combined
# give a higher LB score.
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# for label in labels:
#     print(label)
#     print(np.corrcoef([bi_gru_conc_pool[label].rank(pct=True),
#                        bi_gru_max_avg[label].rank(pct=True),
#                        gru_conc_pool[label].rank(pct=True),
#                        text_cnn[label].rank(pct=True),
#                        gru_attention[label].rank(pct=True)]))

submission = pd.DataFrame()
submission['id'] = bi_gru_conc_pool['id']

for label in labels:
    submission[label] = bi_gru_conc_pool[label].rank(pct=True) * 0.5 + \
                        bi_gru_max_avg[label].rank(pct=True) * 0.15 + \
                        gru_conc_pool[label].rank(pct=True) * 0.15 + \
                        text_cnn[label].rank(pct=True) * 0.15 + \
                        gru_attention[label].rank(pct=True) * 0.05

submission.to_csv('weighted_ensemble_0.9854.csv', index=False)
