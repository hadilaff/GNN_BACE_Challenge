import pandas as pd
from sklearn.metrics import f1_score

def calculate_macro_f1(submission_df, truth_df, config):
    """Calculates the Macro F1 score."""
    submission = submission_df.set_index(config['metrics']['id_column'])
    truth = truth_df.set_index(config['metrics']['id_column'])
    
    # Align and ensure order is the same
    common_ids = submission.index.intersection(truth.index)
    submission = submission.loc[common_ids]
    truth = truth.loc[common_ids]

    score = f1_score(truth[config['metrics']['target_column']], submission[config['metrics']['target_column']], average='macro')
    return score