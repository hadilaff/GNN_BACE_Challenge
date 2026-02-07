import pandas as pd
import yaml
from .metrics import calculate_macro_f1

def run_evaluation(submission_path, config_path='competition/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load submission and truth
    submission_df = pd.read_csv(submission_path)
    truth_df = pd.read_csv(f"{config['data']['public_dir']}/{config['data']['test_labels_file']}")
    
    # Calculate score
    score = calculate_macro_f1(submission_df, truth_df, config)
    print(f"Evaluation complete. Macro F1 Score: {score:.4f}")
    return score