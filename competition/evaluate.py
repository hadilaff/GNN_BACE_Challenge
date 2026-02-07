# competition/evaluate.py
import pandas as pd
import yaml
from competition.metrics import calculate_macro_f1 # Assuming metrics.py is in the same dir

def run_evaluation(submission_path, config_path='competition/config.yaml'):
    """Runs the evaluation and returns the score."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    submission_df = pd.read_csv(submission_path)
    truth_df = pd.read_csv(f"{config['data']['public_dir']}/{config['data']['test_labels_file']}")
    
    score = calculate_macro_f1(submission_df, truth_df, config)
    return score

# This part allows it to be run from the command line for testing
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <submission_file.csv>")
        sys.exit(1)
    
    submission_path = sys.argv[1]
    score = run_evaluation(submission_path)
    print(f"Macro F1 Score: {score:.4f}")