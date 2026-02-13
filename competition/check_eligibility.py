
import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Check if a user is eligible to submit.')
    parser.add_argument('username', type=str, help='The GitHub username of the submitter')
    parser.add_argument('--leaderboard', type=str, default='leaderboard/leaderboard.csv', help='Path to the leaderboard CSV')
    args = parser.parse_args()

    username = args.username
    leaderboard_path = args.leaderboard

    try:
        leaderboard = pd.read_csv(leaderboard_path)
    except FileNotFoundError:
        print(f"Leaderboard file not found at {leaderboard_path}. Assuming first submission.")
        sys.exit(0)

    # Check if 'team' column exists
    if 'team' not in leaderboard.columns:
        print("Error: 'team' column not found in leaderboard. cannot verify eligibility.")
        sys.exit(1)

    # Check if user has already submitted
    # We treat the 'team' column as the username for individual participants
    if username in leaderboard['team'].values:
        print(f"❌ Eligibility Check Failed: User '{username}' has already submitted.")
        print("recalling the Policy: Only one submission per participant is allowed.")
        sys.exit(1)
    
    print(f"✅ Eligibility Check Passed: User '{username}' has not submitted yet.")
    sys.exit(0)

if __name__ == "__main__":
    main()
