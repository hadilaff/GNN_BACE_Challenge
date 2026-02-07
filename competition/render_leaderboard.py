import pandas as pd
import yaml
import os

def render_leaderboard():
    """Reads leaderboard.csv and overwrites leaderboard.md with a formatted table."""
    
    # Load configuration
    try:
        with open('competition/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        leaderboard_path = config['leaderboard']['file']
        md_path = config['leaderboard']['md_file']
        html_path = 'docs/leaderboard.html'
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return

    # Read the CSV data
    try:
        df = pd.read_csv(leaderboard_path)
    except FileNotFoundError:
        print(f"Error: Leaderboard CSV not found at {leaderboard_path}")
        return

    # Sort by score (descending)
    df = df.sort_values(by=['score'], ascending=False)

    # Create a rank column
    df['rank'] = range(1, len(df) + 1)

    # Reorder columns for display
    display_df = df[['rank', 'team', 'model', 'score', 'timestamp_utc', 'notes']]
    display_df.columns = ['Rank', 'Team', 'Model', 'Score (Macro F1)', 'Submitted At (UTC)', 'Notes']

    # --- Generate the HTML content ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GNN Challenge Leaderboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
            h1 {{ color: #0366d6; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f6f8fa; font-weight: 600; }}
            tr:nth-child(even) {{ background-color: #f6f8fa; }}
            tr:hover {{ background-color: #eaf2ff; }}
        </style>
    </head>
    <body>
        <h1>📊 Leaderboard</h1>
        <p>This leaderboard is automatically updated after each valid submission.</p>
        {display_df.to_html(index=False, escape=False)}
    </body>
    </html>
    """

    # Write the content to the .html file in the docs folder
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Leaderboard successfully rendered to {html_path}")

if __name__ == '__main__':
    render_leaderboard()