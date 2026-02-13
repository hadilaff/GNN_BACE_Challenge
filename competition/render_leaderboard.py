import pandas as pd
import yaml
import os

def render_leaderboard():
    try:
        with open('competition/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        leaderboard_path = config['leaderboard']['file']
        html_path = 'docs/leaderboard.html'
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return

    # Read the CSV data
    try:
        df = pd.read_csv(leaderboard_path)
    except Exception as e:
        print(f"Error reading or parsing CSV: {e}")
        return

    # Sort by score (descending)
    df = df.sort_values(by=['score'], ascending=False)

    # Create a rank column
    df['rank'] = df['score'].rank(method='min', ascending=False).astype(int)

    # Reorder columns for display to match the CSV header
    display_df = df[['rank', 'team', 'model', 'score', 'timestamp_utc', 'notes']]
    display_df.columns = ['Rank', 'Team', 'Model', 'Score (Macro F1)', 'Submitted At (UTC)', 'Notes']

    # ... (HTML generation part is fine) ...
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GNN Challenge Leaderboard</title>
        <style> ... </style>
    </head>
    <body>
        <h1>📊 Leaderboard</h1>
        <p>This leaderboard is automatically updated after each valid submission.</p>
        {display_df.to_html(index=False, escape=False)}
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    # Save as Markdown
    md_path = 'leaderboard/leaderboard.md'
    with open(md_path, 'w') as f:
        f.write("# 📊 Leaderboard\n\n")
        f.write("This leaderboard is automatically updated after each valid submission.\n\n")
        f.write(display_df.to_markdown(index=False))
    
    print(f"Leaderboard successfully rendered to {html_path} and {md_path}")

if __name__ == '__main__':
    render_leaderboard()