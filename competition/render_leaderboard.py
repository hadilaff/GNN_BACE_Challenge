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

    # HTML content with modern styling linking to external CSS
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GNN Challenge Leaderboard</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="leaderboard.css">
    </head>
    <body class="bg-gray-50">
        <div class="container">
            <header>
                <h1>📊 Leaderboard</h1>
                <p class="subtitle">Real-time standings for the GNN BACE-1 Inhibition Challenge</p>
            </header>
            
            <div class="table-container">
                {display_df.to_html(index=False, escape=False, classes='leaderboard-table')}
            </div>

            <div class="footer">
                <p>Updated automatically after each valid submission. Last calculation: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                <p><a href="https://github.com/hadilaff/GNN_BACE_Challenge" style="color: var(--accent-color); text-decoration: none;">View Challenge Repository</a></p>
            </div>
        </div>
        
        <script>
            // Enhance the pandas table with badges and formatting
            document.addEventListener('DOMContentLoaded', () => {{
                const rows = document.querySelectorAll('tbody tr');
                rows.forEach(row => {{
                    const rankCell = row.cells[0];
                    const rankValue = rankCell.textContent.trim();
                    const rank = parseInt(rankValue);
                    
                    if (!isNaN(rank) && rank <= 3) {{
                        rankCell.innerHTML = `<span class="rank-badge rank-${{rank}}">${{rank}}</span>`;
                    }}
                    
                    const scoreCell = row.cells[3];
                    scoreCell.classList.add('score-cell');
                    
                    const timeCell = row.cells[4];
                    timeCell.classList.add('timestamp');
                }});
            }});
        </script>
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