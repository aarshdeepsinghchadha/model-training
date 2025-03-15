import pandas as pd
import psycopg2
import re
from datetime import datetime
import os
from tqdm import tqdm
from sqlalchemy import create_engine  # Add this import for SQLAlchemy

# Directory setup (unchanged from v5)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
output_dir = os.path.join(root_dir, 'data', 'output', 'v5')
csv_folder = os.path.join(output_dir, 'csv_files')

# PostgreSQL connection details (for local pgAdmin)
db_config = {
    'dbname': 'timelogs_db',         # Database name (create this in pgAdmin)
    'user': 'postgres',             # Default username (change if different)
    'password': 'password',    # Replace with your PostgreSQL password
    'host': 'localhost',            # Localhost for local pgAdmin
    'port': '5432'                  # Default PostgreSQL port
}
dbpath = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# Setup PostgreSQL database and load CSV data
def setup_database():
    # Connect with psycopg2 for table creation
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_timelogs (
            username TEXT,
            date DATE,
            year INTEGER,
            month INTEGER,
            projectid INTEGER,
            projectname TEXT,
            timelog INTEGER
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_forecasts (
            ds DATE,
            yhat FLOAT,
            yhat_lower FLOAT,
            yhat_upper FLOAT,
            username TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_forecasts (
            ds DATE,
            yhat FLOAT,
            yhat_lower FLOAT,
            yhat_upper FLOAT,
            projectname TEXT
        );
    """)

    conn.commit()
    conn.close()

    # Use SQLAlchemy engine for to_sql
    engine = create_engine(dbpath)

    # Load CSV data dynamically
    timelogs = pd.read_csv(os.path.join(root_dir, 'data', 'v4', 'user_timelogs_v4.csv'))
    timelogs.to_sql('historical_timelogs', engine, if_exists='replace', index=False)
    
    user_forecasts = pd.read_csv(os.path.join(csv_folder, 'user_forecasts_2025_hybrid_v5.csv'))
    user_forecasts.to_sql('user_forecasts', engine, if_exists='replace', index=False)
    
    project_forecasts = pd.read_csv(os.path.join(csv_folder, 'project_forecasts_2025_hybrid_v5.csv'))
    project_forecasts.to_sql('project_forecasts', engine, if_exists='replace', index=False)

    print("Database setup complete with CSV data loaded.")

# Fetch all usernames
def get_all_usernames():
    conn = psycopg2.connect(**db_config)
    query = "SELECT DISTINCT username FROM historical_timelogs ORDER BY username;"
    print(f"SQL Query: {query}")
    usernames = pd.read_sql_query(query, conn)
    conn.close()
    return usernames['username'].tolist()

# Parse user prompt and generate SQL query
def parse_prompt(prompt, username):
    conn = psycopg2.connect(**db_config)

    # Extract key components using regex
    year_match = re.search(r'(\d{4})', prompt)
    project_match = re.search(r'Project\s\w+', prompt, re.IGNORECASE)
    historical = "hours for" in prompt.lower()
    predict = "predict" in prompt.lower()

    year = year_match.group(1) if year_match else None
    project = project_match.group(0) if project_match else None

    # Build SQL query dynamically
    historical_result, predict_result = None, None
    if historical and year:
        query = f"SELECT username, year, month, SUM(timelog) as total_hours FROM historical_timelogs WHERE username = %s AND year = %s"
        params = [username, year]
        if project:
            query += " AND projectname = %s"
            params.append(project)
        query += " GROUP BY username, year, month ORDER BY month"
        print(f"SQL Query: {query % tuple(params)}")
        historical_result = pd.read_sql_query(query, conn, params=params)

    if predict and year == "2025":
        table = 'project_forecasts' if project else 'user_forecasts'
        group_col = 'projectname' if project else 'username'
        query = f"SELECT ds, yhat, yhat_lower, yhat_upper FROM {table} WHERE {group_col} = %s AND EXTRACT(YEAR FROM ds) = %s"
        params = [project or username, year]
        print(f"SQL Query: {query % tuple(params)}")
        predict_result = pd.read_sql_query(query, conn, params=params)

    conn.close()
    return historical_result, predict_result

# Main function to handle user interaction
def handle_user_prompt():
    # Show all usernames
    usernames = get_all_usernames()
    print("\nAvailable Usernames:")
    print(", ".join(usernames))

    # Get username from user
    while True:
        username = input("\nEnter your username: ").strip()
        if username in usernames:
            break
        print("Username not found. Please try again.")

    # Get prompt from user
    prompt = input(f"Enter your request (e.g., 'Show my hours for 2022', 'Predict my hours for Project Alpha in 2025'): ").strip()
    print(f"\nProcessing prompt: '{prompt}' for user '{username}'")

    historical_result, predict_result = parse_prompt(prompt, username)

    if historical_result is not None and not historical_result.empty:
        print(f"\nHistorical Hours for {username}:")
        print(historical_result)
    elif historical_result is not None:
        print(f"No historical data found for {username} matching your request.")

    if predict_result is not None and not predict_result.empty:
        print(f"\nPredicted Hours for {username} in 2025:")
        print(predict_result)
    elif predict_result is not None:
        print(f"No predictions found for {username} matching your request.")
    else:
        print("Prediction not applicable or year not 2025.")

# Setup database if not already done
setup_database()

# Run the interactive prompt
handle_user_prompt()
