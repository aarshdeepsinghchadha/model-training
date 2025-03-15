import pandas as pd
import spacy
import os
from datetime import datetime
from sqlalchemy import create_engine
from tqdm import tqdm

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")  # Install: python -m spacy download en_core_web_sm

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
output_dir = os.path.join(root_dir, 'data', 'output', 'v5')
csv_folder = os.path.join(output_dir, 'csv_files')

# PostgreSQL connection details
db_config = {
    'dbname': 'timelogs_db',
    'user': 'postgres',
    'password': 'password',  # Replace with your PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}
dbpath = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# Setup PostgreSQL database and load CSV data
def setup_database():
    conn = create_engine(dbpath)
    timelogs = pd.read_csv(os.path.join(root_dir, 'data', 'v4', 'user_timelogs_v4.csv'))
    timelogs.to_sql('historical_timelogs', conn, if_exists='replace', index=False)
    
    user_forecasts = pd.read_csv(os.path.join(csv_folder, 'user_forecasts_2025_hybrid_v5.csv'))
    user_forecasts.to_sql('user_forecasts', conn, if_exists='replace', index=False)
    
    project_forecasts = pd.read_csv(os.path.join(csv_folder, 'project_forecasts_2025_hybrid_v5.csv'))
    project_forecasts.to_sql('project_forecasts', conn, if_exists='replace', index=False)
    
    projects = pd.read_csv(os.path.join(root_dir, 'data', 'v4', 'projects_v4.csv'))
    projects.to_sql('projects', conn, if_exists='replace', index=False)

    print("Database setup complete with CSV data loaded.")

# Fetch all usernames
def get_all_usernames(engine):
    query = "SELECT DISTINCT username FROM historical_timelogs ORDER BY username;"
    print(f"SQL Query: {query}")
    return pd.read_sql_query(query, engine)['username'].tolist()

# Fetch all project names
def get_all_projects(engine):
    query = "SELECT projectname FROM projects ORDER BY projectname;"
    print(f"SQL Query: {query}")
    return pd.read_sql_query(query, engine)['projectname'].tolist()

# Parse prompt using NLP and generate SQL query
def parse_prompt(prompt, username, engine, projects):
    doc = nlp(prompt.lower())
    
    # Extract entities and intent
    intent = None
    year = None
    project = None
    
    # Intent detection (basic rules for now, can be ML-based later)
    if "predict" in prompt:
        intent = "predict"
    elif "hours" in prompt and any(token.text.isdigit() for token in doc):
        intent = "historical_hours"
    elif "project" in prompt or "worked" in prompt:
        intent = "list_projects"

    # Extract year
    for token in doc:
        if token.text.isdigit() and 2000 <= int(token.text) <= 2025:
            year = token.text

    # Extract project name
    for p in projects:
        if p.lower() in prompt.lower():
            project = p
            break

    # Dynamic SQL query generation
    result = None
    if intent == "historical_hours" and year:
        query = "SELECT username, year, month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s"
        params = (username, year)
        if project:
            query += " AND LOWER(projectname) = LOWER(%s)"
            params = (username, year, project)
        query += " GROUP BY username, year, month ORDER BY month"
        print(f"Generated SQL Query: {query % params}")
        result = pd.read_sql_query(query, engine, params=params)

    elif intent == "predict" and year == "2025":
        table = 'project_forecasts' if project else 'user_forecasts'
        group_col = 'projectname' if project else 'username'
        query = f"SELECT ds, yhat, yhat_lower, yhat_upper FROM {table} WHERE LOWER({group_col}) = LOWER(%s) AND EXTRACT(YEAR FROM ds::DATE) = %s"
        params = (project or username, year)
        print(f"Generated SQL Query: {query % params}")
        result = pd.read_sql_query(query, engine, params=params)

    elif intent == "list_projects":
        query = "SELECT DISTINCT projectname FROM historical_timelogs WHERE username = %s ORDER BY projectname"
        params = (username,)
        print(f"Generated SQL Query: {query % params}")
        result = pd.read_sql_query(query, engine, params=params)

    return intent, result

# Main function to handle user interaction
def handle_user_prompt():
    engine = create_engine(dbpath)

    usernames = get_all_usernames(engine)
    print("\nAvailable Usernames:")
    print(", ".join(usernames))

    projects = get_all_projects(engine)
    print("\nAvailable Projects:")
    print(", ".join(projects))

    while True:
        username = input("\nEnter your username: ").strip()
        if username in usernames:
            break
        print("Username not found. Please try again.")

    while True:
        prompt = input(f"\nEnter your request for {username} (e.g., 'Show my hours for 2022', 'Predict my hours for Project Alpha in 2025', 'List all projects I worked on') or type 'exit' to quit: ").strip()
        
        if prompt.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
        
        print(f"\nProcessing prompt: '{prompt}' for user '{username}'")
        intent, result = parse_prompt(prompt, username, engine, projects)

        if result is not None and not result.empty:
            if intent == "historical_hours":
                print(f"\nHistorical Hours for {username}:")
            elif intent == "predict":
                print(f"\nPredicted Hours for {username} in 2025:")
            elif intent == "list_projects":
                print(f"\nProjects {username} has worked on:")
            print(result)
        elif result is not None:
            print(f"No data found for {username} matching your request.")
        else:
            print("Request not recognized. Try something like 'Show my hours for <year>', 'Predict my hours for <project> in 2025', or 'List all projects I worked on'.")

# Setup database if not already done
setup_database()

# Run the interactive prompt
handle_user_prompt()