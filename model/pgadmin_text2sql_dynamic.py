import pandas as pd
import spacy
import os
from datetime import datetime
from sqlalchemy import create_engine
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import re

# Load spaCy model for basic NLP (fallback)
nlp = spacy.load("en_core_web_sm")

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
output_dir = os.path.join(root_dir, 'data', 'output', 'v5')
csv_folder = os.path.join(output_dir, 'csv_files')
model_dir = os.path.join(script_dir, 't5_text2sql_model')

# PostgreSQL connection details
db_config = {
    'dbname': 'timelogs_db',
    'user': 'postgres',
    'password': 'password',  # Replace with your PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}
dbpath = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# Schema context for the model
SCHEMA_CONTEXT = "schema: historical_timelogs(username, year, month, projectname, timelog), user_forecasts(ds, yhat, yhat_lower, yhat_upper, username), project_forecasts(ds, yhat, yhat_lower, yhat_upper, projectname), projects(projectid, projectname)"

# Training data (parameterized for any user and year)
training_data = [
    {"prompt": "list me all the projects i have worked on", "sql": "SELECT DISTINCT projectname FROM historical_timelogs WHERE username = %s ORDER BY projectname", "params": ["username"]},
    {"prompt": "show me the logged hours for year %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "which month in year %s i worked the most", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "which month in year %s i worked the most and on which project", "sql": "SELECT month, projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month, projectname ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "show me all the projects associated to me with total hours logged by me", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s GROUP BY projectname ORDER BY projectname", "params": ["username"]},
    {"prompt": "who is the user who has logged most hours in %s", "sql": "SELECT username, SUM(timelog) AS total_hours FROM historical_timelogs WHERE year = %s GROUP BY username ORDER BY total_hours DESC LIMIT 1", "params": ["year"]},
    {"prompt": "which project had most logged hours for %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE year = %s GROUP BY projectname ORDER BY total_hours DESC LIMIT 1", "params": ["year"]},
    {"prompt": "which project had most logged hours for %s and in which month", "sql": "SELECT projectname, month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE year = %s GROUP BY projectname, month ORDER BY total_hours DESC LIMIT 1", "params": ["year"]}
]

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

# Train the T5 model
def train_model():
    if os.path.exists(model_dir):
        print("Model already trained. Loading existing model...")
        return
    
    # Load T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare data for T5
    def preprocess_data(examples):
        inputs = [f"{row['prompt']} | {SCHEMA_CONTEXT}" for row in examples]
        targets = [row['sql'] for row in examples]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert to Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(training_data))
    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./t5_text2sql",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train and save
    trainer.train()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Model training complete and saved.")

# Generate SQL query using T5 model
def generate_sql(prompt, username, tokenizer, model):
    # Extract year from prompt if present
    year_match = re.search(r'\b(20\d{2})\b', prompt)
    year = year_match.group(1) if year_match else None

    # Replace placeholders in prompt
    input_prompt = prompt.replace("i have", "user has").replace("me", "user")  # Normalize to "user"
    if year:
        input_prompt = input_prompt.replace(year, "%s")
    
    input_text = f"{input_prompt} | {SCHEMA_CONTEXT}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    outputs = model.generate(**inputs, max_length=128)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Replace placeholders with actual values
    params = []
    if "%s" in sql_query:
        if "username" in sql_query.lower() and year:
            params = [username, year]
            sql_query = sql_query % (username, year)
        elif "username" in sql_query.lower():
            params = [username]
            sql_query = sql_query % username
        elif year:
            params = [year]
            sql_query = sql_query % year
    
    return sql_query, params

# Parse prompt and execute query
def parse_prompt(prompt, username, engine, projects, tokenizer, model):
    doc = nlp(prompt.lower())
    
    # Use T5 model to generate SQL
    sql_query, params = generate_sql(prompt, username, tokenizer, model)
    print(f"Generated SQL Query: {sql_query}")

    try:
        result = pd.read_sql_query(sql_query, engine, params=params if params else None)
        table = 'historical_timelogs' if 'historical_timelogs' in sql_query.lower() else \
                'project_forecasts' if 'project_forecasts' in sql_query.lower() else \
                'user_forecasts' if 'user_forecasts' in sql_query.lower() else None
        return table, result
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None, None

# Main function to handle user interaction
def handle_user_prompt():
    engine = create_engine(dbpath)

    # Train or load model
    train_model()
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

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
        prompt = input(f"\nEnter your request for {username} (e.g., 'Show my hours for 2022', 'List all projects I worked on with total hours', 'Which month in 2014 I worked the most') or type 'exit' to quit: ").strip()
        
        if prompt.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
        
        print(f"\nProcessing prompt: '{prompt}' for user '{username}'")
        table, result = parse_prompt(prompt, username, engine, projects, tokenizer, model)

        if result is not None and not result.empty:
            if table == 'historical_timelogs':
                if 'total_hours' in result.columns and 'projectname' in result.columns:
                    print(f"\nProjects and Total Hours for {username}:")
                elif 'total_hours' in result.columns and 'month' in result.columns:
                    print(f"\nMonthly Hours for {username}:")
                elif 'total_hours' in result.columns and 'username' in result.columns:
                    print(f"\nUsers with Total Hours:")
                elif 'projectname' in result.columns:
                    print(f"\nProjects {username} has worked on:")
                else:
                    print(f"\nHistorical Data for {username}:")
            elif table in ['user_forecasts', 'project_forecasts']:
                print(f"\nPredicted Hours for {username} in 2025:")
            print(result)
        elif result is not None:
            print(f"No data found for {username} matching your request.")
        else:
            print("Sorry, I couldn’t understand your request or generate a valid query. Please try rephrasing it.")

# Run the program
if __name__ == "__main__":
    setup_database()
    handle_user_prompt()