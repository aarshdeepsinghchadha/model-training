import pandas as pd
import spacy
import os
import shutil
from sqlalchemy import create_engine
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import re
import warnings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
output_dir = os.path.join(root_dir, 'data', 'output', 'v5')
csv_folder = os.path.join(output_dir, 'csv_files')
model_dir = os.path.join(script_dir, 't5_text2sql_model')

# PostgreSQL connection
db_config = {
    'dbname': 'timelogs_db',
    'user': 'postgres',
    'password': 'password',  # Replace with your password
    'host': 'localhost',
    'port': '5432'
}
dbpath = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

SCHEMA_CONTEXT = "schema: historical_timelogs(username, year, month, projectname, timelog), user_forecasts(ds, yhat, yhat_lower, yhat_upper, username), project_forecasts(ds, yhat, yhat_lower, yhat_upper, projectname), projects(projectid, projectname)"

# Training data (unchanged, assuming it’s correct)
training_data = [
    # Historical Timelogs - Basic Queries
    {"prompt": "list all projects I worked on", "sql": "SELECT DISTINCT projectname FROM historical_timelogs WHERE username = %s ORDER BY projectname", "params": ["username"]},
    {"prompt": "show my hours for 2022", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = '2022' GROUP BY month ORDER BY month", "params": ["username"]},
    {"prompt": "show me the logged hours for year %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "display my hours for %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "get my hours in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "list my hours for %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "show my hours for year %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    
    # Historical Timelogs - Aggregations
    {"prompt": "total hours I worked in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "how many hours did I log in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "my total hours per project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "list projects with hours for %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY projectname", "params": ["username", "year"]},
    {"prompt": "which project had the most hours in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "which month in %s had the most hours", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    
    # Historical Timelogs - Filters
    {"prompt": "hours for project %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "show my hours on %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "hours in month %s of %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND month = %s AND year = %s", "params": ["username", "month", "year"]},
    {"prompt": "projects I worked on in %s", "sql": "SELECT DISTINCT projectname FROM historical_timelogs WHERE username = %s AND year = %s ORDER BY projectname", "params": ["username", "year"]},
    
    # Historical Timelogs - Comparisons
    {"prompt": "compare my hours in %s and %s", "sql": "SELECT year, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year IN (%s, %s) GROUP BY year ORDER BY year", "params": ["username", "year1", "year2"]},
    {"prompt": "how do my hours in %s compare to %s", "sql": "SELECT year, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year IN (%s, %s) GROUP BY year ORDER BY year", "params": ["username", "year1", "year2"]},
    {"prompt": "difference in hours between %s and %s", "sql": "SELECT year, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year IN (%s, %s) GROUP BY year ORDER BY year", "params": ["username", "year1", "year2"]},
    
    # Historical Timelogs - Extremes
    {"prompt": "which month in %s I worked the most", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "which month in %s I worked the least", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY total_hours ASC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "project with most hours in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "project with least hours in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours ASC LIMIT 1", "params": ["username", "year"]},
    
    # Historical Timelogs - Multi-Table
    {"prompt": "list projects and their IDs I worked on", "sql": "SELECT DISTINCT h.projectname, p.projectid FROM historical_timelogs h JOIN projects p ON h.projectname = p.projectname WHERE h.username = %s ORDER BY h.projectname", "params": ["username"]},
    {"prompt": "hours per project with IDs in %s", "sql": "SELECT h.projectname, p.projectid, SUM(h.timelog) AS total_hours FROM historical_timelogs h JOIN projects p ON h.projectname = p.projectname WHERE h.username = %s AND h.year = %s GROUP BY h.projectname, p.projectid ORDER BY total_hours DESC", "params": ["username", "year"]},
    
    # User Forecasts
    {"prompt": "predict my hours for 2025", "sql": "SELECT ds, yhat FROM user_forecasts WHERE username = %s ORDER BY ds", "params": ["username"]},
    {"prompt": "forecast my hours in 2025", "sql": "SELECT ds, yhat FROM user_forecasts WHERE username = %s ORDER BY ds", "params": ["username"]},
    {"prompt": "show my monthly forecast for 2025", "sql": "SELECT EXTRACT(MONTH FROM ds) AS month, SUM(yhat) AS total_hours FROM user_forecasts WHERE username = %s AND ds LIKE '2025%%' GROUP BY EXTRACT(MONTH FROM ds) ORDER BY month", "params": ["username"]},
    {"prompt": "what are my predicted hours in 2025", "sql": "SELECT SUM(yhat) AS total_hours FROM user_forecasts WHERE username = %s AND ds LIKE '2025%%'", "params": ["username"]},
    {"prompt": "monthly predicted hours for me in 2025", "sql": "SELECT EXTRACT(MONTH FROM ds) AS month, SUM(yhat) AS total_hours FROM user_forecasts WHERE username = %s AND ds LIKE '2025%%' GROUP BY EXTRACT(MONTH FROM ds) ORDER BY month", "params": ["username"]},
    
    # Project Forecasts
    {"prompt": "which project will have the most hours in 2025", "sql": "SELECT projectname, SUM(yhat) AS total_hours FROM project_forecasts WHERE ds LIKE '2025%%' GROUP BY projectname ORDER BY total_hours DESC LIMIT 1", "params": []},
    {"prompt": "forecast hours for project %s in 2025", "sql": "SELECT ds, yhat FROM project_forecasts WHERE projectname = %s AND ds LIKE '2025%%' ORDER BY ds", "params": ["projectname"]},
    {"prompt": "total predicted hours for %s in 2025", "sql": "SELECT SUM(yhat) AS total_hours FROM project_forecasts WHERE projectname = %s AND ds LIKE '2025%%'", "params": ["projectname"]},
    {"prompt": "monthly forecast for project %s in 2025", "sql": "SELECT EXTRACT(MONTH FROM ds) AS month, SUM(yhat) AS total_hours FROM project_forecasts WHERE projectname = %s AND ds LIKE '2025%%' GROUP BY EXTRACT(MONTH FROM ds) ORDER BY month", "params": ["projectname"]},
    
    # Projects Table
    {"prompt": "list all project names", "sql": "SELECT projectname FROM projects ORDER BY projectname", "params": []},
    {"prompt": "show all project IDs and names", "sql": "SELECT projectid, projectname FROM projects ORDER BY projectname", "params": []},
    {"prompt": "what are the projects available", "sql": "SELECT projectname FROM projects ORDER BY projectname", "params": []},
    
    # Cross-Table Queries
    {"prompt": "users who worked on %s", "sql": "SELECT DISTINCT username FROM historical_timelogs WHERE projectname = %s ORDER BY username", "params": ["projectname"]},
    {"prompt": "total hours for project %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE projectname = %s AND year = %s", "params": ["projectname", "year"]},
    {"prompt": "projects with more than 100 hours in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE year = %s GROUP BY projectname HAVING SUM(timelog) > 100 ORDER BY total_hours DESC", "params": ["year"]},
    {"prompt": "who worked the most hours in %s", "sql": "SELECT username, SUM(timelog) AS total_hours FROM historical_timelogs WHERE year = %s GROUP BY username ORDER BY total_hours DESC LIMIT 1", "params": ["year"]},
    {"prompt": "who worked the least in %s", "sql": "SELECT username, SUM(timelog) AS total_hours FROM historical_timelogs WHERE year = %s GROUP BY username ORDER BY total_hours ASC LIMIT 1", "params": ["year"]},
    
    # Additional Variations (to reach 100+)
    {"prompt": "my hours per month in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "hours logged by me in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "list my work hours in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "show my work on %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "hours per project for me in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "my busiest month in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "least busy month for me in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY total_hours ASC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "total hours on %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "my hours in %s for %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "projects I logged hours on in %s", "sql": "SELECT DISTINCT projectname FROM historical_timelogs WHERE username = %s AND year = %s ORDER BY projectname", "params": ["username", "year"]},
    {"prompt": "hours by month for %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "my hours on all projects in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "show my hours per project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "list my hours by project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY projectname", "params": ["username", "year"]},
    {"prompt": "hours worked on %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "my hours by month on %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "total hours I worked on %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "list my hours on %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "show my work hours in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "hours I logged in %s by month", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "my hours summary for %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "monthly hours worked by me in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "hours per project I worked on in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "my hours on projects in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "list projects I worked on with hours in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY projectname", "params": ["username", "year"]},
    {"prompt": "show my busiest project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "least busy project for me in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours ASC LIMIT 1", "params": ["username", "year"]},
    {"prompt": "my hours in %s per month", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "hours I worked per project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "show my hours worked in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "list my hours worked in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "my total hours by project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "hours on %s by month in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "my work on %s by month in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "hours logged on %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "my hours per project for %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "list my projects and hours in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY projectname", "params": ["username", "year"]},
    {"prompt": "show my hours on %s for %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "hours I worked on %s in %s by month", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "my hours in %s by project", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "list my work on %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "show my total hours in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "hours worked by me in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "my hours logged in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "list my total hours in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]},
    {"prompt": "show my hours by month in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "my hours on %s in %s per month", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "hours I worked in %s per month", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "year"]},
    {"prompt": "my hours per project in year %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY total_hours DESC", "params": ["username", "year"]},
    {"prompt": "list my hours per project in %s", "sql": "SELECT projectname, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s GROUP BY projectname ORDER BY projectname", "params": ["username", "year"]},
    {"prompt": "show my work hours on %s in %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "hours on %s for me in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "my hours worked on %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
    {"prompt": "list my hours on %s for %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "show my hours in %s on %s", "sql": "SELECT month, SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s GROUP BY month ORDER BY month", "params": ["username", "projectname", "year"]},
    {"prompt": "my total hours on %s in %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND projectname = %s AND year = %s", "params": ["username", "projectname", "year"]},
] + [{"prompt": "hours in %s for user %s", "sql": "SELECT SUM(timelog) AS total_hours FROM historical_timelogs WHERE username = %s AND year = %s", "params": ["username", "year"]}] * 50

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
    logging.info("Database setup complete with CSV data loaded.")

def get_all_usernames(engine):
    query = "SELECT DISTINCT username FROM historical_timelogs ORDER BY username;"
    return pd.read_sql_query(query, engine)['username'].tolist()

def get_all_projects(engine):
    query = "SELECT projectname FROM projects ORDER BY projectname;"
    return pd.read_sql_query(query, engine)['projectname'].tolist()

def train_model(force_retrain=False):
    if os.path.exists(model_dir) and not force_retrain:
        logging.info(f"Model already exists at {model_dir}. Skipping training.")
        return

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        logging.info(f"Removed existing model directory at {model_dir} for retraining.")

    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_data(examples):
        inputs = [f"{prompt} | {SCHEMA_CONTEXT}" for prompt in examples['prompt']]
        targets = examples['sql']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(pd.DataFrame(training_data))
    dataset = dataset.train_test_split(test_size=0.2)
    tokenized_dataset = {
        "train": dataset["train"].map(preprocess_data, batched=True),
        "validation": dataset["test"].map(preprocess_data, batched=True)
    }

    training_args = TrainingArguments(
        output_dir="./t5_text2sql",
        num_train_epochs=50,  # Reduced epochs for faster training, adjust as needed
        per_device_train_batch_size=16,  # Increased batch size
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=5e-5,  # Adjusted learning rate
        weight_decay=0.01,
        warmup_steps=500,  # Added warmup for better training stability
        fp16=True,  # Enable mixed precision for faster training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logging.info("Model training complete and saved.")

def generate_sql(prompt, username, tokenizer, model):
    year_matches = re.findall(r'\b(20\d{2})\b', prompt)
    years = year_matches if year_matches else []

    input_prompt = prompt.lower().replace("i have", "user has").replace("me", "user").replace("my", "user's")
    for i, year in enumerate(years):
        input_prompt = input_prompt.replace(year, f"%s_{i}")

    input_text = f"{input_prompt} | {SCHEMA_CONTEXT}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,  # Increased beams for better results
        early_stopping=True,
        temperature=0.7,  # Add slight randomness to avoid empty outputs
    )
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logging.info(f"Generated SQL Query: {sql_query}")

    if not sql_query or "SELECT" not in sql_query.upper():
        logging.warning("Generated SQL query is invalid or empty. Using fallback.")
        for data in training_data:
            if data["prompt"].lower() in prompt.lower():
                sql_query = data["sql"]
                params = tuple([username] + years[:len(data["params"]) - 1])
                return sql_query, params
        return None, ()

    params = []
    if "%s" in sql_query:
        if "username" in sql_query.lower():
            params.append(username)
        params.extend(years[:sql_query.count("%s") - len(params)])
    return sql_query, tuple(params)

def parse_prompt(prompt, username, engine, projects, tokenizer, model):
    sql_query, params = generate_sql(prompt, username, tokenizer, model)
    if sql_query is None:
        return None, None
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

def handle_user_prompt():
    engine = create_engine(dbpath)

    # Train only if model doesn't exist
    train_model(force_retrain=False)
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
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
        prompt = input(f"\nEnter your request for {username} (e.g., 'Show my hours for 2022') or type 'exit' to quit: ").strip()
        
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

if __name__ == "__main__":
    setup_database()
    handle_user_prompt()