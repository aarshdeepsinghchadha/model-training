import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# Set matplotlib style
plt.style.use('seaborn-v0_8')

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Determine directories
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
output_dir = os.path.join(root_dir, 'data', 'output', 'v4')  # Updated to v4
csv_folder = os.path.join(output_dir, 'csv_files')
image_folder = os.path.join(output_dir, 'images')
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# Load data from v4 folder
timelogs = pd.read_csv(os.path.join(root_dir, 'data', 'v4', 'user_timelogs_v4.csv'))  # Updated to v4
timelogs['date'] = pd.to_datetime(timelogs['date'])

print("Timelogs sample (all years):")
print(timelogs.head())

# Aggregate by month
monthly_user = timelogs.groupby(['username', 'year', 'month'])['timelog'].sum().reset_index()
monthly_project = timelogs.groupby(['projectname', 'year', 'month'])['timelog'].sum().reset_index()

monthly_user['month_str'] = monthly_user['month'].apply(lambda x: months[x-1])
monthly_user['month'] = pd.Categorical(monthly_user['month_str'], categories=months, ordered=True)
monthly_project['month_str'] = monthly_project['month'].apply(lambda x: months[x-1])
monthly_project['month'] = pd.Categorical(monthly_project['month_str'], categories=months, ordered=True)

# Define LSTM Model
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Function to prepare data for LSTM
def prepare_lstm_data(df, group_col, value_col, seq_length=12):
    scaler = MinMaxScaler()
    data_dict = {}
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group].sort_values(['year', 'month'])
        if len(group_df) < seq_length + 1:  # Need enough data for sequence + target
            print(f"Skipping {group} due to insufficient data points ({len(group_df)})")
            continue
        values = group_df[value_col].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(values)
        data_dict[group] = (scaled_values, scaler)
    return data_dict

# Function to create sequences (optimized)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    X = np.stack(X)  # Stack into a single NumPy array
    y = np.stack(y)
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Function to train and predict with LSTM
def forecast_2025_lstm(df, group_col, value_col, seq_length=12, periods=12):
    data_dict = prepare_lstm_data(df, group_col, value_col, seq_length)
    forecasts = {}
    
    for group in tqdm(data_dict.keys(), desc=f"Forecasting {group_col} with LSTM"):
        scaled_values, scaler = data_dict[group]
        X, y = create_sequences(scaled_values, seq_length)
        batch_size = X.shape[0]  # Dynamic batch size based on number of sequences
        
        # Define model
        model = LSTMForecast(input_size=1, hidden_size=50, num_layers=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        epochs = 50
        for _ in range(epochs):
            hidden = model.init_hidden(batch_size)  # Match batch size to input
            optimizer.zero_grad()
            outputs, hidden = model(X, hidden)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Forecast for 2025
        last_seq = torch.FloatTensor(scaled_values[-seq_length:]).unsqueeze(0)  # Shape: [1, seq_length, 1]
        predictions = []
        hidden = model.init_hidden(1)  # Batch size of 1 for forecasting
        with torch.no_grad():
            current_seq = last_seq.clone()
            for _ in range(periods):
                pred, hidden = model(current_seq, hidden)
                predictions.append(pred.item())
                pred_reshaped = pred.view(1, 1, 1)  # From [1, 1] to [1, 1, 1]
                current_seq = torch.cat((current_seq[:, 1:, :], pred_reshaped), dim=1)
        
        # Inverse transform predictions
        pred_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start='2025-01-01', periods=periods, freq='MS')
        forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': pred_values, group_col: group})
        forecasts[group] = forecast_df
    
    if not forecasts:
        raise ValueError(f"No forecasts generated for {group_col}; check input data")
    return pd.concat(forecasts.values(), ignore_index=True)

# Prepare data
user_data = monthly_user[['username', 'year', 'month', 'timelog']]
project_data = monthly_project[['projectname', 'year', 'month', 'timelog']]

# Forecast
user_forecasts = forecast_2025_lstm(user_data, 'username', 'timelog')
project_forecasts = forecast_2025_lstm(project_data, 'projectname', 'timelog')

# Save forecasts with new v4 file names
user_forecasts.to_csv(os.path.join(csv_folder, 'user_forecasts_2025_lstm_v4.csv'), index=False)
project_forecasts.to_csv(os.path.join(csv_folder, 'project_forecasts_2025_lstm_v4.csv'), index=False)

# --- Historical and Predicted Analysis ---
top_users_historical = monthly_user.loc[monthly_user.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
top_users_historical = top_users_historical.sort_values(['year', 'month'])
top_users_historical.to_csv(os.path.join(csv_folder, 'historical_top_users_2010_2024_lstm_v4.csv'), index=False)

user_forecasts['month'] = user_forecasts['ds'].dt.strftime('%b')
user_forecasts['month'] = pd.Categorical(user_forecasts['month'], categories=months, ordered=True)
user_forecasts['year'] = user_forecasts['ds'].dt.year
top_users_2025_pred = user_forecasts.loc[user_forecasts.groupby('month', observed=True)['yhat'].idxmax()]
top_users_2025_pred.to_csv(os.path.join(csv_folder, 'predicted_top_users_2025_lstm_v4.csv'), index=False)

# --- Graphing Section ---
# Historical Top Users
plt.figure(figsize=(20, 10))
for year in range(2010, 2025):
    year_data = top_users_historical[top_users_historical['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o', markersize=4)
plt.title('Historical Top User Hours by Month (2010-2024)', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Hours Logged by Top User', fontsize=12)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'historical_top_users_2010_2024_lstm_v4.png'), bbox_inches='tight')
plt.close()

# Historical Total Project Hours
project_totals = monthly_project.groupby(['year', 'month'], observed=True)['timelog'].sum().reset_index()
plt.figure(figsize=(20, 10))
for year in range(2010, 2025):
    year_data = project_totals[project_totals['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o', markersize=4)
plt.title('Historical Total Project Hours by Month (2010-2024)', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Hours Logged', fontsize=12)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'historical_project_hours_2010_2024_lstm_v4.png'), bbox_inches='tight')
plt.close()

# Predicted Top Users 2025
plt.figure(figsize=(12, 6))
plt.plot(top_users_2025_pred['month'], top_users_2025_pred['yhat'],
         label='Predicted 2025 (LSTM)', marker='o', linestyle='--', color='purple')
plt.title('Predicted Top User Hours by Month (2025) - LSTM')
plt.xlabel('Month')
plt.ylabel('Predicted Hours by Top User')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'predicted_top_users_2025_lstm_v4.png'))
plt.close()

# Predicted Total Project Hours 2025
project_forecasts['month'] = project_forecasts['ds'].dt.strftime('%b')
project_forecasts['month'] = pd.Categorical(project_forecasts['month'], categories=months, ordered=True)
project_forecasts_totals = project_forecasts.groupby('month', observed=True)['yhat'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(project_forecasts_totals['month'], project_forecasts_totals['yhat'],
         label='Predicted 2025 (LSTM)', marker='o', linestyle='--', color='purple')
plt.title('Predicted Total Project Hours by Month (2025) - LSTM')
plt.xlabel('Month')
plt.ylabel('Predicted Total Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'predicted_project_hours_2025_lstm_v4.png'))
plt.close()

# --- Output Summary ---
print(f"\nCSVs saved in '{csv_folder}':")
print("  - user_forecasts_2025_lstm_v4.csv")
print("  - project_forecasts_2025_lstm_v4.csv")
print("  - historical_top_users_2010_2024_lstm_v4.csv")
print("  - predicted_top_users_2025_lstm_v4.csv")
print(f"Images saved in '{image_folder}':")
print("  - historical_top_users_2010_2024_lstm_v4.png")
print("  - historical_project_hours_2010_2024_lstm_v4.png")
print("  - predicted_top_users_2025_lstm_v4.png")
print("  - predicted_project_hours_2025_lstm_v4.png")