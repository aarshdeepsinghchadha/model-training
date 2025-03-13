import pandas as pd
import numpy as np
import random
from datetime import datetime
import os
from tqdm import tqdm

# Determine the root directory (one level up from the script's location)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script's directory
root_dir = os.path.dirname(script_dir)  # One level up to C:\Project\Self\prophet
output_dir = os.path.join(root_dir, 'data', 'v4')  # C:\Project\Self\prophet\data\v4

# Create 'data/v4' folder at the root level if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Projects (15 projects)
projects = pd.DataFrame({
    'projectid': range(1, 16),
    'projectname': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta',
                    'Project Epsilon', 'Project Zeta', 'Project Eta', 'Project Theta',
                    'Project Iota', 'Project Kappa', 'Project Lambda', 'Project Mu',
                    'Project Nu', 'Project Xi', 'Project Omicron']
})

# Users (200 users)
users = [f'user{i}' for i in range(1, 201)]

# Days in each month (simplified, ignoring leap years)
days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

# Generate data for 15 years (2010-2024)
data = []
for year in tqdm(range(2010, 2025), desc="Generating years"):
    for month in range(1, 13):
        days = days_in_month[month]
        for username in users:
            # Randomly decide user behavior for the month
            behavior = random.choices(
                ['full_time', 'part_time', 'no_log'],
                weights=[0.7, 0.2, 0.1],  # 70% full time, 20% part time, 10% no log
                k=1
            )[0]

            if behavior == 'no_log':
                # User logs no hours this month
                continue

            elif behavior == 'full_time':
                # Full-time: Aim for 160 hours with 6-8 hours/day
                total_hours = 160
                num_projects = random.randint(1, 15)  # Between 1 and 15 projects
                # Calculate days needed (assuming avg 7 hours/day, adjust later)
                avg_hours_per_day = random.randint(6, 8)
                num_days_needed = total_hours // avg_hours_per_day
                num_days = min(num_days_needed + random.randint(0, 2), days)  # Some flexibility
            else:  # part_time
                # Part-time: 70-159 hours, mix of 4-hour and <4-hour days
                total_hours = random.randint(70, 159)
                num_projects = random.randint(1, 15)
                num_days = min(random.randint(10, days), days)  # Fewer days than full-time

            # Select random projects from the 15 available
            selected_project_ids = random.sample(range(1, 16), num_projects)

            # Distribute total_hours across selected projects
            hours_per_project = np.random.dirichlet(np.ones(num_projects)) * total_hours
            hours_per_project = [round(h) for h in hours_per_project]
            diff = total_hours - sum(hours_per_project)
            if diff != 0:
                hours_per_project[0] += diff

            # Assign hours to days with specific constraints
            for proj_idx, projectid in enumerate(selected_project_ids):
                hours_left = hours_per_project[proj_idx]
                days_selected = random.sample(range(1, days + 1), num_days)

                # Assign daily hours based on behavior
                daily_hours = []
                if behavior == 'full_time':
                    # Full-time: 6-8 hours per day
                    for _ in range(num_days):
                        hours = min(random.randint(6, 8), hours_left)
                        daily_hours.append(hours)
                        hours_left -= hours
                else:  # part_time
                    # Part-time: Mix of 4-hour and <4-hour days
                    for _ in range(num_days):
                        hours = min(random.choice([1, 2, 3, 4]), hours_left) if random.random() < 0.3 else min(4, hours_left)
                        daily_hours.append(hours)
                        hours_left -= hours

                # Adjust if hours remain or overshoot
                total_assigned = sum(daily_hours)
                if total_assigned < hours_per_project[proj_idx]:
                    diff = hours_per_project[proj_idx] - total_assigned
                    for i in range(min(diff, num_days)):
                        daily_hours[i] += 1
                elif total_assigned > hours_per_project[proj_idx]:
                    diff = total_assigned - hours_per_project[proj_idx]
                    for i in range(min(diff, num_days)):
                        if daily_hours[i] > 1:
                            daily_hours[i] -= 1

                # Add to data
                for day_idx, day in enumerate(days_selected):
                    if daily_hours[day_idx] > 0:  # Only log positive hours
                        date_str = f'{year}-{month:02d}-{day:02d}'
                        data.append([username, date_str, year, month, projectid,
                                    projects.loc[projectid - 1, 'projectname'], daily_hours[day_idx]])

# Create DataFrame
timelogs = pd.DataFrame(data, columns=['username', 'date', 'year', 'month',
                                       'projectid', 'projectname', 'timelog'])
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Save files to the 'data/v4' directory at the root level
timelogs.to_csv(os.path.join(output_dir, 'user_timelogs_v4.csv'), index=False)
projects.to_csv(os.path.join(output_dir, 'projects_v4.csv'), index=False)

# Verify
print("Sample of daily timelogs (first 10 entries):")
print(timelogs.head(10))
print("\nTotal hours by project:")
print(timelogs.groupby('projectname')['timelog'].sum())
print("\nTotal hours by user per month (first 10 entries):")
monthly_hours = timelogs.groupby(['username', 'year', 'month'])['timelog'].sum().reset_index()
print(monthly_hours.head(10))

# Additional verification
max_hours_check = monthly_hours['timelog'].max()
print(f"\nMaximum hours logged by any user in a single month: {max_hours_check}")
if max_hours_check > 160:
    print("Warning: Some users exceed the 160-hour monthly limit!")
else:
    print("All users are within the 160-hour monthly limit.")