import pandas as pd
import os
import re

csv_directory = '../03_input/input_data_2022'

# Create an empty list to hold dataframes
running_data = []

# Loop through all CSV files in the directory
for file_name in os.listdir(csv_directory):
    if file_name.endswith(".csv"):
        # Extract the date from the file name
        # Assuming the date format in the file name is YYYY-MM-DD
        date_match = re.search(r"(keep_\d{2}_\d{2})", file_name)
        if date_match:
            date_str = date_match.group(1)
            # print(date_str.split('_'))
            date = pd.to_datetime('2022_'+date_str.split('_')[1]+'_'+date_str.split('_')[2], format='%Y_%m_%d')

            # Read the CSV file
            file_path = os.path.join(csv_directory, file_name)
            df = pd.read_csv(file_path)

            # Add a column for the date
            df["date"] = date
            df['grid_id'] = df.index
            # Add this dataframe to the list
            running_data.append(df)

# Concatenate all dataframes into a single dataframe
combined_running_data = pd.concat(running_data, ignore_index=True)

print(combined_running_data.shape)
combined_running_data.head()
combined_running_data.to_csv('../03_input/combined_running_data.csv', index=False)