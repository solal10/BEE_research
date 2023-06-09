import os
import csv

# Define the root directory path
root_dir = "CSV"

# Create a new directory to store the centralized CSV file
os.makedirs("Concatenated", exist_ok=True)

# Create a list to store the rows from all CSV files
all_rows = []

# Iterate through the directories inside the root directory
for diet_dir_name in os.listdir(root_dir):
    diet_dir_path = os.path.join(root_dir, diet_dir_name)

    # Iterate through the hive directories inside each diet directory
    for hive_dir_name in os.listdir(diet_dir_path):
        hive_dir_path = os.path.join(diet_dir_path, hive_dir_name)

        # Iterate through the bee directories inside each hive directory
        for bee_dir_name in os.listdir(hive_dir_path):
            bee_dir_path = os.path.join(hive_dir_path, bee_dir_name)

            # Iterate through the CSV files (iterations) inside each bee directory
            for csv_file_name in os.listdir(bee_dir_path):
                csv_file_path = os.path.join(bee_dir_path, csv_file_name)

                # Open the CSV file and add columns
                with open(csv_file_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)

                # Extract the diet, hive, bee number, and iteration number
                diet = diet_dir_name
                hive = hive_dir_name
                bee_number = bee_dir_name
                iteration_number = "iteration " + csv_file_name[-5]  # Add "iteration" before the number

                # Modify the rows to convert exponential values to float
                for row in rows:
                    for i, value in enumerate(row):
                        try:
                            row[i] = float(value)
                        except ValueError:
                            pass

                # Append the values from the CSV file to a single row
                row = [diet, iteration_number, bee_number, hive]
                row.extend(rows[1:])  # Skip the header row

                # Append the row to the list of all rows
                all_rows.append(row)

# Write the concatenated rows to the centralized CSV file
concat_file_path = os.path.join("Concatenated", "centralized.csv")
with open(concat_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(all_rows)

