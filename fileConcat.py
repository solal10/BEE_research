import os
import csv

# Define the root directory path
root_dir = "CSV"

# Create a new directory to store the concatenated files
concatenated_dir = "Concatenated"
os.makedirs(concatenated_dir, exist_ok=True)

# Iterate through the directories inside the root directory
for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)

    # Create a new directory to store the concatenated file
    concat_dir = os.path.join(concatenated_dir, dir_name)
    os.makedirs(concat_dir, exist_ok=True)

    # Create a list to store the rows from all CSV files
    all_rows = []

    # Iterate through the subdirectories inside each directory
    for sub_dir_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)

        # Iterate through the bee directories
        for bee_dir_name in os.listdir(sub_dir_path):
            bee_dir_path = os.path.join(sub_dir_path, bee_dir_name)

            # Iterate through the CSV files inside each bee directory
            for csv_file_name in os.listdir(bee_dir_path):
                csv_file_path = os.path.join(bee_dir_path, csv_file_name)

                # Open the CSV file and add columns
                with open(csv_file_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)

                # Remove the header row
                rows = rows[1:]

                # Modify the rows to add the desired columns
                new_rows = []
                for i, row in enumerate(rows):
                    # Extract the type and iteration information
                    csv_type = dir_name
                    iteration = "iteration " + csv_file_name[-5]  # Add "iteration" before the number

                    # Get the hive name from the parent directory (bee directory name)
                    hive = os.path.basename(os.path.dirname(csv_file_path))

                    # Get the directory name of the CSV file
                    file_dir = os.path.basename(os.path.dirname(bee_dir_path))

                    # Create a new row with the added columns
                    new_row = row + [csv_type, iteration, hive, file_dir]
                    new_rows.append(new_row)

                # Add the modified rows to the list of all rows
                all_rows.extend(new_rows)

    # Write the concatenated rows to a new CSV file
    concat_file_path = os.path.join(concat_dir, "concatenated1.1.csv")
    with open(concat_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_rows)
