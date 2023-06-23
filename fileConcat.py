import os
import csv

# Define the root directory path
root_dir = "CSV"

# Create a new directory to store the centralized CSV file
os.makedirs("Concatenated", exist_ok=True)

# Maximum cell size limit
max_cell_size = 32767

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

                # Join the values within square brackets while preserving the brackets
                joined_rows = []
                for row in rows[1:]:
                    joined_row = "[" + ",".join(map(str, row)) + "]"
                    joined_rows.append(joined_row)

                # Split the joined rows into multiple cells if it exceeds the maximum cell size
                num_cells = len(joined_rows)
                if num_cells > 1:
                    split_rows = []
                    current_cell_size = 0
                    current_row = ""
                    for row in joined_rows:
                        row_size = len(row)
                        if current_cell_size + row_size <= max_cell_size:
                            current_row += row
                            current_cell_size += row_size
                        else:
                            split_rows.append(current_row)
                            current_row = row
                            current_cell_size = row_size
                    split_rows.append(current_row)

                    # Append the values from the CSV file to the row
                    row = [diet, iteration_number, bee_number, hive] + split_rows
                else:
                    # Append the values from the CSV file to the row
                    row = [diet, iteration_number, bee_number, hive, joined_rows[0]]

                # Write the row to the centralized CSV file
                concat_file_path = os.path.join("Concatenated", "centralized.csv")
                with open(concat_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
