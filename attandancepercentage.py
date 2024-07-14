import csv
import os
import pandas as pd
from datetime import datetime

# Function to read attendance data from CSV file
def read_attendance_data(file_path):
    attendance_data = []
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return attendance_data

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if exists
        for row in reader:
            attendance_data.append(row)
    return attendance_data

# Function to calculate attendance percentage
def calculate_attendance_percentage(attendance_data):
    # Assuming attendance_data is a list of lists with format [name, timestamp]
    if not attendance_data:
        return {}

    attendance_dict = {}
    for row in attendance_data:
        name = row[0]
        if name in attendance_dict:
            attendance_dict[name].append(datetime.strptime(row[1], '%H:%M:%S'))
        else:
            attendance_dict[name] = [datetime.strptime(row[1], '%H:%M:%S')]

    total_days = len(attendance_dict.keys())
    attendance_percentage = {}
    for name, timestamps in attendance_dict.items():
        total_attendance = len(timestamps)
        percentage = (total_attendance / total_days) * 100
        attendance_percentage[name] = round(percentage, 2)

    return attendance_percentage

# Function to generate attendance report in PDF or Excel format
def generate_attendance_report(attendance_percentage, output_file):
    df = pd.DataFrame(list(attendance_percentage.items()), columns=['Name', 'Attendance Percentage'])
    # You can also add more columns to df, like late arrivals, etc., if needed

    # Export to PDF (requires installing pandas and openpyxl if not already installed)
    df.to_excel(output_file + ".xlsx", index=False)

# Example usage
if __name__ == "__main__":
    # Ensure the directory exists
    directory = 'C:/Users/nithi/facereg/attendance'  # Adjust directory path here
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the specific attendance file exists
    date = datetime.now().strftime('%d-%m-%y')
    attendance_file = f'{directory}/attendance_{date}.csv'  # Adjust file name format here

    output_file = 'attendance_report'  # Output file name

    attendance_data = read_attendance_data(attendance_file)
    if attendance_data:
        attendance_percentage = calculate_attendance_percentage(attendance_data)
        print("Attendance Percentage:")
        for name, percentage in attendance_percentage.items():
            print(f"{name}: {percentage}%")

        generate_attendance_report(attendance_percentage, output_file)
    else:
        print("No attendance data found.")
