import requests
import pandas as pd
from io import StringIO

# Replace 'your_google_sheet_url' with the actual public Google Sheet URL.
# sheet_url = 'your_google_sheet_url'

# Construct the Google Sheets API endpoint URL.
# api_url = f'https://docs.google.com/spreadsheets/d/{sheet_url.split("/")[-2]}/gviz/tq?tqx=out:csv'

test_url = 'https://docs.google.com/spreadsheets/d/1SEydb9-Dm1LqP2ZhqKbIuZrvAYonZiekz5C9cAbBm9U/edit?usp=sharing'

# Send a GET request to fetch the data.
response = requests.get(test_url)

# Check if the request was successful.
if response.status_code == 200:
    # Read the CSV data using pandas
    csv_data = StringIO(response.text)

    data = pd.DataFrame(csv_data)
    
    # Replace 'output_file.xlsx' with your desired Excel file name
    output_excel = 'output_file.xlsx'
    
    # Convert and save the data to an Excel file
    data.to_excel(output_excel, index=False)
    
    print(f"Data saved to {output_excel}")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
