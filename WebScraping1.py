import csv
import re
import pandas
import pandas as pd
import requests
from bs4 import BeautifulSoup

url="https://www.cars24.com/buy-used-cars-pune/"
response=requests.get(url)
# print(response)



soup=BeautifulSoup(response.content,'html.parser')
type=soup.find('div',class_="_2ujGx")

ty=type.get_text()
print(ty)
# print(len(ty))
models=[]
for t in type:
    models.append(t.text.strip())

print(models)

# name= t.find('p')
# name=t.find('h3',class_="_11dVb")
name=t.find('ul',class_="_3J2G-")
print(name.get_text())

for n in name:

    print(name.get_text())
#

data=[]

for t in type:
    list_items=t.find_all('ul',class_="_3J2G-")
    row=[item.get_text(strip=True) for item in list_items]
    data.append(row)
print(data)
#     models.append(model)
# print(models)
prices=[]
for t in type:
    price=t.find_all('div',class_="_2KyOK")
    for p in price:
        print(p.get_text())
        prices.append(p.get_text())
print(prices)





year=[]
for t in type:
    years=t.find_all('h3',class_="_11dVb")
    for y in years:
        print(y.get_text())
        year.append(y.get_text(strip=True))
print(year)


# Define a regular expression pattern to extract the year
year_pattern = r'\b\d{4}\b'

# Initialize the list to store years
year_list = []

# Loop through each text entry
for yN in year:
    match = re.search(year_pattern, yN)
    if match:
        yearss = match.group(0)  # Extract the matched year
        year_list.append(yearss)
print(year_list)





total_data=[year_list,models,data]
cols=['Year', 'Model', 'Kilometers',  'Fuel','Transmission']
df=pd.DataFrame(total_data)
print(df)
# CSV file name
csv_file = 'cars24_pune_data.csv'
fieldnames=['Year', 'Model', 'Kilometers',  'Fuel','Transmission']
with open(csv_file,mode='w', newline='', encoding='utf-8') as file:
   writer=csv.writer(file)
   writer.writerow(fieldnames)
   writer.writerows(total_data)
print(f"Data successfully written to {csv_file}")

# df.to_csv('carsPune.csv' ,index=False)


# Read the CSV file into a DataFrame
# df = pd.read_csv('cars24_pune_data.csv',delimiter=';' )
#
# # Transpose the DataFrame
# df_transposed = df.T
#
# # Optionally, set the first row as the header
# df_transposed.columns = df_transposed.iloc[0]
# df_transposed = df_transposed[1:]
#
# # Reset index for a clean CSV output (optional)
# df_transposed.reset_index(drop=True, inplace=True)
#
# # Save the transposed DataFrame to a new CSV file
# df_transposed.to_csv('transposed_data.csv', index=False)
#
# print("Data successfully transposed and saved to 'transposed_data.csv'")
#
#
