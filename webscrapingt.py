import requests
from bs4 import BeautifulSoup

# URL of the page to scrape
url = 'https://ackodrive.com/collection/tata-cars/pune/'

# Send a GET request to fetch the raw HTML content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Create a list to hold the extracted car data
# cars_data = []

# Find all car elements on the page
# You might need to inspect the webpage and find the correct class names or HTML structure
car_list = soup.find_all('div', class_='styles__Wrapper-sc-cbcba852-0 jtAUjc')  # Adjust the class based on actual HTML structure
cl=car_list[0].find_all('p' ,class_="styles__ParaWithoutMargins-sc-57d31ed8-33 jvhnwy")
print(cl)
for car in car_list:
    try:
        # Extract the car year, model, kilometers, and transmission type
        name = car.find('h', class_='car-year').text.strip()  # Adjust the class based on actual HTML structure
        model = car.find('span', class_='car-model').text.strip()  # Adjust the class based on actual HTML structure
        km = car.find('span', class_='car-km').text.strip()  # Adjust the class based on actual HTML structure
        transmission = car.find('span', class_='car-transmission').text.strip()  # Adjust the class based on actual HTML structure

        # Append the data to the list
#         cars_data.append({
#             'year': year,
#             'model': model,
#             'km': km,
#             'transmission': transmission
#         })
#
#     except AttributeError:
#         # Handle missing elements or errors in extraction
#         print("An error occurred while extracting data for one of the cars.")
#
# # Print the extracted car data
# for car in cars_data:
#     print(car)
# print(cars_data)