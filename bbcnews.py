import csv

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

url='https://www.bbc.com/news/world'
response=requests.get(url)
# print(response)


soup=BeautifulSoup(response.content,'html.parser')
# print(soup.prettify())
results=soup.find_all('div',class_='sc-b8778340-0 kFuHJG')

print(len(results))

# headlines=[]
# for h in results:
#     if results:
#         headlines.append(h.text)
#     else:
#         headlines.append(None)
# print(headlines)
#
#
# categories=[]
# category=soup.find_all('span',class_='sc-4e537b1-2 eRsxHt')
# print(len(category))
# for c in category:
#     if category:
#      categories.append(c.text)
#     else:
#         categories.append(None)
# print(categories)
#
#
# date=soup.find_all('span',class_='sc-4e537b1-1 dsUUMv')
# print(len(date))
#
# dates=[]
# for d in date:
#     if date:
#      dates.append(d.text)
#     else:
#         dates.append(None)
# print(dates)
#
# # df=pd.DataFrame({
# #     'Headline': headlines,
# #     'Category': categories,
# #     'Date': dates
# # })
# # print(df)

headlines=[]
category=[]
date=[]
for r in results:
    headlines.append(r.find('h2').text)
    category.append(r.find('span',class_='sc-4e537b1-1 dsUUMv'))
    for c in category:
        category.append(c)
    date.append(r.find('span',class_='sc-4e537b1-2 eRsxHt'))

print(len(headlines))



# results2=soup.find('div',class_='sc-4e537b1-0 gtLVrL').text


# for r2 in results2:
#     if category:
#      category.append(r2.find('span',class_='sc-4e537b1-1 dsUUMv', datatestid_='card-metadata-tag'))
#     else:
#         category.append(None)
    # date.append(r2.find('span',class_='sc-4e537b1-2 eRsxHt').text)
print(len(category))
print(len(date))


print(headlines)
print(category)
print(date)




# file = 'bbcNews_data.csv'
#
# # Write data to CSV file
# with open(file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Headline', 'Category', 'Dates'])  # Write the header
#     for h, c, d in zip(headlines, categories, dates):
#         writer.writerow([h, c, d])  # Write each row of data
#
# print(f"Data saved to {file}")