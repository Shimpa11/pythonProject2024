import pandas as pd
import requests
from bs4 import BeautifulSoup
website = "https://ackodrive.com/collection/tata-cars/"
response = requests.get(website)

soup = BeautifulSoup(response.content,'html.parser')
results = soup.find_all('div',class_='styles__Wrapper-sc-cbcba852-0 jtAUjc')

# print(results)
# print(len(results))
name=results[0].find('h2').get_text()
print(name)

variant=results[0].find('p',{'data-testid':'car_variant_fuel_type'}).get_text()
print(variant)

type=results[0].find('p',{'class':'styles__ParaWithoutMargins-sc-57d31ed8-33 jvhnwy'}).get_text()
print(type)

city=results[0].find('div',{'class':'styles__CityName-sc-57d31ed8-17 lhXIXl'}).get_text()
print(city)
ava_color = results[0].find('div',{'class':'styles__VariantInfo-sc-57d31ed8-12 fuepeG'})
available_color = ava_color.find_all('p',{'class':'styles__ParaWithoutMargins-sc-57d31ed8-33 jvhnwy'})
color=available_color[2].get_text()
print(color)

ava_color = results[0].find('div',{'class':'styles__VariantInfo-sc-57d31ed8-12 fuepeG'})
available= ava_color.find_all('p')
for p in available:
    text=p.get_text(strip=True)
    print(text)


print(text)

