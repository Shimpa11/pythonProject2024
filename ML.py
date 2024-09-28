from datetime import datetime,timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
cols=['Date','Open','High','Low','Close','Volume','Market Cap']
df=pd.DataFrame(columns=cols)
# print(df)

total_rows=30

startdate=datetime.now()
currentdate=startdate.date()
print(currentdate)

enddate= currentdate+timedelta(days=30)
print(enddate)


data = pd.date_range(currentdate, enddate)

df = pd.DataFrame(data, columns=['Date'])
data1 = np.random.uniform(1000, 100000,  size=(total_rows,6))


df2 = pd.DataFrame(data1, columns= ['Open','High','Low','Close','Volume','Market Cap'])



# print(df2)
frames=[df, df2]
df3=pd.concat(frames)
print(df3.tail(10))
print(df3.describe())

plt.plot(df3['Close'],linestyle='dotted')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')


# Display the plot
plt.show()
df3['Volume'].hist()

plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.title('Distribution of Volume')
plt.show()