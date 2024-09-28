import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app_df = pd.read_csv(r"C:\Users\ershi\Downloads\application_data.csv")
# print(app_df)


desc=app_df.describe()
# print(desc)
desc1=app_df.info()
# print(desc1)
cols=pd.DataFrame(app_df.isnull().mean().round(4) *100,columns=['percentage_missing_values']).sort_values(by=['percentage_missing_values'])
print(cols)
print(str(round(100.0* cols[cols['percentage_missing_values']==0].count(0))))

# checking row-wise null percentages
row_null= pd.DataFrame(app_df.isnull().sum(axis=1), columns=['num_missing_value'])
print(row_null)

#Droping column having more than 50 % null values
app_df_1 = app_df.drop(app_df.columns[ app_df.apply(lambda col: (col.isnull().sum()/len(app_df)*100) > 50)], axis=1)
print (app_df_1.columns)
# Checking dimensions of dataframe after dropping columns
app_df_1.shape

#re-checking columns with missing values
round(100.0* app_df_1.isnull().sum()/len(app_df_1), 2).sort_values()


# dividing by 1000 for the ease of read and converting value in ('000s')
sns.boxplot(app_df_1['AMT_GOODS_PRICE']/1000.0)
plt.show()

Gcount=app_df['CODE_GENDER'].value_counts()
print(Gcount)


# Check the Age Summary

(app_df['DAYS_BIRTH'] // 365).describe()
# Binning DAYS_BIRTH based on above summary

bins = [0,20,30,40,50,60,100]
labels = ['Below 20','20-30','30-40','40-50','50-60','Above 60']
app_df['AGE_GROUP'] = pd.cut(app_df['DAYS_BIRTH'] // 365, bins = bins, labels = labels )
# Checking the values

app_df['AGE_GROUP'].value_counts().plot(kind='bar')
plt.title("No. of Loan Applicants Vs Age Group\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel('No. of applicants', fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Grey'})
plt.xlabel('Age Group', fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Grey'})
plt.xticks(rotation=30)
plt.show()


# Checking imbalance percentage

app_df['TARGET'].value_counts(normalize = True)*100

#Extracting the imbalance percentage
Repayment_Status = app_df['TARGET'].value_counts(normalize=True)*100

# Defining the x values
x= ['Others','Defaulters']

# Defining the y ticks
axes= plt.axes()
axes.set_ylim([0,100])
axes.set_yticks([10,20,30,40,50,60,70,80,90,100])

# Plotting barplot
sns.barplot(x, Repayment_Status)

# Adding plot title, and x & y labels
plt.title('Imbalance Percentage\n', fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Brown'})
plt.xlabel("Borrower Category")
plt.ylabel("Percentage")

# Displaying the plot
plt.show()
