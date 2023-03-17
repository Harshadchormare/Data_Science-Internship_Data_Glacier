#!/usr/bin/env python
# coding: utf-8

# #Harshad Chormare_data_glacier_virtual_internship_LISUM19

# # A. Import Libraries and Dataset

# In[1]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# # Cab Data set

# In[7]:


cab_df = pd.read_csv("C://Users//Niharika//Desktop//Cab_Data.csv")
cab_df.head(5)


# In[3]:


cab_df.info()


# # Data Type

# In[4]:


cab_df.dtypes


# we need to change the type of Data of Travel Cols

# In[8]:


# change the type of Date of Travel datatype into DateTime 
from datetime import datetime, date
a = cab_df['Date of Travel'].to_list()
base_date = pd.Timestamp('1899-12-29')
dates = [base_date + pd.DateOffset(date_offset) for date_offset in a]
cab_df['Date of Travel'] = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S')


# In[9]:


cab_df.describe()


# In[11]:


#find unique values in columns  


# In[12]:


cab_df['Company'].unique()


# In[13]:


cab_df['City'].unique()


# # City Data set 

# In[14]:


city_df = pd.read_csv("C://Users//Niharika///Desktop//City.csv")
city_df.head(5)


# In[15]:


city_df.info()


# In[16]:


#We need to change the type of Populations and Users Cols in City DF


# In[17]:


# Attribute 'Population' should be an integer
city_df['Population'] = [x.replace(',','') for x in city_df['Population']]
city_df['Population'] = city_df['Population'].astype(float)

# Attribute 'Users' should be an integer
city_df['Users'] = [x.replace(',','') for x in city_df['Users']]
city_df['Users'] = city_df['Users'].astype(float)


# In[18]:


# Now check the type
city_df.dtypes


# In[19]:


city_df.describe()


# # Transaction_id Dataset

# In[23]:


transaction_id_df = pd.read_csv("C://Users//Niharika///Desktop//Transaction_ID.csv")
transaction_id_df.head(5)


# In[24]:


transaction_id_df.info()


# In[25]:


transaction_id_df.describe(include = 'all', datetime_is_numeric=True)


# # Customer data

# In[26]:


customer_id_df = pd.read_csv("C://Users//Niharika//Desktop//Customer_ID.csv")
customer_id_df.head(5)


# In[27]:


customer_id_df.info()


# In[28]:


customer_id_df.describe( include = 'all')


# # Merge and Visualize the Whole Dataset

# In[ ]:


#Merge the Whole Dataset


# In[29]:


df= cab_df.merge(transaction_id_df, on= 'Transaction ID').merge(customer_id_df, on ='Customer ID').merge(city_df, on = 'City')
df.head(5)


# # B.Visualization of the data

# In[30]:


sns.pairplot(df.head(1000), hue = 'Company')


# # Check the correlation

# In[31]:


data_corr = df.corr()
data_corr


# In[32]:


# Define the figure size
plt.figure(figsize = (16, 9))

# Cutomize the annot
annot_kws={'fontsize':10,                      # To change the size of the font
           'fontstyle':'italic',               # To change the style of font 
           'fontfamily': 'serif',              # To change the family of font 
           'alpha':1 }                         # To change the transparency of the text  


# Customize the cbar
cbar_kws = {"shrink":1,                        # To change the size of the color bar
            'extend':'min',                    # To change the end of the color bar like pointed
            'extendfrac':0.1,                  # To adjust the extension of the color bar
            "drawedges":True,                  # To draw lines (edges) on the color bar
           }

# take upper correlation matrix
matrix = np.triu(data_corr)

# Generate heatmap correlation
ax = sns.heatmap(data_corr, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, cbar_kws=cbar_kws)

# Set the title etc
plt.title('Correlation Heatmap of "G2M Insight for Cab Investment", fontsize = 20')

# Set the size of text
sns.set(font_scale = 1.2)


# > <b> As we can see there is strong Correlation between </b>  <br> ◉ Population vs Users <br> ◉ Price Charged vs Cost of Trip vs KM Travelled

# # C.Investigate the Data

# In[ ]:


#User Travel


# In[33]:


user=df.groupby('Company')
avg_user = user.Users.mean()
index = avg_user.index
value = avg_user.values 


# In[34]:


figp, axp = plt.subplots(figsize=(10,7))
axp.pie(value , labels=index, autopct='%1.1f%%',shadow=True, startangle=90,)
axp.axis('equal')

plt.title('Users Travel', fontsize = 15)
plt.show()


# > <b> As we can see users like to ride on `Yellow cab` more as compared to Pink Cab </b>

# In[ ]:


#Price Charged 


# In[36]:


sns.set(style = 'darkgrid')

plt.figure(figsize = (16, 9))

sns.boxplot(df['Company'], df['Price Charged'])
plt.title('Price Charged of Both Companies', fontsize=20)
plt.show()


# > <b> As we can see Price Charged of `Yellow Cab` is highest as compared to Pink Cab </b>

# In[ ]:


#KM Travelled Distribution


# In[38]:


plt.figure(figsize = (16, 9))
plt.hist(df['KM Travelled'], bins = 40)
plt.title('Km Travelled Distribution', fontsize=20)
plt.ylabel('Frequency')
plt.xlabel('Km Travelled')
plt.show()


# > <b> Most of the rides varies from `2 to 48` KM. </b>

# In[ ]:


#Payment Mode


# In[39]:


plt.figure(figsize = (16, 9))
ax = sns.countplot(x="Company", hue="Payment_Mode", data=df)
plt.title('Payment Mode in both companies', fontsize=25)
plt.show()


# > <b> As we can see that users prefer to pay with a `card` more as compared to cash

# In[ ]:


# Gender


# In[40]:


gender_cab=df.groupby(['Company','Gender'])
gender_cab  = gender_cab['Customer ID'].nunique()
print(gender_cab)


# In[41]:


labs = gender_cab.index
vals = gender_cab.values
figp, axp = plt.subplots(figsize=(10,7))
axp.pie(vals , labels=labs, autopct='%1.1f%%',shadow=True, startangle=90,)
axp.axis('equal')

plt.title('Customer share per gender per cab', fontsize = 15)
plt.show()


# >◉ <b>`Male` users are prefer more to travel in Cab<br></b>◉ <b>Users prefer to travel in `Yellow Cab` </b>

# In[42]:


#Users respected Cities


# In[43]:


city_users = df.groupby('City')
city_users = city_users.Users.count()
labs = city_users.index
vals = city_users.values

plt.style.use('fivethirtyeight')
figp, axp = plt.subplots(figsize=(18,13))
axp.pie(vals , labels=labs, autopct='%1.1f%%',shadow=True, startangle=90,)
axp.axis('equal')
plt.title('Users per City')
plt.show()


# > <b> `New York City` has the highest Cab users with 28% followed by `Chicago` with 16% and `Los Angeles` with 13%

# In[ ]:


#Profit Margin 


# In[44]:


company = df.groupby('Company')
price_charged = company['Price Charged'].mean()
cost_trip = company['Cost of Trip'].mean()
c = cost_trip.index
c_v = cost_trip.values
c_p = price_charged.values


# In[45]:


plt.style.use('fivethirtyeight')
plt.figure(figsize = (16, 9))
plt.bar(c, c_p, edgecolor='black', label="Revenue")
plt.bar(c, c_v, edgecolor='black', label="Profit")
plt.title('Profit Margin')
plt.ylabel('Price Charged')
plt.xlabel('Cost of Trip')
plt.legend()
plt.show()


# > <b> The `Yellow cab` has a higher Profit Margin (Price Charged - Cost of Trip) compared to Pink cab

# In[46]:


df['Year'] = df['Date of Travel'].dt.year
df['Month'] = df['Date of Travel'].dt.month
df['Day'] = df['Date of Travel'].dt.day
df['Profit'] = df['Price Charged'] - df['Cost of Trip']


# In[47]:


plt.figure(figsize = (16, 9))
sns.lineplot(x='Year', y='Profit', hue="Company", data=df, marker='o')
plt.xlabel("Year", size=14)
plt.ylabel("Profit %", size=14)
plt.title("Profit % per year")
plt.show()


# > <b> The profit margin `decrease` w.r.t year

# In[48]:


plt.figure(figsize = (16, 9))
sns.lineplot(x='Month', y='Profit', hue="Company", data=df, marker='o')
plt.xlabel("Month", size=14)
plt.ylabel("Profit %", size=14)
plt.title("Profit % per month")
plt.show()


# > <b> The profit margin `varies` w.r.t month

# In[ ]:


# Users Respective Population


# In[49]:


urp = (city_df['Users'] /city_df['Population']) * 100 
city = city_df['City']


# In[50]:


# Get the list of color
from random import randint

colors = []
n = 16

for i in range(n):
    colors.append('#%06X' % randint(0, 0xFFFFFF))


# In[51]:


plt.figure(figsize = (16, 9))
plt.bar(city, urp, edgecolor='black', color = colors)
plt.gcf().autofmt_xdate()
plt.title('Users Respective Population')
plt.ylabel('Percentage (%)')
plt.xlabel('Cities')
plt.show()


# > <b> As we can see  in cities `San Francisco`, `Washington` and `Boston` more than 30% of population use cab service </b>

# #Average Age of Users

# In[54]:


sns.set(style = 'darkgrid') 

plt.figure(figsize = (16, 9))

sns.violinplot(df['Gender'], df['Age'], hue = df['Company'], palette = 'husl', inner = 'quartiles')
plt.title('Avg age of users', fontsize=20)
plt.show()


# > <b>As we can see `35 Avg age` of Female and Male who use Cab service </b>

# In[ ]:


#Average Income


# In[55]:


sns.set(style = 'darkgrid')

plt.figure(figsize = (16, 9))

sns.boxplot(df['Company'], df['Income (USD/Month)'])
plt.title('User Income', fontsize=20)
plt.show()


# > <b> As we can see Avg income is around `15k$` who use cab sevice 

# In[56]:


#Price Charged w.r.t Distance


# In[57]:


plt.figure(figsize = (16, 9))

sns.scatterplot(data=df, x="KM Travelled", y='Price Charged', hue='Company')
plt.title('Price Charged w.r.t Distance',fontsize = 20)
plt.ylabel('Price Charged',fontsize = 14)
plt.xlabel('KM Travelled',fontsize = 14)
plt.show()


# > <b> As we can see there is a `linear relationship` between KM traveled and Price Charged as we expected. However, `Yellow Cab` has high charges compared to Pink.

# # D. Create Multiple Hypothesis and Investigate

# # Hypothesis 1: Is there any difference in profit regarding Gender
# > <b>H0 :</b> There is no difference regarding Gender in both cab companies. <br><b>H1 :</b> There is difference regarding Gender in both cab companies.

# # Pink Cab 

# In[58]:


a = df[(df.Gender=='Male')&(df.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
b = df[(df.Gender=='Female')&(df.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
print(a.shape[0],b.shape[0])

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference regarding gender for Pink Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference regarding gender for Pink Cab')


# # Yellow Cab

# In[59]:


a = df[(df.Gender=='Male')&(df.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
b = df[(df.Gender=='Female')&(df.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
print(a.shape[0],b.shape[0])

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference regarding gender for Yellow Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference regarding gender for Yellow Cab')


# > <b> There is no difference regarding Gender in both cab companies.</b>

# # Hypothesis 2: Is there any difference in Profit regarding Age
# > <b>H0 :</b> There is no difference regarding Age in both cab companies. <br><b>H1 :</b> There is difference regarding Age in both cab companies.

# # Pink Cab 

# In[61]:


a = df[(df.Age <= 60)&(df.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
b = df[(df.Age >= 60)&(df.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
print(a.shape[0],b.shape[0])

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference regarding age for Pink Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference regarding age for Pink Cab')


# # Yellow Cab 

# In[62]:


a = df[(df.Age <= 60)&(df.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
b = df[(df.Age >= 60)&(df.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
print(a.shape[0],b.shape[0])

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference regarding age for Yellow Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference regarding age for Yellow Cab')


# > <b> Yellow Cab company offers discounts for their customers who are older than 60 years old.</b>

# # Hypothesis 3: Is there any difference in Profit regarding Payment mode
# > <b>H0 :</b> There is no difference regarding Payment_Mode in both cab companies. <br><b>H1 :</b> There is difference regarding Payment_Mode in both cab companies..

# # Pink Cab

# In[63]:


a = df[(df['Payment_Mode']=='Cash')&(df.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
b = df[(df['Payment_Mode']=='Card')&(df.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference in payment mode for Pink Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference in payment mode for Pink Cab')


# # Yellow Cab 

# In[65]:


a = df[(df['Payment_Mode']=='Cash')&(df.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
b = df[(df['Payment_Mode']=='Card')&(df.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference in payment mode for Yellow Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference in payment mode for Yellow Cab')


# > <b>There is no difference in payment mode for both cab companies. </b>

# In[ ]:




