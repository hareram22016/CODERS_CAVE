#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('C:/Users/Admin/Downloads/Global Terrorism - START data/terrorism.csv', encoding='latin1')


# In[3]:


data.head()


# In[4]:


data.columns.values


# In[5]:


data.rename(columns={'iyear':'Year','imonth':'Month','iday':"day",'gname':'Group','country_txt':'Country','region_txt':'Region','provstate':'State','city':'City','latitude':'latitude',
    'longitude':'longitude','summary':'summary','attacktype1_txt':'Attacktype','targtype1_txt':'Targettype','weaptype1_txt':'Weapon','nkill':'kill',
     'nwound':'Wound'},inplace=True)


# In[6]:


data = data[['Year','Month','day','Country','State','Region','City','latitude','longitude',"Attacktype",'kill',
               'Wound','target1','summary','Group','Targettype','Weapon','motive']]


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.isnull().sum()


# In[10]:


data['Wound'] = data['Wound'].fillna(0)
data['kill'] = data['kill'].fillna(0)


# In[11]:


data['Casualities'] = data['kill'] + data['Wound']


# In[12]:


data.info()


# In[13]:


data.describe()


# In[17]:


year = data['Year'].unique()
years_count = data['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (20,10))
sns.barplot(x = year,
           y = years_count,
           palette = "tab10")
plt.xticks(rotation = 55)
plt.xlabel('Attacking Year',fontsize=22)
plt.ylabel('Number of Attacks Each Year',fontsize=22)
plt.title('Attacks In Years',fontsize=32)
plt.show()


# In[18]:


pd.crosstab(data.Year, data.Region).plot(kind='area',stacked=False,figsize=(20,10))
plt.title('Terrorist Activities By Region In Each Year',fontsize=25)
plt.ylabel('Number of Attacks',fontsize=20)
plt.xlabel("Year",fontsize=20)
plt.show()


# In[19]:


attack = data.Country.value_counts()[:10]
attack


# In[20]:


data.Group.value_counts()[1:10]


# In[24]:


plt.subplots(figsize=(20,10))
sns.barplot(x=data['Country'].value_counts()[:10].index, y=data['Country'].value_counts()[:10].values,palette='YlGnBu')
plt.title('Top Countries Affected', fontsize=25)
plt.xlabel('Countries', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(rotation = 50)
plt.show()


# In[25]:


df = data[['Year','kill']].groupby(['Year']).sum()
fig, ax4 = plt.subplots(figsize=(20,10))
df.plot(kind='bar',alpha=0.7,ax=ax4)
plt.xticks(rotation = 50)
plt.title("People Died Due To Attack",fontsize=25)
plt.ylabel("Number of killed peope",fontsize=20)
plt.xlabel('Year',fontsize=20)
top_side = ax4.spines["top"]
top_side.set_visible(False)
right_side = ax4.spines["right"]
right_side.set_visible(False)


# In[26]:


data['City'].value_counts().to_frame().sort_values('City',axis=0,ascending=False).head(10).plot(kind='bar',figsize=(20,10),color='blue')
plt.xticks(rotation = 50)
plt.xlabel("City",fontsize=20)
plt.ylabel("Number of attack",fontsize=20)
plt.title("Top 10 most effected city",fontsize=25)
plt.show()


# In[27]:


data['Attacktype'].value_counts().plot(kind='bar',figsize=(20,10),color='magenta')
plt.xticks(rotation = 50)
plt.xlabel("Attacktype",fontsize=20)
plt.ylabel("Number of attack",fontsize=20)
plt.title("Name of attacktype",fontsize=25)
plt.show()


# In[28]:


data[['Attacktype','kill']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['darkslateblue'])
plt.xticks(rotation=50)
plt.title("Number of killed ",fontsize=25)
plt.ylabel('Number of people',fontsize=20)
plt.xlabel('Attack type',fontsize=20)
plt.show()


# In[29]:


data[['Attacktype','Wound']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['blue'])
plt.xticks(rotation=50)
plt.title("Number of wounded  ",fontsize=25)
plt.ylabel('Number of people',fontsize=20)
plt.xlabel('Attack type',fontsize=20)
plt.show()


# In[33]:


plt.subplots(figsize=(20,10))
sns.countplot(data=data, x="Targettype",order=data['Targettype'].value_counts().index,palette="YlGnBu",edgecolor=sns.color_palette("mako"));
plt.xticks(rotation=90)
plt.xlabel("Attacktype",fontsize=20)
plt.ylabel("count",fontsize=20)
plt.title("Attack per year",fontsize=25)
plt.show()


# In[34]:


data['Group'].value_counts().to_frame().drop('Unknown').head(10).plot(kind='bar',color='black',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=25)
plt.xlabel("terrorist group name",fontsize=20)
plt.ylabel("Attack number",fontsize=20)
plt.show()


# In[35]:


df=data[['Group','Country','kill']]
df=df.groupby(['Group','Country'],axis=0).sum().sort_values('kill',ascending=False).drop('Unknown').reset_index().head(10)
df


# In[36]:


kill = data.loc[:,'kill']
print('Number of people killed by terror attack:', int(sum(kill.dropna())))


# In[37]:


typeKill = data.pivot_table(columns='Attacktype', values='kill', aggfunc='sum')
typeKill


# In[38]:


countryKill = data.pivot_table(columns='Country', values='kill', aggfunc='sum')
countryKill


# In[ ]:




