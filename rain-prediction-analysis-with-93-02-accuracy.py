#!/usr/bin/env python
# coding: utf-8

# # ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤRain Prediction Preprocessing

# ![image-2.png](attachment:image-2.png)

# <a id='part1'></a>
# # ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤPart 1

# ***

# ## Table of Contents
# [Part I](#part1)
#  - [1.Import Libiraries](#libiraries)
#  - [2.Import Dataset](#dataset)
#  - [3.Data Wrangling](#wrangling)
#  - [4.Univariate Analysis](#univariante_analysis)
#  - [5.Explore Categorical Variables](#categorical)
#  - [6.Explore Numerical Variables](#numerical)
#  - [7.Data Visalization](#visualization)
#  - [8.Multivariante Analysis](#multivariante_analysis)
#  - [9.Feature Engineering](#feature_eng)
#  - [9.Applying discretization on numeric data](#discretization)
#  
# [Part II](#part2)
#  - [1.Feature Encoding](#encoding)
#  - [2.Data Modelling](#modelling)
#  - [3.KNN](#knn)
#  - [4.Decision Tree](#decisiontree)
#  - [5.Naïve Bayes](#naive)
#  - [6.Hyperparameters Tuning](#hyperparameters-tuning)

# Rain is an essential part for our life. Clouds give the gift of rain to humans. Meteorological Authority tries to forecast when will it rain. So, I will try to predict whether it will rain in Australia tomorrow or not.

# Hence, in this notebook, I will implement Classification model with Python using Scikit-Learn and build a classifier to predict whether or not it will rain tomorrow in Australia. I will use the rain in Australia dataset for this project.

# <a id='libiraries'></a>
# ##  Import Libraries

# In[4]:


import numpy as np # Numerical Computations
import pandas as pd # Data Preprocessing

# Import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# <a id='dataset'></a>
# ##  Import Dataset

# In[2]:


df = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')


# <a id='wrangling'></a>
# ##  Data Wrangling

# - we have imported the data.
# - now, its time to explore the data to gain insights about it.

# ### Preview dataset

# In[3]:


df.head()


# ### View dimention of dataset

# In[4]:


df.shape


# we can see that there are 145460 row and 23 columns in the dataset

# ### View column names

# In[5]:


col_names = df.columns
col_names


# ### Checking For datatypes of the attributes

# In[6]:


df.info()


# Comment
# - We can see that the dataset contains mixture of categorical and numerical variables.
# - Categorical variables have data type : object
# - Numerical variables have data type : float64
# - Also, there are missing values in data set, we will explore it later.

# 
# ### View statistical properties of dataset

# In[7]:


df.describe()


# <a id='univariante_analysis'></a>
# ##  Univariate Analysis

# Explore **RainTomorrow** Target variable

# #### Check for missing values 

# In[8]:


df['RainTomorrow'].isnull().sum()


# We can see that there are 3267 missing values in "RainTomorrow"

# #### Check for unique values

# In[9]:


df['RainTomorrow'].unique()


# #### View the frequency of values 

# In[10]:


df['RainTomorrow'].value_counts(dropna=False)


# Important points to note
# - There are 3267 "NaN" missing values
# - There are 31877 "Yes" that it will rain
# - there are 110316 "No" that it wont rain 

# ### View percentage of frecquency values 

# In[11]:


rain = df.RainTomorrow.fillna('null value').value_counts(normalize=True)
rain


# Hence
# - We can see that the total number of rain tomorrow value : No = 76% 
# - We can see that the total number of rain tomorrow value : Yes = 22%
# - We can see that the total number of rain tomorrow value : Missing values = 2%     

# ### Visualize frequency distribution of RainTomorrow variable

# In[12]:


from IPython.core.display import display, HTML
display(HTML("<div class='tableauPlaceholder' id='viz1652722975843' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ra&#47;RainTomorrowFrequencies&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='RainTomorrowFrequencies&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ra&#47;RainTomorrowFrequencies&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1652722975843');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"))


# <a id='categorical'></a>
# ### Explore Categorical Variables

# Showing categorical variables in dataset

# In[13]:


categorical = df.select_dtypes(include=['object'])
categorical.head()


#  Summary of categorical variables 
#  - There are 6 categorical variables. They are given by Location, WindGustDir, WindDir9am, WindDir3pm, RainToday and RainTomorrow
#  - There are two binary categorical variables - RainToday and RainTomorrow
#  - RainTomorrow is the target variable.

# ### Missing values in Categorical Variables

# In[14]:


categorical.isna().sum().to_frame('number of null values')


# #### Number of labels: cardinality

# >The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is known as high cardinality. High cardinality may causes some serious problems in the machine learning model. So, I will check for high cardinality.

# In[15]:


# Check for cardinality in categorical variables

for var in categorical:
    print(var, ' contains ', df[var].nunique(), ' labels')


# We can see that there is a Date variable which needs to be preprocessed. I will preprocess in the following section.
# 
# All the other variables contain relatively smaller number of variables.

# #### Feature Engineering of Date Variable

# In[16]:


# Convert date to Datetime datatype

df['Date'] = pd.to_datetime(df['Date'])


# In[17]:


# Separate date feature to 3 attributes

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


# In[18]:


# Final check
df.head()


# #### Explore Categorical Variables one by one 

# In[19]:


new_categorical = df.select_dtypes(include=['object'])
new_categorical.head()


# ****

# In all variables :
# - I will check number of labels and show it.
# - Convert categorical variable into dummy/indicator variables.

# <a id='numerical'></a>
# ### Explore Numerical Variables 

# In[20]:


Numerical = df.select_dtypes(include=['float64','int'])
Numerical.head()


# In[21]:


Numerical.columns


# ### Check for duplicated values

# In[22]:


df[df.duplicated()]


# > Data does not contain any duplicates between attributes

# ### Missing values in numerical variables

# In[23]:


Numerical.isnull().sum()


# ### Check summary statistics

# In[24]:


Numerical.describe()


# <a id='visualization'></a>
# # ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤData Visualization

# ## Locations Frequencies

# In[25]:


from IPython.core.display import display, HTML
display(HTML("<div class='tableauPlaceholder' id='viz1652712003286' style='position: relative'><noscript><a href='#'><img alt='RainFall Per Location ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ra&#47;RainFallinLocations&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='RainFallinLocations&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ra&#47;RainFallinLocations&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1652712003286');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"))


# Here, We took an overview on the average amount of rainfall for each city in Australia and the frequency of each rainfall.

# In[26]:


#  plot Numerical Data

a = 4  # number of rows
b = 4  # number of columns
c = 1  # initialize plot counter
fig = plt.figure(figsize=(40,20))
for i in df:
    if df[i].dtype=='float64':
        plt.subplot(a, b, c)
        sns.distplot(df[i])
        c = c+1
    else:
        continue
        
plt.tight_layout()
plt.show()       


# ## Date Plot

# We are going to plot features with datetime. Here, I am going to use date from last 3 years.

# In[27]:


df_dateplot = df.iloc[-950:,:]
plt.figure(figsize=[20,5])
plt.plot(df_dateplot['Date'],df_dateplot['MinTemp'],color='blue',linewidth=1, label= 'MinTemp')
plt.plot(df_dateplot['Date'],df_dateplot['MaxTemp'],color='red',linewidth=1, label= 'MaxTemp')
plt.fill_between(df_dateplot['Date'],df_dateplot['MinTemp'],df_dateplot['MaxTemp'], facecolor = '#EBF78F')
plt.title('MinTemp vs MaxTemp by Date')
plt.legend(loc='lower left', frameon=False)
plt.show()


# - Above plot shows that the MinTemp and MaxTemp relatively increases and decreases every year.
# - The weather conditions are always opposite in the two hemispheres. As, the Australia is situated in the southern hemisphere. The seasons are bit different.
# - As you can see that, December to February is summer; March to May is autumn; June to August is winter; and September to November is spring.

# ### WindGustSpeed

# In[28]:


df_dateplot = df.iloc[-950:,:]
plt.figure(figsize=[20,5])
plt.plot(df_dateplot['Date'], df_dateplot['WindGustSpeed'], color='darkorange', linewidth=2, label='WindGustSpeed')
plt.legend(loc='upper left', frameon=False)
plt.title('WindGustSpeed by Date')
plt.show()


# ### Pressure9am and Pressure3am 

# In[29]:


df_dateplot = df.iloc[-950:,:]
plt.figure(figsize=[20,5])
plt.plot(df_dateplot['Date'],df_dateplot['Pressure9am'],color='blue', linewidth=2, label= 'Pressure9am')
plt.fill_between(df_dateplot['Date'],df_dateplot['Pressure9am'],df_dateplot['Pressure3pm'], facecolor = '#EBF78F')
plt.plot(df_dateplot['Date'],df_dateplot['Pressure3pm'],color='red', linewidth=2, label= 'Pressure3pm')
plt.legend(loc='upper left', frameon=False)
plt.title('Pressure9am vs Pressure3pm by Date')
plt.show()


# We can see that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns may contain outliers.
# 
# I will draw boxplots to visualize outliers in the above variables.

# ## Outliers Detection

# - An outlier is an observation that lies an abnormal distance from other values in a random sample from a population.
# - We are using Boxplot to detect the outliers of each features in our dataset, where any point above or below the whiskers represent an outlier. This is also known as “Univariate method” as here we are using one variable outlier analysis.

# We can see that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns may contain outliers.
# 
# I will draw boxplots to visualise outliers in the above variables.

# In[30]:


# Remove date

df.drop('Date',inplace= True,axis=1)


# In[31]:


from IPython.core.display import display, HTML
display(HTML("<div class='tableauPlaceholder' id='viz1652724172312' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;We&#47;Weather_16526360468760&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Weather_16526360468760&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;We&#47;Weather_16526360468760&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1652724172312');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"))


# ### Find all  outliers

# In[32]:


# Find outliers for Rainfall variable
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 1.5)
print('{lowerboundary} > Rainfall outliers are values > {upperboundary}'.format(lowerboundary="{:.2f}".format(Lower_fence), upperboundary="{:.2f}".format(Upper_fence)))

# Find outliers for Evaporation variable
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 1.5)
print('{lowerboundary} > Evaporation outliers are values > {upperboundary}'.format(lowerboundary="{:.2f}".format(Lower_fence), upperboundary="{:.2f}".format(Upper_fence)))

# Find outliers for WindSpeed9am variable
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 1.5)
print('{lowerboundary} > WindSpeed9am outliers are values > {upperboundary}'.format(lowerboundary="{:.2f}".format(Lower_fence), upperboundary="{:.2f}".format(Upper_fence)))

# Find outliers for WindSpeed3pm variable
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 1.5)
print('{lowerboundary} > WindSpeed3pm outliers are values > {upperboundary}'.format(lowerboundary="{:.2f}".format(Lower_fence), upperboundary="{:.2f}".format(Upper_fence)))

# Find outliers for WindGustSpeed variable
IQR = df.WindGustSpeed.quantile(0.75) - df.WindGustSpeed.quantile(0.25)
Lower_fence = df.WindGustSpeed.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.WindGustSpeed.quantile(0.75) + (IQR * 1.5)
print('{lowerboundary} > WindGustSpeed outliers are values > {upperboundary}'.format(lowerboundary="{:.2f}".format(Lower_fence), upperboundary="{:.2f}".format(Upper_fence)))


# Important point to note :
# - For Rainfall, the minimum and maximum values are 0.0 and 371.0 So, the outliers are values > 2.0
# - For Evaporation, the minimum and maximum values are 0.0 and 145.0 So, the outliers are values > 14.6
# - For WindSpeed9am, the minimum and maximum values are 0.0 and 130.0 So, the outliers are values > 37.0
# - For WindSpeed3pm, the minimum and maximum values are 0.0 and 87.0 So, the outliers are values > 40.5
# - For WindGustSpeed, the minimum and maximum values are 6.0 and 135.0 So, the outliers are values > 73.5

# ************

# <a id='multivariante_analysis'></a>
# ## Multivariante Analysis 

# - An important step in EDA is to discover patterns and relationships between variables in the dataset.
# 
# - I will use heatmap and pair plot to discover the patterns and relationships in the dataset.
# 
# - First of all, I will draw a heatmap.

# In[33]:


correlation = df.corr()


# In[34]:


plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
plt.show()


# Interpretation
# 
# - From the above correlation heatmap, we can conclude that:
# 
# 
# ㅤㅤㅤ- MinTemp and MaxTemp variables are highly positively correlated (correlation coefficient = 0.74).
# 
# 
# ㅤㅤㅤ- MinTemp and Temp3pm variables are also highly positively correlated (correlation coefficient = 0.71).
# 
# 
# ㅤㅤㅤ- MinTemp and Temp9am variables are strongly positively correlated (correlation coefficient = 0.90).
# 
# 
# ㅤㅤㅤ- MaxTemp and Temp9am variables are strongly positively correlated (correlation coefficient = 0.89).
# 
# 
# ㅤㅤㅤ- MaxTemp and Temp3pm variables are also strongly positively correlated (correlation coefficient = 0.98).
# 
# 
# ㅤㅤㅤ- WindGustSpeed and WindSpeed3pm variables are highly positively correlated (correlation coefficient = 0.69).
# 
# 
# ㅤㅤㅤ- Pressure9am and Pressure3pm variables are strongly positively correlated (correlation coefficient = 0.96).
# 
# 
# ㅤㅤㅤ- Temp9am and Temp3pm variables are strongly positively correlated (correlation coefficient = 0.86)

# ### Remove correlated attributes 

# > Remove 'Temp9am', 'Temp3pm', 'Pressure3pm' as this columns are irrelevant attributes

# In[35]:


df.drop(['Temp9am','Temp3pm','Pressure3pm'],inplace= True,axis=1)


# ### Show columns  

# In[36]:


df.columns


# #### Show New Data after
#     - Remove duplicate records
#     - Remove irrelevant attributes
#     - Remove correlated attributes

# In[37]:


df.head()


# - We found that shape of dataset after remove noise data have 22 columns

# <a id='feature_eng'></a>
# # ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤFeature Engineering

# ***

# ## Declare feature vector and target variable

# **Feature Engineering** is the process of transforming raw data into useful features that help us understand our model better and increase its predictive power.
# <br>
# I will carry out feature engineering on different types of variables.
# 
# First, I will display categorical and numerical variables again separately.

# In[38]:


# Display categorical variables

categorical = [col for col in df.columns if df[col].dtypes == 'O']
categorical


# In[39]:


# Display numerical variables

numerical = [col for col in df.columns if df[col].dtypes != 'O']
numerical


# ### Missing values in numerical variables

# In[40]:


# Checking on missing values in numerical variables in X_train

df[numerical].isnull().sum()


# In[41]:


# Impute missing values in X_train and X_test with respective column median in X_train

for col in numerical:
    col_median=df[col].median()
    df[col].fillna(col_median, inplace=True) 


# In[42]:


# Check again missing values in numerical variables in X_train

df[numerical].isnull().sum()


# In[43]:


# Impute missing categorical variables with most frequent value
for col in categorical:
    col_mode=df[col].mode()[0]
    #print(col_mode)
    df[col].fillna(col_mode, inplace=True) 


# In[44]:


# Check missing values in categorical variables in X_train
df[categorical].isnull().sum()


# ### Outliers Engineering in numerical variables 

# We have seen that Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns contain outliers.
# <br>
# I will use top-coding approach to cap maximum values and remove outliers from the above variables.

# In[45]:


for i in df:
    if df[i].dtype=='float64':
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3-q1
        Lower_tail = q1 - 1.5 * iqr
        Upper_tail = q3 + 1.5 * iqr
        med = np.median(df[i])
        for j in df[i]:
            if j > Upper_tail or j < Lower_tail:
                df[i] = df[i].replace(j, med)
    else:
        continue


# In[46]:


df.isnull().sum()


# <a id='discretization'></a>
# # Applying discretization on numeric data

# In[47]:


# Main smoothing cell
# Applying discretication on numeric data

for col in numerical:
    print(col, end=' ')
    percentiles = list(df[col].describe()[3:])
    print(percentiles)
    for b in range(4):
        binn = df[col].between(percentiles[b], percentiles[b+1], inclusive='left')
        bin_mean = df[binn][col].mean()
        df.loc[binn, col] = bin_mean


# In[48]:


# Checking

for col in numerical:
    print(col, df[col].unique())


# <a id='part2'></a>
# # ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤPart 2

# <a id='encoding'></a>
# ## Feature Encoding

# In[49]:


# Unique values for locations

new_categorical['Location'].unique()


# In[50]:


# Dummies of Locations

pd.get_dummies(df.Location, drop_first=True).head()


# #### Explore WindGustDir variable

# In[51]:


# Unique values for WindGustDir

new_categorical['WindGustDir'].unique()


# In[52]:


# Dummies of WindGustDir

pd.get_dummies(df.WindGustDir,drop_first=True,dummy_na=True).head()


# #### Explore WindDir9am variable

# In[53]:


# Unique values for WindDir9am

new_categorical['WindDir9am'].unique()


# In[54]:


# Dummies of WindDir9am

pd.get_dummies(df.WindDir9am,drop_first=True,dummy_na=True).head()


# #### Explore WindDir3pm variable

# In[55]:


# Unique values for WindDir3pm

new_categorical['WindDir3pm'].unique()


# In[56]:


# Dummies of WindDir3pm

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()


# #### Explore RainToday variable

# In[57]:


# Unique values for RainToday

new_categorical['RainToday'].unique()


# In[58]:


# Count each value in RainToday feature

df.RainToday.value_counts()


# In[59]:


# Dummies for RainToday

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()


# In[60]:


# Spliting data to X and Y : Features and Target

X = df.drop(['RainTomorrow'], axis=1)
Y = df['RainTomorrow']


# In[61]:


# Getting Dummies for Target feature

Y_dumies = pd.get_dummies(Y, drop_first=True)


# In[62]:


# Encoding the Target feature

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_coded = encoder.fit_transform(X)


# In[63]:


X_coded.head()


# In[64]:


# Enconding all features in X_coded

X_conc = pd.get_dummies(X_coded)


# In[65]:


X_conc.head()


# In[66]:


# Standarize data Scale
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
X_scale = Scaler.fit_transform(X_conc)
X_scale = pd.DataFrame(X_scale, columns=[X_conc.columns])


# In[67]:


X_scale.head()


# In[68]:


# Splitting data to train and test 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scale, Y_dumies, test_size = 0.2, random_state = 0)


# In[69]:


pd.DataFrame(X_train).head()


# <a id='modelling'></a>
# ## Data Modeling
# - Package Importing

# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB,GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, validation_curve, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report


# ## Function To apply all Models 

# In[71]:


# Plotting ROC Curve
def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    
# Run given model
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    #coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    cm = confusion_matrix(y_test, y_pred)
    
    

    print(f'Training set score: {round(model.score(X_train, y_train) * 100, 2)}%')
    print ('--')
    print("Accuracy = {}%".format(round(accuracy*100, 2)))
    print ('--')
    print("ROC Area under Curve = {}".format(roc_auc))
    print ('--')
   # print("Cohen's Kappa = {}".format(coh_kap))
    #print ('--')
    print("Time taken = {}".format(time_taken))
    print ('--')
   # print(classification_report(y_test,y_pred,digits=5))
    print ('Confusion Matrix\n\n', cm)
    print ('--')
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_cur(fper, tper)
    
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    return model, accuracy, time_taken


# <a id='knn'></a>
# # KNN

# In[72]:


# Applying KNN model on data

model_dt = KNeighborsClassifier()
model_dt, accuracy_dt,  tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)


# <a id='decisiontree'></a>
# # DecisionTree

# In[73]:


# Applying Decision Tree model on data
params_dt = {'max_depth': 20,
             'max_features': "sqrt",
            'splitter':'best',
            'max_leaf_nodes':None}

model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt,  tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)
import matplotlib.pyplot as plt
dt = DecisionTreeClassifier()


# ### Display Decision Tree

# In[74]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=2, random_state=0)
fitted=dt.fit(X_train, y_train);

y_pred = dt.predict(X_test)


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text#both are new in 0.21
plt.figure(figsize= (8, 6))
plot_tree(fitted, filled=True);

from IPython.display import Image  
from six import StringIO 
from sklearn.tree import export_graphviz
import pydot

dot_data = StringIO()  
export_graphviz(dt, out_file=dot_data,feature_names=X_train.columns,filled=True,rounded=True,)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())
# <a id='naive'></a>
# # Naïve Bayes

# In[75]:


# Applying Naïve Bayes model on Data

params_dt = {'alpha': 0.1,
             'binarize': "6.0",
            'fit_prior':True}

model_dt = BernoulliNB(alpha=1,binarize=1,fit_prior=True)
model_dt, accuracy_dt,  tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)


# <a id='hyperparameters-tuning'></a>
# # Hyperparameters Tuning

# I will use **GridSearchCV** to find the best hyperparameters.
# <br/>
# So, what is it ?
# <br/>
# **Cross-Validation** (**CV**): Number of cross-validation you have to try for each selected set of hyperparameters.
# <br/>
# **Verbose**: You can set it to 1 to get the detailed print out while you fit the data to GridSearchCV at the end, you can select the best parameters from the listed hyperparameters

# In[76]:


# Assume values for some paramaters

tree_params = param_dist = {
    "criterion" : ['gini', 'entropy', 'log_loss'],
    "max_depth" : [33, 40, None],
    #'splitter' : ['best', 'random'],
    'max_features' : ['sqrt', 'log2'],
    #'random_state' : [4,5,6,7,8,9,None],
    #'max_leaf_nodes' : [5,6,7,8,9,None]
}
# apply gridsearch model 
tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring = 'accuracy',cv = 4)
tree_grid.fit(X_train, y_train)
tree_grid.best_estimator_
y_pred = tree_grid.predict(X_test)
best_score = tree_grid.best_score_
best_params = tree_grid.best_params_
precision = precision_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print(f'Accuracy =: {round(tree_grid.score(X_train, y_train) * 100, 2)}%')

print ('--')
print ('Best Parameters is', best_params)
print ('--')
print ('ROC Score is', roc)
print ('--')
print ('Recall Score is ', recall)
print ('--')
print ('Confusion Matrix\n\n', cm)


# ### ROC curve
# 
# **roc curve :**is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied

# In[77]:


y_pred_prob = tree_grid.predict_proba(X_test)[:,1]

# Instantiating the roc_cruve
fpr,tpr,threshols=roc_curve(y_test,y_pred_prob)

# Plotting the curve
plt.figure(figsize = (8, 8))
plt.plot([0,1],[0,1],"k--",'r+')
figsize=(16,12)
plt.plot(fpr,tpr,color = '#b01717', label = 'ROC = %0.3f' % roc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DecisionTree ROC Curve")
plt.legend()
plt.show()


# ## Confusion matrix

# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called **Type I error**.
# 
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error**.
# 
# These four outcomes are summarized in a confusion matrix given below.

# In[79]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# The confusion matrix shows 21314 + 2736 = 24050 correct predictions and 1412 + 3630 = 5050 incorrect predictions.
# 
# In this case, we have
# 
# - True Positives (Actual Positive:1 and Predict Positive:1) - 21314
# 
# 
# - True Negatives (Actual Negative:0 and Predict Negative:0) - 2736
# 
# 
# - False Positives (Actual Negative:0 but Predict Positive:1) - 1412 (Type I error)
# 
# 
# - False Negatives (Actual Positive:1 but Predict Negative:0) - 3630 (Type II error)

# In[80]:


cm_matrix = pd.DataFrame(data = cm, columns = ['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
cm_matrix.head()


# In[81]:


sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu');


# # Made by Devesh Binwal

# #  made by pankaj

# In[ ]:




