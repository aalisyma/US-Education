# Importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, RobustScaler
from numpy import absolute
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from numpy.linalg import inv
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor
from numpy.linalg import inv
import matplotlib.ticker as mtick
from sklearn.cluster import KMeans
from pandas.plotting import parallel_coordinates
from itertools import cycle, islice
# %matplotlib inline

# Commented out IPython magic to ensure Python compatibility.
# Changing to the directry containing the dataset
# %cd /content/drive/MyDrive/Applied ML Project

# Suppressing warnings 
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

"""# Data

Importing the dataset
"""

# Importing dataset 
df = pd.read_csv('states_all.csv')

# Checking the column names
df.columns

# Analysing the dataset by checking its null value numbers by column
df.isna().sum()*100/df.shape[0]

df.describe().T

df.columns

# Robust Scaling
robust_scaler = RobustScaler(with_centering=True)

cols = ['YEAR', 'ENROLL', 'TOTAL_REVENUE',
       'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE',
       'TOTAL_EXPENDITURE', 'INSTRUCTION_EXPENDITURE',
       'SUPPORT_SERVICES_EXPENDITURE', 'OTHER_EXPENDITURE',
       'CAPITAL_OUTLAY_EXPENDITURE', 'GRADES_PK_G', 'GRADES_KG_G',
       'GRADES_4_G', 'GRADES_8_G', 'GRADES_12_G', 'GRADES_1_8_G',
       'GRADES_9_12_G', 'GRADES_ALL_G', 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE',
       'AVG_READING_4_SCORE', 'AVG_READING_8_SCORE']

# Not applying Robust Scaler to PRIMARY_KEY (will be dropped eventually) and STATE (will be target encoded)
df[cols] = robust_scaler.fit_transform(df[cols])

"""# Exploratory Data Analysis on the Dataset

## Part 1
"""

# Plotting the breakdown of revenue over years
df_revenue = df[["YEAR", 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']].dropna(axis = 0)

plt.figure(figsize=(20,6))
ax = df_revenue.groupby("YEAR").sum().plot.barh(x='TOTAL_REVENUE', stacked=True, cmap = 'viridis', figsize=(8,6))
ax.set_title('YEARLY TOTAL REVENUE', fontsize=10)
ax.set_yticklabels(list(range(1992, 2017)), rotation=0)
ax.set_xlabel("TOTAL REVENUE (in 100 million $)")
ax.set_ylabel("YEAR")

# Plotting a boxplot of the distribution of local revenue over years
plt.figure(figsize = (12,8))
sns.boxplot(y = df_revenue['LOCAL_REVENUE'], x = df_revenue['YEAR'])
plt.ylabel("LOCAL REVENUE (in 10 million $)")

num_features = []
target_var = [ 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE']
cat_features = ["PRIMARY_KEY","STATE","YEAR"]
for col in df.columns:
    num_features.append(col)

num_features = [x for x in num_features if x not in cat_features and x not in target_var]
num_features

# Correlation heatmap
plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(df[num_features].corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

# Creating a new column EXPENDITURE_PER_ENROLLMENT in order to target encode STATE column based on it

df["EXPENDITURE_PER_ENROLLMENT"] = df["TOTAL_EXPENDITURE"]/df["ENROLL"]

# Dropping highly correlated features
CorrM = df[num_features].corr().abs()
upper_tri = CorrM.where(np.triu(np.ones(CorrM.shape),k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.97)]
df = df.drop(df[to_drop], axis=1, inplace=False)
df

# Mean imputation of missing values from the corresponding group of records from the associated state
for column in df.columns[3:]:
  df[column] = df[column].fillna(df.groupby('STATE')[column].transform('mean'))
df.dropna(axis=0, inplace=True)

# Performing target encoding of the state column using the mean of the EXPENDITURE_PER_ENROLLMENT column for the corresponding state 
df["STATE"] = df.groupby("STATE")["EXPENDITURE_PER_ENROLLMENT"].transform("mean")

updated_num = []

target_var = [ 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE']
cat_features = ["PRIMARY_KEY","STATE","YEAR"]
for col in df.columns:
    updated_num.append(col)

updated_num = [x for x in updated_num if x not in cat_features and x not in target_var]
updated_num

# Calculating the TOTAL_SCORE column, i.e, the sum of the AVG scores
df["TOTAL_SCORE"] = df.apply(lambda x: x['AVG_MATH_4_SCORE'] + x['AVG_MATH_8_SCORE'] + x['AVG_READING_4_SCORE'] + x['AVG_READING_8_SCORE'], axis = 1)

# Dropping PRIMARY KEY column as its not used for training
df.drop("PRIMARY_KEY", axis = 1, inplace=True)

# Saving the final dataset after cleaning, imputation and encoding
df.to_csv("us_education_final.csv", index=False)

# df = pd.read_csv("us_education_final.csv")



"""## Part 2"""

data = pd.read_csv('states_all.csv')
pd.set_option('display.max_columns', None)
data.head()

data.columns

"""### Plot histogram for the top 10 states with the highest accumlated total expenditures"""

year = data['YEAR'].unique()
state = data['STATE'].unique()

"""Checking for missing values in TOTAL_EXPENDITURE."""

print(data['TOTAL_EXPENDITURE'].isnull().values.any())
print(data['ENROLL'].isnull().values.any())

data['ENROLL']

year_state_exp = data[['YEAR', 'STATE','TOTAL_EXPENDITURE', 'ENROLL']]
year_state_exp = year_state_exp.drop(year_state_exp[(year_state_exp.YEAR <= 1992) | 
                                                    (year_state_exp.YEAR >= 2017) |
                                                    (year_state_exp.STATE == 'DODEA') |
                                                    (year_state_exp.STATE == 'NATIONAL')].index)
year_state_exp = year_state_exp.reset_index()
year_state_exp_summary = pd.DataFrame(columns = ['State', 'Sum_Expenditure', 'Sum_Enrollment'])
print(year_state_exp['TOTAL_EXPENDITURE'].isnull().values.any())

for index, row in year_state_exp.iterrows():
    state = row['STATE']
    exp = row['TOTAL_EXPENDITURE']
    enroll = row['ENROLL']
    if state in year_state_exp_summary['State'].values:
        year_state_exp_summary.loc[year_state_exp_summary['State'] == state, 'Sum_Expenditure'] += exp
        year_state_exp_summary.loc[year_state_exp_summary['State'] == state, 'Sum_Enrollment'] += enroll
    else:
        df = {'State': state, 'Sum_Expenditure': exp, 'Sum_Enrollment': enroll}
        year_state_exp_summary = year_state_exp_summary.append(df, ignore_index = True)

year_state_exp_summary.sort_values(by=['Sum_Expenditure'], inplace=True, ascending=False)
year_state_exp_summary

top_10 = year_state_exp_summary[0:10]
last_10 = year_state_exp_summary[41:51]
last_10.sort_values(by=['Sum_Expenditure'], inplace=True, ascending=True)
top_10_states = list(top_10['State'])
last_10_states = list(last_10['State'])
labels = []
for i in range(len(top_10_states)):
    l = top_10_states[i] + ", " + last_10_states[i]
    labels.append(l)

fig, ax = plt.subplots(2, figsize=(15, 12))
fig.tight_layout()
index = np.arange(10)
top = ax[0].bar(index, top_10['Sum_Expenditure']/1e6, 0.35, label="Top 10 states")
last = ax[0].bar(index+0.35, last_10['Sum_Expenditure']/1e6, 0.35, label="Last 10 states")
top_e = ax[1].bar(index, (top_10['Sum_Expenditure'])/top_10['Sum_Enrollment'], 0.35, label="Top 10 states")
last_e = ax[1].bar(index+0.35, (last_10['Sum_Expenditure'])/last_10['Sum_Enrollment'], 0.35, label="Last 10 states")

ax[0].set_ylabel('Accumulative Expenditures ($M)')
ax[0].set_title('The Top and Last 10 States with the Highest and Lowest Accumulative Expenditures from 1993-2016')
ax[0].legend()
ax[1].set_xlabel('States')
ax[1].set_ylabel('Accumulative Expenditures per Enrollment ($/person)')
ax[1].set_title('Accumulative Expenditures per Enrollment from 1993-2016')
ax[1].set_xticks(index + 0.35 / 2)
ax[1].set_xticklabels(labels, rotation=45)
ax[1].legend()

plt.show()
fig.savefig('accumulated_enrollment.png', bbox_inches='tight')

year_state_exp_summary['Exp_Enroll'] = year_state_exp_summary['Sum_Expenditure']/year_state_exp_summary['Sum_Enrollment']
year_state_exp_summary.sort_values(by=['Exp_Enroll'], inplace=True, ascending=False)
top_10_per = year_state_exp_summary[0:10]
top_labels = list(top_10_per['State'])
last_10_per = year_state_exp_summary[41:51]
year_state_exp_summary.sort_values(by=['Exp_Enroll'], inplace=True, ascending=True)
last_labels = list(last_10_per['State'])
per_labels = []
for i in range(len(top_labels)):
    l = top_labels[i] + ", " + last_labels[i]
    per_labels.append(l)
year_state_exp_summary

fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
index = np.arange(10)
top = ax.barh(index, top_10['Sum_Expenditure']/1e6, 0.35, label="Top 10 states by Accumulative Expenditures")
last = ax.barh(index+0.35, last_10['Sum_Expenditure']/1e6, 0.35, label="Last 10 states by Accumulative Expenditures")
ax.invert_yaxis()
ax.set_ylabel('States')
ax.set_xlabel('Accumulative Expenditures ($M)')
ax.set_title('The Top and Last 10 States with the Highest and Lowest Accumulative Expenditures from 1993-2016')
ax.set_yticks(index + 0.35 / 2)
ax.set_yticklabels(labels)
ax.legend()
ax.bar_label(top, fmt = '%.2f', padding=3)
ax.bar_label(last, fmt = '%.2f', padding=3)
plt.show()
fig.savefig('accumulated_exp.png', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout()
top_e = ax.barh(index, top_10_per['Exp_Enroll'], 0.35, label="Top 10 states by Expenditures per Enrollment")
last_e = ax.barh(index+0.35, last_10_per['Exp_Enroll'], 0.35, label="Last 10 states by Expenditures per Enrollment")
ax.invert_yaxis()
ax.set_xlabel('States')
ax.set_ylabel('Accumulative Expenditures per Enrollment ($/person)')
ax.set_title('The Top and Last 10 States with the Highest and Lowest Accumulative Expenditures per Enrollment from 1993-2016')
ax.set_yticks(index + 0.35 / 2)
ax.set_yticklabels(per_labels)
ax.legend()
ax.bar_label(top_e, fmt = '%.2f', padding=3)
ax.bar_label(last_e, fmt = '%.2f',padding=3)
plt.show()
fig.savefig('exp_enroll.png', bbox_inches='tight')

"""Instruction expenditures
Expenditures for activities related to the interaction between teachers and students. Include salaries and benefits for teachers and teacher aides, textbooks, supplies and purchased services. These expenditures also include expenditures relating to extracurricular and cocurricular activities.

Support services expenditures
An expenditure function divided into seven subfunctions: student support services, instructional staff support, general administration, school administration, operations and maintenance, student transportation, and other support services.

Capital outlay expenditures
The capital outlay fund of the school district is a fund provided by law to meet expenditures of which result in the acquisition of or lease of or additions to real property, plant, or equipment.

sources: https://nces.ed.gov/pubs2011/expenditures/appendix_b.asp#:~:text=instruction%20expenditures,textbooks%2C%20supplies%20and%20purchased%20services.
https://legislativeaudit.sd.gov/resources/schools/accountingmanual/School_Section_12/School_Section%2012_Interpretation_13.pdf

### Compare grades among the top and last 10 states
"""

year_exp_grades = data[['YEAR', 'STATE','TOTAL_EXPENDITURE', 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE', 
                       'AVG_READING_4_SCORE', 'AVG_READING_8_SCORE']]
year_exp_grades = year_exp_grades.drop(year_exp_grades[(year_exp_grades.YEAR <= 1992) | 
                                                    (year_exp_grades.YEAR >= 2017) |
                                                    (year_exp_grades.STATE == 'DODEA') |
                                                    (year_exp_grades.STATE == 'NATIONAL')].index)
year_exp_grades = year_exp_grades.reset_index()
year_exp_grades = year_exp_grades.dropna(subset=['AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE', 
                                                 'AVG_READING_4_SCORE', 'AVG_READING_8_SCORE'])
year_exp_grades

year_exp_grades.isnull().values.any()

year_exp_grades_summary = pd.DataFrame(columns = ['Year', 'Sum_Expenditure', 'Sum_Score'])

for index, row in year_exp_grades.iterrows():
    year = row['YEAR']
    exp = row['TOTAL_EXPENDITURE']
    scores = row['AVG_MATH_4_SCORE'] + row['AVG_MATH_8_SCORE'] + row['AVG_READING_4_SCORE'] + row['AVG_READING_8_SCORE']
    if year in year_exp_grades_summary['Year'].values:
        year_exp_grades_summary.loc[year_exp_grades_summary['Year'] == year, 'Sum_Score'] += scores
        year_exp_grades_summary.loc[year_exp_grades_summary['Year'] == year, 'Sum_Expenditure'] += exp
    else:
        df = {'Year': year, 'Sum_Expenditure': exp, 'Sum_Score': scores}
        year_exp_grades_summary = year_exp_grades_summary.append(df, ignore_index = True)

year_exp_grades_summary['Sum_Expenditure'] = year_exp_grades_summary['Sum_Expenditure']/650900851.0
year_exp_grades_summary['Sum_Score'] = year_exp_grades_summary['Sum_Score']/51676.0
year_exp_grades_summary

years = list(year_exp_grades_summary['Year'])
years = [int(x) for x in years]

import seaborn as sns

fig, ax = plt.scatter(year_exp_grades_summary['Sum_Expenditure']/650900851.0, 
                      year_exp_grades_summary['Sum_Score']/51676.0)
ax.axline([0.85, 0.85], [1, 1])
for i, y in enumerate(years):
    ax.annotate(y, (year_exp_grades_summary['Sum_Expenditure'][i], year_exp_grades_summary['Sum_Score'][i]))
plt.show()

sns.scatterplot(data=year_exp_grades_summary,x='Sum_Expenditure',y='Sum_Score')
plt.axline([0.97, 0.97], [1, 1], label='y=x')
for i in range(year_exp_grades_summary.shape[0]):
    plt.text(x=year_exp_grades_summary.Sum_Expenditure[i]+0.002,
             y=year_exp_grades_summary.Sum_Score[i]+0.001,
             s=years[i], 
          fontdict=dict(color='red',size=10),
          bbox=dict(facecolor='yellow',alpha=0.5))
plt.xlabel('Normalized Accumulated Expenditures')
plt.ylabel('Normalized Scores')
plt.title('Expenditures and Academic Performances from 2003 to 2015')
plt.legend()

fig, ax = plt.subplots(figsize=(10, 6))
fig.tight_layout()
index = np.arange(7)
e = ax.bar(index, year_exp_grades_summary['Sum_Expenditure']/650900851.0, 0.35, label="Expenditure")
g = ax.bar(index+0.35, year_exp_grades_summary['Sum_Score']/51456.0, 0.35, label="Score")
ax.set_xlabel('Year')
ax.set_ylabel('Normalized Expenditure and Score')
ax.set_title('Expenditure Growth and Academic Performance from 2003 to 2015')
ax.set_xticks(index + 0.35 / 2)
ax.set_xticklabels(years)
ax.legend()
plt.show()
fig.savefig('exp_grades.png', bbox_inches='tight')

"""## Part 3"""

df_raw = pd.read_csv('states_all.csv')

num_features = ["TOTAL_REVENUE","FEDERAL_REVENUE","LOCAL_REVENUE","OTHER_EXPENDITURE","CAPITAL_OUTLAY_EXPENDITURE","GRADES_PK_G"]

df_raw['TOTAL_SCORE'] = df_raw.loc[:,['AVG_MATH_4_SCORE','AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_MATH_8_SCORE']].sum(axis=1)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
from plotly import subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import plotly.express as px
import datetime
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}
codes = {}
for i in us_state_to_abbrev:
    codes[i.upper()] = us_state_to_abbrev[i]

"""### Plot Different Features on Map"""

df = df_raw[df_raw.TOTAL_SCORE != 0].groupby('STATE')['TOTAL_SCORE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['TOTAL_SCORE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Total Score",
))

fig.update_layout(
    title_text = 'Total Score by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['TOTAL_REVENUE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['TOTAL_REVENUE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Total Revenue",
))

fig.update_layout(
    title_text = 'Total Revenue by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['FEDERAL_REVENUE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['FEDERAL_REVENUE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Federal Revenue",
))

fig.update_layout(
    title_text = 'Federal Revenue by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['LOCAL_REVENUE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['LOCAL_REVENUE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Local Revenue",
))

fig.update_layout(
    title_text = 'Local Revenue by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['OTHER_EXPENDITURE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['OTHER_EXPENDITURE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Other Expenditure",
))

fig.update_layout(
    title_text = 'Other Expenditure by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['CAPITAL_OUTLAY_EXPENDITURE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['CAPITAL_OUTLAY_EXPENDITURE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Capital Outlay Expenditure",
))

fig.update_layout(
    title_text = 'Capital Outlay Expenditure by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['GRADES_PK_G'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['GRADES_PK_G'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Pre-Kindergarten Students",
))

fig.update_layout(
    title_text = 'Pre-Kindergarten Students by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['AVG_MATH_4_SCORE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['AVG_MATH_4_SCORE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "AVG_MATH_4_SCORE",
))

fig.update_layout(
    title_text = 'AVG_MATH_4_SCORE by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['AVG_MATH_8_SCORE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['AVG_MATH_8_SCORE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "AVG_MATH_8_SCORE",
))

fig.update_layout(
    title_text = 'AVG_MATH_8_SCORE by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['AVG_READING_8_SCORE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['AVG_READING_8_SCORE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "AVG_READING_8_SCORE",
))

fig.update_layout(
    title_text = 'AVG_READING_8_SCORE by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['AVG_READING_4_SCORE'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['AVG_READING_4_SCORE'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "AVG_READING_4_SCORE",
))

fig.update_layout(
    title_text = 'AVG_READING_4_SCORE by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

df = df_raw.groupby('STATE')['GRADES_ALL_G'].mean().reset_index()
df['STATE'] = df['STATE'].map(codes)

fig = go.Figure(data=go.Choropleth(
    locations=df['STATE'], # Spatial coordinates
    z = df['GRADES_ALL_G'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Count of All Students",
))

fig.update_layout(
    title_text = 'Count of All Students by State',
    geo_scope='usa', # limite map scope to USA
    height= 550,
    width = 900
)

fig.show()

"""States with high total revenue are states with large number of students, but not necessarily states with high total score

States with highest total score across the years are 
Alaska, Illinois, and Vermont.

States with high total revenue across the years such as California, Texas and Florida are not the states with the highest total score. This may due to the fact that these states also rank among the top of the states that have a large number of total students.

### Compare Selected Features between Year 2003 and 2015 Among States
"""

df1 = df_raw[df_raw.YEAR == 2003].groupby('STATE')['TOTAL_SCORE'].mean().reset_index()
df1['STATE'] = df1['STATE'].map(codes)
df1.rename(columns={'TOTAL_SCORE':'TOTAL_SCORE_1'}, inplace=True)
df2 = df_raw[df_raw.YEAR == 2015].groupby('STATE')['TOTAL_SCORE'].mean().reset_index()
df2['STATE'] = df2['STATE'].map(codes)
df2.rename(columns={'TOTAL_SCORE':'TOTAL_SCORE_2'}, inplace=True)
df1 = df1.dropna()
df2 = df2.dropna()
plt.plot(df1.STATE.astype(str),df1.TOTAL_SCORE_1.astype(int),label='2003')
plt.plot(df2.STATE.astype(str),df2.TOTAL_SCORE_2.astype(int),label='2015')
#plt.xticks(rotation=90)
plt.legend()
plt.xlabel("States")
plt.ylabel("Total Score")
plt.title("Total Score Among States in 2003 and 2015")
plt.rcParams["figure.figsize"] = (14,3)
plt.show()

df = pd.merge(df1,df2, how = "left")
df['DIFF'] = df['TOTAL_SCORE_2'] - df['TOTAL_SCORE_1']
df.sort_values('DIFF')

"""For the count of total score, we can see that overall, the 
scores for different states increased from 2003 to 2015. 
However, there are vibrations. For example, some of the 
states had a large increase. The top being Hawaii, with a 
total score increase of 44 points. It was followed by 
Mississippi and Tennessee, each of which increased by 40 
points. Some of the states did not change much, especially 
Connecticut and Kansas, where total score remained the 
same. Relative ranking between states changed as an 
result.
"""

df1 = df_raw[df_raw.YEAR == 2003].groupby('STATE')['TOTAL_REVENUE'].mean().reset_index()
df1['STATE'] = df1['STATE'].map(codes)
df2 = df_raw[df_raw.YEAR == 2015].groupby('STATE')['TOTAL_REVENUE'].mean().reset_index()
df2['STATE'] = df2['STATE'].map(codes)
df1 = df1.dropna()
df2 = df2.dropna()
plt.plot(df1.STATE.astype(str),df1.TOTAL_REVENUE.astype(int),label='2003')
plt.plot(df2.STATE.astype(str),df2.TOTAL_REVENUE.astype(int),label='2015')
#plt.xticks(rotation=90)
plt.legend()
plt.xlabel("States")
plt.ylabel("Total Revenue")
plt.title("Total Revenue Among States in 2003 and 2015")
plt.rcParams["figure.figsize"] = (14,3)
plt.show()

"""For the amount of total revenue, we can see that almost all 
the states had a higher total revenue in 2015 as compared 
to 2003, but the overall trend remained the same.
"""

df1 = df_raw[df_raw.YEAR == 2003].groupby('STATE')['GRADES_ALL_G'].mean().reset_index()
df1['STATE'] = df1['STATE'].map(codes)
df2 = df_raw[df_raw.YEAR == 2015].groupby('STATE')['GRADES_ALL_G'].mean().reset_index()
df2['STATE'] = df2['STATE'].map(codes)
df1 = df1.dropna()
df2 = df2.dropna()
plt.plot(df1.STATE.astype(str),df1.GRADES_ALL_G.astype(int),label='2003')
plt.plot(df2.STATE.astype(str),df2.GRADES_ALL_G.astype(int),label='2015')
#plt.xticks(rotation=90)
plt.legend()
plt.xlabel("States")
plt.ylabel("Total Students")
plt.title("Total Students Among States in 2003 and 2015")
plt.rcParams["figure.figsize"] = (14,3)
plt.show()

"""For the number of total students, we can see that the 
number barely changed among the states in 2003 
compared to 2015, as the two lines almost align with each 
other. Some minor differences occurred such as Michigan 
had slightly less students and Texas had slightly more 
students in 2015.

Total revenue and number of students remain relatively the same between 2003 and 2015 among the states, but changes occur for total score

## Part 4
"""

df_raw = pd.read_csv("states_all.csv")

num_features = ["TOTAL_REVENUE","FEDERAL_REVENUE","LOCAL_REVENUE","OTHER_EXPENDITURE","CAPITAL_OUTLAY_EXPENDITURE","GRADES_PK_G"]

df_raw['TOTAL_SCORE'] = df_raw.loc[:,['AVG_MATH_4_SCORE','AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_MATH_8_SCORE']].sum(axis=1)

fig,ax=plt.subplots(3,2,figsize=(20,10),constrained_layout = True)
i=0
while i < len(num_features):
    for j in range(0,3):
        for k in range(0,2):
            f=num_features[i]
            sns.histplot(data=df_raw[f],ax=ax[j][k])
            ax[j][k].set_title(num_features[i])
            i+=1

i=0
while i < len(num_features):
  f=num_features[i]
  sns.jointplot(data=df_raw,x=f,y="TOTAL_SCORE",alpha=0.3)
  i+=1

df_raw.describe().T

plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(df_raw[num_features].corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);



"""## Part 5"""

# importing an external csv file containing the state to region mapping
df_region = pd.read_csv('drive/MyDrive/Colab Notebooks/AML/Project/archive/state_region_mapping.csv')
df = pd.read_csv('states_all.csv')
df_region.head()

def underscore(row):
  return row.replace(" ", "_").upper()

df_region['state'] = df_region.apply(lambda row : underscore(row['STATE']), axis = 1)

df_region.head()

df_merged = pd.merge(df_bar1, df_region, left_on=['STATE'], right_on=['state'], how='left')
df_merged.head()

df_merged2 = pd.merge(df_bar2, df_region, left_on=['STATE'], right_on=['state'], how='left')
df_merged2.head()

df_merged = df_merged[['state','REGION','average_revenue']]
df_merged2 = df_merged2[['state','REGION','average_expenditure']]

df_merged.head()

import matplotlib.ticker as mtick
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)

# average revenue by state in 2016

fig, ax = plt.subplots(figsize=(18, 18))
sns.barplot(x="average_revenue", y="state", data=df_merged,hue='REGION',dodge=False)
plt.legend(loc='center right',prop = {'size' : 20})
plt.title('Average Revenue By State', fontdict={'fontsize': 20})
plt.xlabel('Average Revenue ($ amount per student)', fontsize=18)
ax.xaxis.set_major_formatter(tick) 
plt.ylabel('State', fontsize=18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.savefig('state_region.png');

# average expenditure by state in 2016

fig, ax = plt.subplots(figsize=(18, 18))
sns.barplot(x="average_expenditure", y="state", data=df_merged2,hue='REGION',dodge=False)
plt.legend(loc='center right',prop = {'size' : 20})
plt.title('Average Expenditure By State', fontdict={'fontsize': 20})
plt.xlabel('Average Expenditure ($ amount per student)', fontsize=18)
ax.xaxis.set_major_formatter(tick) 
plt.ylabel('State', fontsize=18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13);

"""**Scatter plots**"""

# Total expenditure vs avg math 4 score

h = sns.jointplot("TOTAL_EXPENDITURE", "AVG_MATH_4_SCORE", data=df, kind="reg");
h.ax_joint.set_xlabel('TOTAL_EXPENDITURE');

# Average expenditure vs avg math 4 score

h = sns.jointplot("average_expenditure", "AVG_MATH_4_SCORE", data=df, kind="reg")
h.ax_joint.set_xlabel('AVERAGE_EXPENDITURE');

# total revenue vs avg math 4 score

h = sns.jointplot("TOTAL_REVENUE", "AVG_MATH_4_SCORE", data=df, kind="reg");
h.ax_joint.set_xlabel('TOTAL_REVENUE');

# average revenue vs avg math 4 score

h = sns.jointplot("average_revenue", "AVG_MATH_4_SCORE", data=df, kind="reg")
h.ax_joint.set_xlabel('AVERAGE_REVENUE');

"""**Parallel coordinates plot**"""

df = pd.read_csv('drive/MyDrive/Colab Notebooks/AML/Project/archive/states_all.csv')

del df['PRIMARY_KEY']

df.set_index('STATE')
df.head()

scores = ['AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE']

scores_df = df[scores].dropna().copy()
print (scores_df.isna().sum())

X = StandardScaler().fit_transform(scores_df)
X

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X)

centers = model.cluster_centers_
centers

def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')

    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas data frame for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P

from pandas.plotting import parallel_coordinates
from itertools import cycle, islice

def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')

P = pd_centers(scores, centers)
P

parallel_plot(P)

"""Since the lines don't cross here, we can make a conclusion that the states that have low scores have similarly low scores across all the scoring criteria.

# Models

## Linear Regression
"""

df = pd.read_csv("us_education_final.csv")

te_features = ['STATE']
num_features = ['YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', \
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', \
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', \
                'EXPENDITURE_PER_ENROLLMENT']
target = df['TOTAL_SCORE']

df = df[num_features+te_features]
df.head()

X_dev, X_test, y_dev, y_test = train_test_split(df, target, test_size=0.2, \
                                                random_state=42)

preprocess = make_column_transformer((StandardScaler(),num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, LinearRegression())
linear_model = pipe[-1]

cv_results = cross_validate(linear_model, X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
best_model

coefs = cv_results['estimator'][best_model].coef_

fig = plt.figure(figsize=(12,9))
sns.barplot(y=df.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

"""Observation:

Linear regression is highly unstable depending on the features we select. If we include the avg math and reading scores in our features, they can almost perfectly predict total scores and our MSE will go to 0 and R^2 to 1. The importance of all other features drop to 0. On the flip side, if we exclude them in our feature set, the model performance becomes poor. The dominant features in this case are year, state and expenditure per enrollment.

The MSE and R^2 are 560 and 0.16 on test set.

The MSE and R^2 are 668 and 0.12 on cross validation on average.

We think the reason why linear regression performs poorly is due to the fact that features in our dataset are somewhat correlated despite our effort to drop the most correlated ones, and we know multicolinearity can yield solutions that are wildly varying and possibly numerically unstable.

### Additional Analysis

#### Including math 8 in the features to predict reading 8 instead of total score
"""

df = pd.read_csv('us_education_final.csv')

target = df['AVG_READING_8_SCORE']

te_features = ['STATE']
num_features = ['YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', \
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', \
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', \
                'EXPENDITURE_PER_ENROLLMENT','AVG_MATH_8_SCORE']

df = df[num_features+te_features]

X_dev, X_test, y_dev, y_test = train_test_split(df, target, test_size=0.2, \
                                                random_state=42)

preprocess = make_column_transformer((StandardScaler(),num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, LinearRegression())

cv_results = cross_validate(pipe[-1], X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
best_model

y_pred = cv_results['estimator'][best_model].predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

"""#### Including reading 4 in the features to predict math 4 instead of total score"""

df = pd.read_csv('us_education_final.csv')

target = df['AVG_MATH_4_SCORE']

te_features = ['STATE']
num_features = ['YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', \
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', \
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', \
                'EXPENDITURE_PER_ENROLLMENT','AVG_READING_4_SCORE']

df = df[num_features+te_features]

X_dev, X_test, y_dev, y_test = train_test_split(df, target, test_size=0.2, \
                                                random_state=42)

preprocess = make_column_transformer((StandardScaler(),num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, LinearRegression())

cv_results = cross_validate(pipe[-1], X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
best_model

y_pred = cv_results['estimator'][best_model].predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

"""Observation: 

Instead of predicting total score, we try to predict reading 8 score using math 8 score in the second analysis. Both MSE and R^2 on the test set have improved a lot to 9 and 0.75. In addition, we also tried to predict math 4 score using reading 4 score, and MSE and R^2 improved to 18 and 0.62 in this case.

## Ridge Regression
"""

df = pd.read_csv("us_education_final.csv")

y = df['TOTAL_SCORE']

te_feature = ['STATE']

num_features = ['YEAR', 'ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'LOCAL_REVENUE', 'OTHER_EXPENDITURE', 
         'CAPITAL_OUTLAY_EXPENDITURE', 'GRADES_PK_G', 'EXPENDITURE_PER_ENROLLMENT']

X = df[num_features+te_feature]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = make_column_transformer((StandardScaler(), num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, Ridge())
ridge_model = pipe[-1]

cv_results = cross_validate(ridge_model, X_dev, y_dev, cv=10, return_estimator=True,
                            scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

# pick the model with the highest R^2
best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=X.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

# Tuning 
from numpy import arange
from sklearn.model_selection import GridSearchCV
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

search = GridSearchCV(ridge_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print("best score", results.best_score_)
print("best params", results.best_params_)

y_pred_opt = search.predict(X_test)
print("MSE", mean_squared_error(y_test,y_pred_opt))

ridge_optimal_model = Ridge(alpha = 0.99)
ridge_optimal_model.fit(X_dev,y_dev)

feature_importance = pd.Series(index = X_dev.columns, data = ridge_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (20,8))

"""Summary for ridge linear regression model

In our primary analysis where the target is the total score, which is the sum of 4th and 8th grade math and reading scores, Ridge linear regression model performs badly, as the best model after tuning on alpha gives MSE of 561. Feature importance figure shows that the most significant features are expenditure per enrollment, state, and year. However, this result doesn't make sense because this suggests that the more resources spent on students, the worse they perform. Overall, Ridge performs poorly, and we choose not to proceed with it.

### Additional analysis on math and reading scores predicability

#### Use 4th grade math score to predict 4th grade reading score
"""

y = df['AVG_READING_4_SCORE']
te_feature = ['STATE']
num_features = ['YEAR', 'ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'LOCAL_REVENUE', 'OTHER_EXPENDITURE', 
         'CAPITAL_OUTLAY_EXPENDITURE', 'GRADES_PK_G', 'EXPENDITURE_PER_ENROLLMENT', 'AVG_MATH_4_SCORE']

X = df[num_features+te_feature]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = make_column_transformer((StandardScaler(), num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, Ridge())
ridge_model = pipe[-1]

cv_results = cross_validate(ridge_model, X_dev, y_dev, cv=10, return_estimator=True,
                            scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=X.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

# Tuning 
from numpy import arange
from sklearn.model_selection import GridSearchCV
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

search = GridSearchCV(ridge_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print("best score", results.best_score_)
print("best alpha", results.best_params_['alpha'])

y_pred_opt = search.predict(X_test)
print("MSE", mean_squared_error(y_test,y_pred_opt))

ridge_optimal_model = Ridge(alpha = results.best_params_['alpha'])
ridge_optimal_model.fit(X_dev,y_dev)

feature_importance = pd.Series(index = X_dev.columns, data = ridge_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (20,8))

"""#### Use 8th grade math score to predict 8th grade reading score"""

y = df['AVG_READING_8_SCORE']
te_feature = ['STATE']
num_features = ['YEAR', 'ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'LOCAL_REVENUE', 'OTHER_EXPENDITURE', 
         'CAPITAL_OUTLAY_EXPENDITURE', 'GRADES_PK_G', 'EXPENDITURE_PER_ENROLLMENT', 'AVG_MATH_8_SCORE']

X = df[num_features+te_feature]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = make_column_transformer((StandardScaler(), num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, Ridge())
ridge_model = pipe[-1]

cv_results = cross_validate(ridge_model, X_dev, y_dev, cv=10, return_estimator=True,
                            scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=X.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

# Tuning 
from numpy import arange
from sklearn.model_selection import GridSearchCV
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

search = GridSearchCV(ridge_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print("best score", results.best_score_)
print("best alpha", results.best_params_['alpha'])

y_pred_opt = search.predict(X_test)
print("MSE", mean_squared_error(y_test,y_pred_opt))

ridge_optimal_model = Ridge(alpha = results.best_params_['alpha'])
ridge_optimal_model.fit(X_dev,y_dev)

feature_importance = pd.Series(index = X_dev.columns, data = ridge_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (20,8))

"""#### Using 4th grade math score to predict 8th grade math score"""

y = df['AVG_MATH_8_SCORE']
te_feature = ['STATE']
num_features = ['YEAR', 'ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'LOCAL_REVENUE', 'OTHER_EXPENDITURE', 
         'CAPITAL_OUTLAY_EXPENDITURE', 'GRADES_PK_G', 'EXPENDITURE_PER_ENROLLMENT', 'AVG_MATH_4_SCORE']

X = df[num_features+te_feature]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = make_column_transformer((StandardScaler(), num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, Ridge())
ridge_model = pipe[-1]

cv_results = cross_validate(ridge_model, X_dev, y_dev, cv=10, return_estimator=True,
                            scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=X.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

# Tuning 
from numpy import arange
from sklearn.model_selection import GridSearchCV
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

search = GridSearchCV(ridge_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print("best score", results.best_score_)
print("best alpha", results.best_params_['alpha'])

y_pred_opt = search.predict(X_test)
print("MSE", mean_squared_error(y_test,y_pred_opt))

ridge_optimal_model = Ridge(alpha = results.best_params_['alpha'])
ridge_optimal_model.fit(X_dev,y_dev)

feature_importance = pd.Series(index = X_dev.columns, data = ridge_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (20,8))

"""#### Using 4th grade reading score to predict 8th grade reading score"""

y = df['AVG_READING_8_SCORE']
te_feature = ['STATE']
num_features = ['YEAR', 'ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'LOCAL_REVENUE', 'OTHER_EXPENDITURE', 
         'CAPITAL_OUTLAY_EXPENDITURE', 'GRADES_PK_G', 'EXPENDITURE_PER_ENROLLMENT', 'AVG_READING_4_SCORE']

X = df[num_features+te_feature]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = make_column_transformer((StandardScaler(), num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, Ridge())
ridge_model = pipe[-1]

cv_results = cross_validate(ridge_model, X_dev, y_dev, cv=10, return_estimator=True,
                            scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=X.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

# Tuning 
from numpy import arange
from sklearn.model_selection import GridSearchCV
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

search = GridSearchCV(ridge_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print("best score", results.best_score_)
print("best alpha", results.best_params_['alpha'])

y_pred_opt = search.predict(X_test)
print("MSE", mean_squared_error(y_test,y_pred_opt))

ridge_optimal_model = Ridge(alpha = results.best_params_['alpha'])
ridge_optimal_model.fit(X_dev,y_dev)

feature_importance = pd.Series(index = X_dev.columns, data = ridge_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (20,8))



"""## Lasso Regression"""

df = pd.read_csv("us_education_final.csv")

X = df.drop(["TOTAL_SCORE", 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE', 'AVG_READING_8_SCORE'], axis = 1)
y = df["TOTAL_SCORE"]

from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate

# Training using cross validation
lasso_model = make_pipeline(StandardScaler(with_mean=False), Lasso())
scores = cross_validate(lasso_model, X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])
print(" Mean R^2 score after cross-validation: ", scores['test_r2'].mean())
print("Mean MSE after cross-validation: ", -scores['test_neg_mean_squared_error'].mean())

best_model = np.argmax(scores['test_r2'])
y_pred = scores['estimator'][best_model].predict(X_test)

coefs = scores['estimator'][best_model][-1].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=df[['STATE','YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', \
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', \
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', \
                'EXPENDITURE_PER_ENROLLMENT']].columns, x=coefs, orient='h')

# The mean squared error on test set using the best model
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination on test set using the best model
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

"""Observation:

Lasso model's performance is looked at deeply in order to check if feature selection could improve the performance for the specific task and dataset at hand. Results help us conclude that the performance received is poor (just like any other linear model in this situation).

Feature importance map says that federal revenue and total revenue are the primary contributors to the TOTAL_SCORE, but they have opposite signs. From an intuition perspective, both of them should be positively correlated to the TOTAL_SCORE and increase in either one of them should positively impact the performance of the children in exams (as they would be given more facilities). Hence, we should be skeptical in pursuing the lasso model.

The MSE and R^2 are 582.72 and 0.13 on test set.

The MSE and R^2 are 682.62 and 0.1 on cross validation on average.

### Additional Analysis: Training on AVG_MATH_8_SCORE to predict AVG_READING_8_SCORE
"""

X = df.drop(["TOTAL_SCORE",'AVG_MATH_4_SCORE', 'AVG_READING_4_SCORE', "AVG_READING_8_SCORE"], axis = 1)
y = df["AVG_READING_8_SCORE"]

from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate

# Training using cross validation
lasso_model = make_pipeline(StandardScaler(with_mean=False), Lasso())
scores = cross_validate(lasso_model, X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])
print(" Mean R^2 score after cross-validation: ", scores['test_r2'].mean())
print("Mean MSE after cross-validation: ", -scores['test_neg_mean_squared_error'].mean())

best_model = np.argmax(scores['test_r2'])
y_pred = scores['estimator'][best_model].predict(X_test)

coefs = scores['estimator'][best_model][-1].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=df[['STATE','YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', \
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', \
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', \
                'EXPENDITURE_PER_ENROLLMENT', "AVG_MATH_8_SCORE"]].columns, x=coefs, orient='h')

# The mean squared error on test set using the best model
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination on test set using the best model
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

"""Observation:

When it comes to predicting AVG_READING_8_SCORE from AVG_MATH_8_SCORE, the model performs better. This analysis was primarily done in order to check the feature importance map and see which features contribute to AVG_READING_8_SCORE. To our surprise, EXPENDITURE_PER_ENROLLMENT column contributes 100% to AVG_READING_8_SCORE and no other feature is selected by the lasso model.

The MSE and R^2 are 10.62 and 0.69 on test set.

The MSE and R^2 are 11.90 and 0.69 on cross validation on average.
"""



"""### Additional Analysis: Training on AVG_MATH_4_SCORE to predict AVG_READING_4_SCORE"""

X = df.drop(["TOTAL_SCORE",'AVG_MATH_8_SCORE', 'AVG_READING_8_SCORE', "AVG_READING_4_SCORE"], axis = 1)
y = df["AVG_READING_4_SCORE"]

from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate

# Training using cross validation
lasso_model = make_pipeline(StandardScaler(with_mean=False), Lasso())
scores = cross_validate(lasso_model, X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])
print(" Mean R^2 score after cross-validation: ", scores['test_r2'].mean())
print("Mean MSE after cross-validation: ", -scores['test_neg_mean_squared_error'].mean())

best_model = np.argmax(scores['test_r2'])
y_pred = scores['estimator'][best_model].predict(X_test)

coefs = scores['estimator'][best_model][-1].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=df[['STATE','YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', \
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', \
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', \
                'EXPENDITURE_PER_ENROLLMENT', "AVG_MATH_8_SCORE"]].columns, x=coefs, orient='h')

# The mean squared error on test set using the best model
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination on test set using the best model
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

"""Observation:

When it comes to predicting AVG_READING_4_SCORE from AVG_MATH_4_SCORE, the model still performs better compared to the original model that was trained without any of the average score values as features. This analysis was primarily done in order to check the feature importance map and see which features contribute to AVG_READING_4_SCORE. To our surprise, EXPENDITURE_PER_ENROLLMENT column contributes 100% to AVG_READING_4_SCORE and no other feature is selected by the lasso model (similar to the trend observed while predicting AVG_READING_8_SCORE from AVG_MATH_8_SCORE).

The MSE and R^2 are 18.36 and 0.63 on test set.

The MSE and R^2 are 15.92 and 0.59 on cross validation on average.

## Elastic Net Regression
"""

df = pd.read_csv("us_education_final.csv")

te_features = ['STATE']
num_features = ['YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', 
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', 
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', 
                'EXPENDITURE_PER_ENROLLMENT']
target = df['TOTAL_SCORE']

df = df[num_features+te_features]
df

sns.histplot(target)

X_dev, X_test, y_dev, y_test = train_test_split(df, target, test_size=0.2,random_state=42)

preprocess = make_column_transformer((StandardScaler(),num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, ElasticNet())
elastic_net_model = pipe[-1]

#Elastic Net 
cv_results = cross_validate(elastic_net_model, X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=df.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

#Tuning 
from numpy import arange
from sklearn.model_selection import GridSearchCV
grid = dict()
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = arange(0, 1, 0.01)

search = GridSearchCV(elastic_net_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print(results.best_score_)
print(results.best_params_)

y_pred_opt = search.predict(X_test)
print(mean_squared_error(y_test,y_pred_opt))

elastic_net_optimal_model = ElasticNet(alpha = 10.0, l1_ratio = 0.0)
elastic_net_optimal_model.fit(X_dev,y_dev)

elastic_net_optimal_model.coef_

feature_importance = pd.Series(index = X_edu.columns, data = elastic_net_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (20,8))

"""### Additional Analysis

#### Using Math 8 to predict Reading 8
"""

df = pd.read_csv('us_education_final.csv')
target = df['AVG_READING_8_SCORE']

te_features = ['STATE']
num_features = ['YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', 
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', 
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', 
                'EXPENDITURE_PER_ENROLLMENT','AVG_MATH_8_SCORE']

df = df[num_features+te_features]

X_dev, X_test, y_dev, y_test = train_test_split(df, target, test_size=0.2,random_state=42) 
preprocess = make_column_transformer((StandardScaler(),num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, ElasticNet())

cv_results = cross_validate(pipe[-1], X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=df.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

#Tuning
search = GridSearchCV(elastic_net_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print(results.best_score_)
print(results.best_params_)

elastic_net_optimal_model = ElasticNet(alpha = 0.1, l1_ratio = 0.0)
elastic_net_optimal_model.fit(X_dev,y_dev)

y_pred_opt = search.predict(X_test)
print(mean_squared_error(y_test,y_pred_opt))

elastic_net_optimal_model.coef_

feature_importance = pd.Series(index = X_dev.columns, data = elastic_net_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))

"""#### Using reading 4 to predict math 4 """

df = pd.read_csv('us_education_final.csv')

target = df['AVG_MATH_4_SCORE']

te_features = ['STATE']
num_features = ['YEAR','ENROLL','TOTAL_REVENUE', 'FEDERAL_REVENUE', 
                'LOCAL_REVENUE' ,'OTHER_EXPENDITURE', 
                'CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G', 
                'EXPENDITURE_PER_ENROLLMENT','AVG_READING_4_SCORE']

df = df[num_features+te_features]

X_dev, X_test, y_dev, y_test = train_test_split(df, target, test_size=0.2,random_state=42) 
preprocess = make_column_transformer((StandardScaler(),num_features),remainder='passthrough')
pipe = make_pipeline(preprocess, ElasticNet())

cv_results = cross_validate(pipe[-1], X_dev, y_dev, cv=10, return_estimator=True,scoring =['r2','neg_mean_squared_error'])

print(cv_results['test_r2'].mean())
print(-cv_results['test_neg_mean_squared_error'].mean())

best_model = np.argmax(cv_results['test_r2'])
coefs = cv_results['estimator'][best_model].coef_
fig = plt.figure(figsize=(12,9))
sns.barplot(y=df.columns, x=coefs, orient='h')

y_pred = cv_results['estimator'][best_model].predict(X_test)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.5f" % r2_score(y_test, y_pred))

#Tuning
search = GridSearchCV(elastic_net_model, grid, cv=10,scoring='r2', n_jobs=-1)
results = search.fit(X_dev, y_dev)
print(results.best_score_)
print(results.best_params_)

elastic_net_optimal_model = ElasticNet(alpha = 0.1, l1_ratio = 0.08)
elastic_net_optimal_model.fit(X_dev,y_dev)

y_pred_opt = search.predict(X_test)
print(mean_squared_error(y_test,y_pred_opt))

elastic_net_optimal_model.coef_

feature_importance = pd.Series(index = X_dev.columns, data = elastic_net_optimal_model.coef_)

n_selected_features = (feature_importance!=0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))

"""## Random Forest"""

df = pd.read_csv("us_education_final.csv")

X = df.drop(['TOTAL_SCORE', 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE', 'AVG_READING_4_SCORE', 'AVG_READING_8_SCORE'], axis = 1)
y = df['TOTAL_SCORE']

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_dev.head()

gsc = GridSearchCV(estimator=RandomForestRegressor(),
                   param_grid={'max_depth': (3, 6, 9, 12, 15),
                               'n_estimators': (10, 50, 100, 500, 1000),},
                   cv=10, scoring='r2', verbose=0, n_jobs=-1)
grid_result = gsc.fit(X_dev, y_dev)
best_params = grid_result.best_params_

best_params

random_forest_model = RandomForestRegressor(max_depth=best_params["max_depth"], 
                                            n_estimators=best_params["n_estimators"],
                                            random_state=False, verbose=False)

scores = cross_validate(random_forest_model, X_dev, y_dev, cv=10, 
                        scoring=['neg_mean_squared_error', 'r2'])

scores['test_neg_mean_squared_error'].mean(), scores['test_r2'].mean()

random_forest_model = RandomForestRegressor(max_depth=best_params["max_depth"],
                                            n_estimators=best_params["n_estimators"]).fit(X_dev, y_dev)

mean_squared_error(y_test, random_forest_model.predict(X_test))

r2_score(y_test, random_forest_model.predict(X_test))

pd.DataFrame({"predicted": random_forest_model.predict(X_test), "target":y_test})

import seaborn as sns
import matplotlib.pyplot as plt
feature_names = X_dev.columns
feat_imps = zip(feature_names, random_forest_model.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x:x[1]!=0, feat_imps)),
key = lambda x:x[1], reverse = True)))
ax = sns.barplot(x=list(feats), y=list(imps))
ax.tick_params(axis='x', rotation=90)
plt.title("Feature Importance")
plt.show()

"""Random Forest performs well when we try to predict the total score. After doing a 10 fold grid search cross validation to select the best parameters, we ended up chosing max_depth being 15 and n_estimators being 1000 as the best parameters for the model. 

Using this model:

The MSE and R^2 are 81.99 and 0.88 on test set.

The MSE and R^2 are 86.04 and 0.89 on cross validation on average.

The top three most important features are STATE, followed by ENROLL and LOCAL_REVENUE.

### Additional analysis on math and reading scores predicability

Use 4th grade reading score to predict 4th grade math score
"""

X = df.drop(['TOTAL_SCORE', 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE', 'AVG_READING_8_SCORE'], axis = 1)
y = df['AVG_MATH_4_SCORE']

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gsc = GridSearchCV(estimator=RandomForestRegressor(),
                   param_grid={'max_depth': (3, 6, 9, 12, 15),
                               'n_estimators': (10, 50, 100, 500, 1000),},
                   cv=10, scoring='r2', verbose=0, n_jobs=-1)
grid_result = gsc.fit(X_dev, y_dev)
best_params = grid_result.best_params_

best_params

random_forest_model = RandomForestRegressor(max_depth=best_params["max_depth"], 
                                            n_estimators=best_params["n_estimators"],
                                            random_state=False, verbose=False)
scores = cross_validate(random_forest_model, X_dev, y_dev, cv=10, 
                        scoring=['neg_mean_squared_error', 'r2'])
scores['test_neg_mean_squared_error'].mean(), scores['test_r2'].mean()

random_forest_model = RandomForestRegressor(max_depth=best_params["max_depth"],
                                            n_estimators=best_params["n_estimators"]).fit(X_dev, y_dev)    
mean_squared_error(y_test, random_forest_model.predict(X_test))

r2_score(y_test, random_forest_model.predict(X_test))

pd.DataFrame({"predicted": random_forest_model.predict(X_test), "target":y_test})

import seaborn as sns
import matplotlib.pyplot as plt
feature_names = X_dev.columns
feat_imps = zip(feature_names, random_forest_model.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x:x[1]!=0, feat_imps)),
key = lambda x:x[1], reverse = True)))
ax = sns.barplot(x=list(feats), y=list(imps))
ax.tick_params(axis='x', rotation=90)
plt.title("Feature Importance")
plt.show()

"""Random Forest performs quite good when we try to predict 4th grade math score after including 4th grade reading score. After doing a 10 fold grid search cross validation to select the best parameters, we ended up chosing max_depth being 12 and n_estimators being 500 as the best parameters for the model. 

Using this model:

The MSE and R^2 are 10.77 and 0.77 on test set.

The MSE and R^2 are 9.97 and 0.81 on cross validation on average.

The most important feature is AVERSGE_READING_4_SCORE, followed by YEAR.

Use 8th grade math score to predict 8th grade reading score
"""

X = df.drop(['TOTAL_SCORE', 'AVG_READING_4_SCORE', 'AVG_MATH_4_SCORE', 'AVG_READING_8_SCORE'], axis = 1)
y = df['AVG_READING_8_SCORE']

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gsc = GridSearchCV(estimator=RandomForestRegressor(),
                   param_grid={'max_depth': (3, 6, 9, 12, 15),
                               'n_estimators': (10, 50, 100, 500, 1000),},
                   cv=10, scoring='r2', verbose=0, n_jobs=-1)
grid_result = gsc.fit(X_dev, y_dev)
best_params = grid_result.best_params_

best_params

random_forest_model = RandomForestRegressor(max_depth=best_params["max_depth"], 
                                            n_estimators=best_params["n_estimators"],
                                            random_state=False, verbose=False)
scores = cross_validate(random_forest_model, X_dev, y_dev, cv=10, 
                        scoring=['neg_mean_squared_error', 'r2'])
scores['test_neg_mean_squared_error'].mean(), scores['test_r2'].mean()

random_forest_model = RandomForestRegressor(max_depth=best_params["max_depth"],
                                            n_estimators=best_params["n_estimators"]).fit(X_dev, y_dev)    
mean_squared_error(y_test, random_forest_model.predict(X_test))

r2_score(y_test, random_forest_model.predict(X_test))

pd.DataFrame({"predicted": random_forest_model.predict(X_test), "target":y_test})

feature_names = X_dev.columns
feat_imps = zip(feature_names, random_forest_model.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x:x[1]!=0, feat_imps)),
key = lambda x:x[1], reverse = True)))
ax = sns.barplot(x=list(feats), y=list(imps))
ax.tick_params(axis='x', rotation=90)
plt.title("Feature Importance")
plt.show()

"""Random Forest performs well when we try to predict 8th grade reading score after including 8th grade math score. After doing a 10 fold grid search cross validation to select the best parameters, we ended up chosing max_depth being 15 and n_estimators being 1000 as the best parameters for the model.

Using this model:

The MSE and R^2 are 2.87 and 0.92 on test set.

The MSE and R^2 are 3.20 and 0.92 on cross validation on average.

The most important feature is AVERSGE_MATH_8_SCORE, followed by STATE and ENROLL.

## XGBoost Regressor
"""

import xgboost

X = df.drop(["TOTAL_SCORE", 'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE', 'AVG_READING_8_SCORE'], axis = 1)
y = df["TOTAL_SCORE"]

from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

"""Bayesian Hyperparameter tuning on a search space for the XGBoost Regressor using HYPEROPT"""

from hyperopt import hp, STATUS_OK

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,10),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    model = XGBRegressor(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_dev, y_dev), ( X_test, y_test)]
    
    model.fit(X_dev, y_dev,
            eval_set=evaluation, eval_metric=["rmse"],
            verbose=False)
    

    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    print ("SCORE:", loss)
    return {'loss': loss, 'status': STATUS_OK, 'model':model }

from hyperopt import Trials, fmin, tpe
from sklearn.metrics import mean_squared_error
trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print(best_hyperparams)

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, RepeatedKFold
from numpy import absolute
from sklearn.metrics import r2_score, mean_squared_error

# Training using cross validation with the best hyperparameters
xgb_model = XGBRegressor(random_state = 42)
cv = RepeatedKFold(n_splits=10)
scores = cross_validate(xgb_model, X, y, scoring=["r2", 'neg_mean_squared_error'], cv=cv, return_estimator=True)
print(" Mean R^2 score after cross-validation: ", scores['test_r2'].mean())
print("Mean MSE after cross-validation: ", -scores['test_neg_mean_squared_error'].mean())

best_xgb_model = XGBRegressor(colsample_bytree = best_hyperparams['colsample_bytree'],
             gamma = best_hyperparams['gamma'],
             max_depth = int(best_hyperparams['max_depth']),
             min_child_weight = best_hyperparams['min_child_weight'],
             reg_alpha = best_hyperparams['reg_alpha'] ,
             reg_lambda = best_hyperparams['reg_lambda'],
             random_state = 42)

best_xgb_model.fit(X_dev, y_dev)
y_pred = best_xgb_model.predict(X_test)

# The mean squared error
print("Mean squared error on test set: %.2f" % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 
print("Coefficient of determination on test set: %.2f" % r2_score(y_test, y_pred))

# Feature importance plot
import seaborn as sns
import matplotlib.pyplot as plt
feature_names = X_dev.columns
feat_imps = zip(feature_names, best_xgb_model.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x:x[1]!=0, feat_imps)),
key = lambda x:x[1], reverse = True)))
ax = sns.barplot(x=list(feats), y=list(imps))
ax.tick_params(axis='x', rotation=90)
plt.title("Feature Importance")
plt.show()

"""Observation:

We can see here that the XGBoost model is able to understand the dependencies between the features and weight them appropriately to get good performance as compared to other linear models. We have performed Bayesian Hyperparamater Optimisation using the hyperopt package in order to arrive at the most optimal hyperparameter set and that model is evaluated on the test set in order to receive the performance values.

The feature importance map looks good as all features are contributing to the final result in some way or the other with the STATE feature contributing the most

The MSE and R^2 are 49.5 and 0.93 on test set.

The MSE and R^2 are 118.42 and 0.837 on cross validation on average.

### Additional Analysis: Training on AVG_MATH_8_SCORE to predict AVG_READING_8_SCORE
"""

X = df.drop(["TOTAL_SCORE",'AVG_MATH_4_SCORE', 'AVG_READING_4_SCORE', "AVG_READING_8_SCORE"], axis = 1)
y = df["AVG_READING_8_SCORE"]

from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from hyperopt import hp, STATUS_OK

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,10),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    model = XGBRegressor(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_dev, y_dev), ( X_test, y_test)]
    
    model.fit(X_dev, y_dev,
            eval_set=evaluation, eval_metric=["rmse"],
            verbose=False)
    

    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    print ("SCORE:", loss)
    return {'loss': loss, 'status': STATUS_OK, 'model':model }

from hyperopt import Trials, fmin, tpe
from sklearn.metrics import mean_squared_error
trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print(best_hyperparams)

# Training using cross validation with the best hyperparameters
xgb_model = XGBRegressor(random_state = 42)
cv = RepeatedKFold(n_splits=10)
scores = cross_validate(xgb_model, X, y, scoring=["r2", 'neg_mean_squared_error'], cv=cv, return_estimator=True)
print(" Mean R^2 score after cross-validation: ", scores['test_r2'].mean())
print("Mean MSE after cross-validation: ", -scores['test_neg_mean_squared_error'].mean())

best_xgb_model = XGBRegressor(colsample_bytree = best_hyperparams['colsample_bytree'],
             gamma = best_hyperparams['gamma'],
             max_depth = int(best_hyperparams['max_depth']),
             min_child_weight = best_hyperparams['min_child_weight'],
             reg_alpha = best_hyperparams['reg_alpha'] ,
             reg_lambda = best_hyperparams['reg_lambda'],
             random_state = 42)

best_xgb_model.fit(X_dev, y_dev)
y_pred = best_xgb_model.predict(X_test)

# The mean squared error
print("Mean squared error on test set: %.2f" % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 
print("Coefficient of determination on test set: %.2f" % r2_score(y_test, y_pred))

# Feature importance plot
import seaborn as sns
import matplotlib.pyplot as plt
feature_names = X_dev.columns
feat_imps = zip(feature_names, best_xgb_model.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x:x[1]!=0, feat_imps)),
key = lambda x:x[1], reverse = True)))
ax = sns.barplot(x=list(feats), y=list(imps))
ax.tick_params(axis='x', rotation=90)
plt.title("Feature Importance")
plt.show()

"""Observation:

When it comes to predicting AVG_READING_8_SCORE from AVG_MATH_8_SCORE, the model performs a bit less. This analysis was primarily done in order to check the feature importance map and see ehich features contribute to AVG_READING_8_SCORE. It can be seen that AVG_MATH_8_SCORE is the feature that is correlated the most followed by STATE and the other feature contributes relatively less.

This analysis helps us reaasure ourselves that dropping the AVG scores to predict the TOTAL_SCORE was indeed the right choice because in the case where we don't drop them, the tatget variable would be highly correlated to them and the model would hence overfit and give us a perfect 1 R2 score (which we generally prefer to avoid)

The MSE and R^2 are 3.29 and 0.90 on test set.

The MSE and R^2 are 3.56 and 0.90 on cross validation on average.
"""