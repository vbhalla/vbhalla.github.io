
---
layout: post
title: Healthy Data Wrangling
date: '2019-10-28 12:07:25 +0000'
categories:
  - data
published: false
---

The past year I have really come to appreciate the rollercoaster of health both personally and how it relates to the people we care about.  It has been a sea of emotions: daunting, confusing, intimidating, saddening, and actually quite empowering.  That is a story for another day.  Building a data-driven foundation culture into an organization: infrastructure, experimentation, and consistently monitoring, mining, extrapolating for insights in order to make large calculated bets has been a huge part of what I love about my jobs.  I kinda wondered, why can't I do that same thing; swapping out the business ofr my body :-)  I'd like to share a subset of a project, it was humbling how much more complicated, noisy, and just plain messy some of the data contained in these platforms currently are.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import fitbit_batch_update_script

from importlib import reload as reload
```


```python
# put this into 1 function for fitbit batch script
import glob
import os
import re
list_of_files = glob.glob('./master_data_backups/*.f') 
latest_file = max(list_of_files, key=os.path.getctime)
last_run_date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", latest_file).group(1)
print ("Latest File: ",latest_file)

from datetime import date,timedelta, datetime
last_run_date = datetime.strptime(last_run_date, '%Y-%m-%d').date()
print("Last Run Date:",last_run_date)
df_master = pd.read_feather(latest_file)
```

    Latest File:  ./master_data_backups/df_master_2019-10-24.f



```python
def analysis_calcs(df_master):
    #gives day of the week column
    df_master['date'] = pd.to_datetime(df_master['date'])
    df_master['day'] = df_master['date'].dt.day_name()
    #df_master['day'] = pd.to_datetime(df_master['date']).dt.day_name()
    # sum calories
    df_master['hr_total_calories'] = df_master[['hr_OutofRange_caloriesOut', 'hr_FatBurn_caloriesOut', 'hr_Cardio_caloriesOut', 'hr_Peak_caloriesOut']].sum(axis=1)
    #del columns
    del_cols = ['hr_OutofRange_max','hr_FatBurn_max','hr_Cardio_max','hr_Peak_max','hr_OutofRange_min','hr_FatBurn_min','hr_Cardio_min','hr_Peak_min']
    df_master.drop(columns=del_cols,inplace=True)
    
    return df_master

def sleep_calcs(df):
    
    df['calc_deep_sleep_perc']=df['totalSleep_deep_mins']/df['totalSleepTimeInBed']
    df['calc_light_sleep_perc']=df['totalSleep_light_mins']/df['totalSleepTimeInBed']
    df['calc_rem_sleep_perc']=df['totalSleep_rem_mins']/df['totalSleepTimeInBed']
        
    return df

#see if this works
def mins_to_hours_calc(df, cols):
    for i in cols:
        #df_monthly[i] = pd.to_datetime(df_monthly[i], unit='m').dt.strftime('%H:%M')
        df[i] = pd.to_datetime(df[i], unit='m').dt.strftime('%H:%M')
    
    return df
```


```python
df_master = analysis_calcs(df_master)

#convert to hours after converting to percentages
df_master = sleep_calcs(df_master)

```


```python
cols_agg = ['date', 'calories', 'steps', 'dist', 'mins_sedant','hr_total_calories', 'mins_active_light','calc_active_mins', 'totalSleepMinutesAsleep','totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed','sleep_start','sleep_minutesAsleep'] 
cols_avg = ['date', 'calories', 'steps', 'dist', 'mins_sedant','hr_total_calories', 'mins_active_light','calc_active_mins', 'totalSleepMinutesAsleep','totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed','sleep_start','resting_hr','sleep_efficiency','sleep_minutesAsleep']
```

## Sums - Weekday


```python
df_monthly = df_master[~df_master.day.isin(['Saturday','Sunday'])].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('sum')
df_monthly[['calories', 'steps', 'dist', 'mins_sedant', 'hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed']][df_monthly.index > '2017-07-31'].tail(10)
df_monthly.reset_index(inplace=True)
df_monthly[['date','calories', 'steps', 'dist']][df_monthly.date>'2017-11-01'].plot(subplots=True,x='date',figsize=(12,9), sharex=True, legend=True,title='Monthly Cumulative Calories, Steps, Distances')
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x1a87a95908>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x1a87cc04e0>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x1a87ad4860>],
          dtype=object)




![png](support_post_health_files/support_post_health_6_1.png)


## Monthly Averages 


```python
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2017-11-01')].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('mean')
df_monthly.reset_index(inplace=True)
df_monthly['date']=df_monthly['date'].dt.strftime('%Y-%m-%d')

g=sns.catplot(x="date", y="resting_hr", kind="bar", data=df_monthly[['date','calories','resting_hr']], legend_out=False)
g.fig.set_size_inches(15,8)


(g.set_axis_labels("", "Resting HR")
   .set(ylim=(55, 72))
   )

plt.title("Avg Monthly Resting Heart Rate excl. Weekends")
plt.xticks(rotation=45) 
# Add tight_layout to ensure the labels don't get cut off
plt.tight_layout()
plt.show()
```


![png](support_post_health_files/support_post_health_8_0.png)



```python
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2018-01-01')&(df_master.totalSleepMinutesAsleep>0)&(df_master.totalSleep_light_mins>0)].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('mean')
df_monthly = sleep_calcs(df_monthly)
corr_cols = ['calories', 'steps', 'mins_sedant', 'resting_hr','hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins','totalSleepTimeInBed', 'sleep_efficiency']
df_monthly.reset_index(inplace=True)
corr_matrix = df_monthly[corr_cols].corr()

# Setup
fig, ax = plt.subplots(figsize=(14, 9))

# vmin and vmax control the range of the colormap
sns.heatmap(corr_matrix, cmap='RdBu', annot=True, fmt='.2f',
           vmin=-1, vmax=1)

plt.title("Correlations of Monthly Data from Jan 1, 2018")

#plt.xticks(rotation=50) 
# Add tight_layout to ensure the labels don't get cut off
plt.tight_layout()
plt.show()
```


![png](support_post_health_files/support_post_health_9_0.png)



```python
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2017-10-01')].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('mean')

fig, ax = plt.subplots(figsize=(14,9), dpi= 80)
ax.vlines(x=df_monthly.index, ymin=0, ymax=df_monthly['calc_active_mins'], color='black', alpha=0.7, linewidth=2)
ax.scatter(x=df_monthly.index, y=df_monthly['calc_active_mins'], s=75, color='black', alpha=0.7)

ax.set_title('Avg Daily Active Mins', fontdict={'size':22})
ax.set_ylabel('calc_active_mins')
ax.set_xticks(df_monthly.index)
ax.set_xticklabels(df_monthly.index.strftime('%Y-%m-%d'), rotation=45)

plt.show()

```


![png](support_post_health_files/support_post_health_10_0.png)



```python
df_monthly.reset_index(inplace=True)
df_monthly.plot(title='Monthly Avg Time Spent Sedentary Per Weekday', x='date', y= ['mins_sedant'],figsize=(14,9))
plt.xlabel('date')
plt.ylabel('minutes')
plt.show()

```


![png](support_post_health_files/support_post_health_11_0.png)


## Dive Into Sleep


```python
df_monthly.columns
```




    Index(['date', 'calories', 'steps', 'dist', 'mins_sedant', 'hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed', 'resting_hr', 'sleep_efficiency', 'sleep_minutesAsleep'], dtype='object')




```python
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2017-07-01')&(df_master.totalSleepMinutesAsleep>0)&(df_master.totalSleep_light_mins>0)].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('mean')
df_monthly.reset_index(inplace=True)
df_monthly['date']=df_monthly['date'].dt.strftime('%Y-%m-%d')
df_monthly[['date','sleep_efficiency','totalSleepMinutesAsleep']].plot(subplots=True,x='date',figsize=(12,7), sharex=False, legend=True,title='Monthly Avg. Sleep Metrics')
plt.legend(loc='lower right')
```




    <matplotlib.legend.Legend at 0x1a847d8630>




![png](support_post_health_files/support_post_health_14_1.png)


## Sleep Efficiency Is A Blackbox Number, Disregarding


```python
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2017-07-01')&((df_master.totalSleepMinutesAsleep>0))&((df_master.totalSleepMinutesAsleep>0))].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('mean')
df_monthly.reset_index(inplace=True)
df_monthly['date']=df_monthly['date'].dt.strftime('%Y-%m-%d')
df_monthly[['date','totalSleepMinutesAsleep', 'totalSleep_deep_mins','totalSleep_rem_mins']].plot(subplots=True,x='date',figsize=(12,7), sharex=False, legend=True,title='Monthly Avg. Sleep Metrics')

```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x1a8b5252b0>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x1a867fa438>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x1a87047208>],
          dtype=object)




![png](support_post_health_files/support_post_health_16_1.png)



```python
len(df_monthly.date)
```




    28




```python
fig = plt.figure(figsize=(16,8))
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2017-07-01')&((df_master.totalSleepMinutesAsleep>0))].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('mean')
df_monthly.reset_index(inplace=True)
df_monthly['date']=df_monthly['date'].dt.strftime('%Y-%m-%d')
ax1 = df_monthly[['date','totalSleepMinutesAsleep']].plot(x='date',kind='bar',figsize=(12,7), sharex=False, legend=False,title='Monthly Avg. Total Sleep Minutes \n 8hrs = 480 mins',rot=45,color='Black')
ax1.set_ylim([290, 485])
ax1.plot([-1, 28], [480, 480], "k--")

plt.show()

```


    <matplotlib.figure.Figure at 0x1a8e480c18>



![png](support_post_health_files/support_post_health_18_1.png)



```python
fig = plt.figure(figsize=(16,8))
df_monthly = df_master[(~df_master.day.isin(['Saturday','Sunday']))&(df_master.date>'2017-07-01')&((df_master.totalSleepMinutesAsleep>0))].groupby(pd.Grouper(key='date', freq='Q'))[cols_avg].agg('mean')
df_monthly.reset_index(inplace=True)
df_monthly['date']=df_monthly['date'].dt.strftime('%Y-%m-%d')
ax1 = df_monthly[['date','totalSleepMinutesAsleep']].plot(x='date',kind='bar',figsize=(12,7), sharex=False, legend=False,title='Quarterly Avg. Total Sleep Minutes \n 8hrs = 480 mins',rot=45,color='Black')
ax1.set_ylim([290, 400])
ax1.plot([-1, 28], [480, 480], "k--")
plt.show()




```


    <matplotlib.figure.Figure at 0x1a86a4ff98>



![png](support_post_health_files/support_post_health_19_1.png)


## Now we got a broad view of things improving, albeit still very far from ideal!
### Let's dig a bit deeper


```python
df_dow = df_master[(df_master.date>'2018-05-01')&(df_master.totalSleepMinutesAsleep>0)&(df_master.totalSleep_light_mins>0)].groupby([pd.Grouper(key='date', freq='M')])[cols_avg].agg('mean')
df_dow.index=df_dow.index.strftime('%Y-%m-%d')
df_dow[['totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins']].plot(kind='bar',stacked=True,rot=45,figsize=(12,8),title='Minutes of Sleep Distributed Across Sleep Cycle')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7ac5c898>




![png](support_post_health_files/support_post_health_21_1.png)



```python
df_monthly.columns
```




    Index(['date', 'calories', 'steps', 'dist', 'mins_sedant', 'hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed', 'resting_hr', 'sleep_efficiency', 'sleep_minutesAsleep'], dtype='object')




```python
dow_cats= ['Sunday','Monday','Tuesday','Wednesday','Thursday', 'Friday','Saturday']
dow_cats.reverse()
df_dow = df_master[df_master.date>'2019-01-01'].groupby('day')[cols_avg].agg('mean').reindex(dow_cats) 
df_dow.reset_index(inplace=True)
#df_dow = sleep_calcs(df_dow)
print(df_dow[['day','totalSleepMinutesAsleep']])
df_dow.plot.barh(x='day', y=['totalSleep_rem_mins','totalSleep_deep_mins'],title='Avg Daily Sleep from Jan 2019',figsize=(12,8))#, color='seagreen')
```

             day  totalSleepMinutesAsleep
    0   Saturday               397.404762
    1     Friday               311.166667
    2   Thursday               317.069767
    3  Wednesday               287.976744
    4    Tuesday               344.428571
    5     Monday               481.309524
    6     Sunday               477.071429





    <matplotlib.axes._subplots.AxesSubplot at 0x1a8ddbac88>




![png](support_post_health_files/support_post_health_23_2.png)


## Explain Box and Whiskers Plot


```python
df_dow = df_master[(df_master.date>'2019-01-01')&(df_master.totalSleepMinutesAsleep>0)&(df_master.totalSleep_light_mins>0)].groupby([pd.Grouper(key='date', freq='M'),'day'])['totalSleepMinutesAsleep','totalSleep_deep_mins'].agg('mean')
df_dow.reset_index(inplace=True)
df_dow['date']=df_dow['date'].dt.strftime('%Y-%m-%d')
g = sns.catplot(x="day", y="totalSleepMinutesAsleep", kind="box", data=df_dow, 
            order=['Sunday','Monday','Tuesday','Wednesday','Thursday', 'Friday','Saturday'])
g.fig.set_size_inches(14,10)
```


![png](support_post_health_files/support_post_health_25_0.png)


A box plot perfectly illustrates what we can do with basic statistical features:
•When the box plot is short it implies that much of your data points are similar, since there are many values in a small range
•When the box plot is tall it implies that much of your data points are quite different, since the values are spread over a wide range
•If the median value is closer to the bottom then we know that most of the data has lower values. If the median value is closer to the top then we know that most of the data has higher values. Basically, if the median line is not in the middle of the box then it is an indication of skewed data.
•Are the whiskers very long? That means your data has a high standard deviation and variance i.e the values are spread out and highly varying. If you have long whiskers on one side of the box but not the other, then your data may be highly varying only in one direction.


```python
from importlib import reload as reload
```


```python
reload(gsheets_health)
```




    <module 'gsheets_health' from '/Users/Shalu/Dropbox/Administrative/Documents/Health/datascience/main/gsheets_health.py'>



## Next Let's Look At Some Vitals and Biomarkers!


```python
import gsheets_health
df_daily_journal = gsheets_health.get_health_journal()
#Remove Future Dates
df_daily_journal = df_journal[df_journal.dow.notnull()]
df_daily_journal.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>dow</th>
      <th>Notes</th>
      <th>Meds</th>
      <th>Sleep Notes</th>
      <th>Sleep Meds</th>
      <th>Woke Up:</th>
      <th>Exercise Time</th>
      <th>Meditation/Health</th>
      <th>Exercise</th>
      <th>Alcohol</th>
      <th>Weight Time</th>
      <th>Weight</th>
      <th>Scale_BMI</th>
      <th>Scale_Fat</th>
      <th>Scale_TBW</th>
      <th>Scale_Musc</th>
      <th>Scale_Bone</th>
      <th>Handle_Athlete</th>
      <th>Handle_BMI</th>
      <th>Handle_Normal</th>
      <th>Arm | TOD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>137</th>
      <td>10/28/2019</td>
      <td>Mon</td>
      <td></td>
      <td>FertilAidex3, MotilityBoostx1, Add_10x3=30mg, ...</td>
      <td>Poor: 44, 51</td>
      <td>Magnesium, Zinc, SleepGlasses, Lavender_Diffus...</td>
      <td>2:00 AM</td>
      <td>Afternoon</td>
      <td></td>
      <td>Running 7:50mins@7.7@2.0@8:40mins1.1mile[coold...</td>
      <td>0</td>
      <td>wakeup morning</td>
      <td>183.6</td>
      <td>25.2</td>
      <td>19.70%</td>
      <td>60.80%</td>
      <td>40.00%</td>
      <td>7.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>138</th>
      <td>10/29/2019</td>
      <td>Tue</td>
      <td></td>
      <td>FertilAidex3, MotilityBoostx1, Add_10x3=30mg, ...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>Outdoor_run_4laps_1.32mil@11:33mins_8'46'Pace,...</td>
      <td></td>
      <td>wakeup morning</td>
      <td>179.8</td>
      <td>24.6</td>
      <td>18.90%</td>
      <td>61.40%</td>
      <td>40.40%</td>
      <td>7.4</td>
      <td>12.80%</td>
      <td></td>
      <td>13.10%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>155</th>
      <td>11/15/2019</td>
      <td></td>
      <td>2 months without drinking</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
range_name='doc_lab_tests!A:O'
```


```python
df_labs = gsheets_health.get_health_journal(range_name=range_name)

```

#### Formatting


```python
df_labs.shape
```




    (76, 15)




```python
df_labs.Component=df_labs.Component.str.lower()

df_labs['Threshold']= df_labs['Threshold'].replace(r'^\s*$', np.nan, regex=True)

labs_arr = ['vitamin b12, serum','homocysteine, serum','glucose','zinc, serum','cholesterol','hdl cholesterol','ldl cholesterol']

lab_date_cols = ['Component', '7/2/2015', '1/20/2017', '7/2/2017', '7/5/2017', '12/5/2017', '6/28/2018', '6/11/2019', '7/29/2019', '8/20/2019', '10/11/2019']

thresholds = df_labs[['Component','Threshold']][df_labs.Threshold.notnull()]
print(thresholds)
df_labs_short = df_labs[lab_date_cols][df_labs.Component.isin(labs_arr)]
# Remove everything after the comma
df_labs_short['Component'] = df_labs_short['Component'].str.split(',').str[0]
df_labs_short= df_labs_short.replace(r'^\s*$', np.nan, regex=True)
df_labs_short.set_index('Component',inplace=True,drop=True)

df_labs_short = df_labs_short.astype(float)
```


```python
df_labs_short
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>7/2/2015</th>
      <th>1/20/2017</th>
      <th>7/2/2017</th>
      <th>7/5/2017</th>
      <th>12/5/2017</th>
      <th>6/28/2018</th>
      <th>6/11/2019</th>
      <th>7/29/2019</th>
      <th>8/20/2019</th>
      <th>10/11/2019</th>
    </tr>
    <tr>
      <th>Component</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>vitamin b12</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>478.0</td>
      <td>353.0</td>
      <td>676.0</td>
    </tr>
    <tr>
      <th>homocysteine</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.9</td>
      <td>NaN</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>glucose</th>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>117.0</td>
      <td>92.0</td>
      <td>101.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>zinc</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>cholesterol</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>152.0</td>
      <td>172.0</td>
      <td>181.0</td>
      <td>NaN</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>hdl cholesterol</th>
      <td>62.0</td>
      <td>63.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>53.0</td>
      <td>58.0</td>
      <td>56.0</td>
      <td>NaN</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>ldl cholesterol</th>
      <td>83.0</td>
      <td>116.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>89.0</td>
      <td>100.0</td>
      <td>111.0</td>
      <td>NaN</td>
      <td>110.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_lipids = df_labs_short.transpose()
```


```python
lipid_dict = {
    'glucose':['6/28/2018', '6/11/2019','10/11/2019'],
    'cholesterol':['6/28/2018', '6/11/2019','10/11/2019'],
    'hdl cholesterol': ['6/28/2018', '6/11/2019','10/11/2019'],
    'ldl cholesterol': ['6/28/2018', '6/11/2019','7/29/2019']
}
```


```python
df_lipids
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Component</th>
      <th>vitamin b12</th>
      <th>homocysteine</th>
      <th>glucose</th>
      <th>zinc</th>
      <th>cholesterol</th>
      <th>hdl cholesterol</th>
      <th>ldl cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7/2/2015</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>1/20/2017</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>63.0</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>7/2/2017</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7/5/2017</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12/5/2017</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>117.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6/28/2018</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>92.0</td>
      <td>NaN</td>
      <td>152.0</td>
      <td>53.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>6/11/2019</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>101.0</td>
      <td>NaN</td>
      <td>172.0</td>
      <td>58.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>7/29/2019</th>
      <td>478.0</td>
      <td>12.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181.0</td>
      <td>56.0</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>8/20/2019</th>
      <td>353.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10/11/2019</th>
      <td>676.0</td>
      <td>8.7</td>
      <td>88.0</td>
      <td>62.0</td>
      <td>178.0</td>
      <td>58.0</td>
      <td>110.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
graph={}
for component in lipid_dict:
    print(df_lipids[component][lipid_dict[component]].values)
    graph[component]=df_lipids[component][lipid_dict[component]].values
```

    [ 92. 101.  88.]
    [152. 172. 178.]
    [53. 58. 58.]
    [ 89. 100. 111.]



```python
temp = pd.DataFrame.from_dict(graph,'index',columns=['last year','July 2019','3 months later'])
temp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last year</th>
      <th>July 2019</th>
      <th>3 months later</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>glucose</th>
      <td>92.0</td>
      <td>101.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>cholesterol</th>
      <td>152.0</td>
      <td>172.0</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>hdl cholesterol</th>
      <td>53.0</td>
      <td>58.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>ldl cholesterol</th>
      <td>89.0</td>
      <td>100.0</td>
      <td>111.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
temp1 = pd.DataFrame(temp.stack()).reset_index()
temp1.rename(columns = {'level_0':'component','level_1':'time',0:'value'}, inplace=True)
```


```python
temp1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>component</th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>glucose</td>
      <td>last year</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>glucose</td>
      <td>July 2019</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>glucose</td>
      <td>3 months later</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cholesterol</td>
      <td>last year</td>
      <td>152.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cholesterol</td>
      <td>July 2019</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cholesterol</td>
      <td>3 months later</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hdl cholesterol</td>
      <td>last year</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hdl cholesterol</td>
      <td>July 2019</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>hdl cholesterol</td>
      <td>3 months later</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ldl cholesterol</td>
      <td>last year</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ldl cholesterol</td>
      <td>July 2019</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ldl cholesterol</td>
      <td>3 months later</td>
      <td>111.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

x_label = ['July 2019','3 months later']
df_lipids[['vitamin b12']] [df_lipids.index.isin(['8/20/2019', '10/11/2019'])].reset_index().plot(ax=ax1,x='index',y='vitamin b12',kind='bar',use_index=False,rot=45,legend=False,title='B12 Levels',color='Green')
ax1.set_xticklabels(x_label)
ax1.plot([-1, 2], [400, 400], "k--")

x_label = ['July 2019','3 months later']
df_lipids[['homocysteine']] [df_lipids.index.isin(['7/29/2019', '10/11/2019'])].reset_index().plot(ax=ax2,x='index',y='homocysteine',kind='bar',use_index=False,rot=45,legend=False,title='Homocysteine',color='LightBlue')
ax2.set_xticklabels(x_label)
ax2.plot([-1, 2], [10, 10], "k--")

```




    [<matplotlib.lines.Line2D at 0x1a8ef04be0>]




![png](support_post_health_files/support_post_health_44_1.png)



```python
temp.transpose().reset_index().plot(kind='line',grid=True,subplots=False,x='index',figsize=(12,9), sharex=True, legend=True,title='Levels')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a8ddc3da0>




![png](support_post_health_files/support_post_health_45_1.png)



```python
ax = plt.gca()
temp.transpose().reset_index().plot(kind='bar',x='index',y='glucose', color='purple', ax=ax,rot=45,figsize=(10,7))
ax.set_ylim([45, 115])
```




    (45, 115)




![png](support_post_health_files/support_post_health_46_1.png)



```python
temp.transpose().reset_index()[['index', 'cholesterol', 'hdl cholesterol','ldl cholesterol']].plot(kind='bar',grid=True,subplots=False,x='index',figsize=(12,9), sharex=True, legend=True,title='Lipid Panels')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a81eb6e80>




![png](support_post_health_files/support_post_health_47_1.png)



```python
ax = plt.gca()

temp.transpose().reset_index().plot(kind='bar',x='index',y='ldl cholesterol', color='Black', ax=ax,rot=45,legend=False,title='LDL Levels')
ax.set_ylim([80, 115])
ax.plot([-1, 3], [100,100], "k--",color='Grey')

highlight = 2
ax.patches[highlight].set_facecolor('#aa3333')
plt.show()
```


![png](support_post_health_files/support_post_health_48_0.png)



```python

```


```python

```


```python

```


```python

```


```python

```

# What Ranges?
resting_HR - big negatives .78 on calc_acitve mins, -.55 with light sleep, .57 mins sedant, .63 steps 
totalSleep_deep_mins - -.35 resting hr, mins_sedant .82
calc_deep_sleep - HR_Total calories?

The big metrics to focus on are:
calc_active_mins, mins_sedant, hr_total_calories, steps

calc_active_mins = >125, min 100]
mins_sedant = <700, max 800
hr_total_calories = 3200, min 3000 
steps = , min 12500




```python

```


```python

```
