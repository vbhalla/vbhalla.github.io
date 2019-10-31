
---
layout: post
title: Healthy Data Wrangling
date: '2019-10-28 12:07:25 +0000'
categories:
  - data
published: true
---

In the past year, I have really come to appreciate the rollercoaster of health both personally and how it relates to the people we care about.  It's been a rollercoaster of emotions: daunting, confusing, intimidating, saddening, madding, and even quite empowering.  All that is a story for another day, but building a data-driven foundation and weaving it into the DNA of an organization is my bread and butter; infrastructure, experimentation, mining for insights, and bringing it all together is what I love doing what I do.  I wondered if I could swapping out the business for my body?  Like most things, much easier said than done....enjoy!

## Monthly Cumulative - Weekdays


```python
df_monthly = df_master[~df_master.day.isin(['Saturday','Sunday'])].groupby(pd.Grouper(key='date', freq='M'))[cols_avg].agg('sum')
df_monthly[['calories', 'steps', 'dist', 'mins_sedant', 'hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed']][df_monthly.index > '2017-07-31'].tail(10)
df_monthly.reset_index(inplace=True)
df_monthly[['date','calories', 'steps', 'dist']][df_monthly.date>'2017-11-01'].plot(subplots=True,x='date',figsize=(12,9), sharex=True, legend=True,title='Monthly Cumulative Calories, Steps, Distances')
```

![png](../images/health_post/support_post_health_6_1.png)

## Monthly Averages - Weekdays


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

![png](../images/health_post/support_post_health_8_0.png)




![png](../images/health_post/support_post_health_9_0.png)



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


![png](../images/health_post/support_post_health_10_0.png)




![png](../images/health_post/support_post_health_11_0.png)


## Dive Into Sleep



    Index(['date', 'calories', 'steps', 'dist', 'mins_sedant', 'hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed', 'resting_hr', 'sleep_efficiency', 'sleep_minutesAsleep'], dtype='object')




![png](../images/health_post/support_post_health_14_1.png)


## Sleep Efficiency Is A Blackbox Number, Disregarding




![png](../images/health_post/support_post_health_16_1.png)




![png](../images/health_post/support_post_health_18_1.png)



![png](../images/health_post/support_post_health_19_1.png)


## Now we got a broad view of things improving, albeit still very far from ideal!
### Let's dig a bit deeper


```python
df_dow = df_master[(df_master.date>'2018-05-01')&(df_master.totalSleepMinutesAsleep>0)&(df_master.totalSleep_light_mins>0)].groupby([pd.Grouper(key='date', freq='M')])[cols_avg].agg('mean')
df_dow.index=df_dow.index.strftime('%Y-%m-%d')
df_dow[['totalSleep_deep_mins', 'totalSleep_light_mins', 'totalSleep_rem_mins']].plot(kind='bar',stacked=True,rot=45,figsize=(12,8),title='Minutes of Sleep Distributed Across Sleep Cycle')
```



![png](../images/health_post/support_post_health_21_1.png)


             day  totalSleepMinutesAsleep
    0   Saturday               397.404762
    1     Friday               311.166667
    2   Thursday               317.069767
    3  Wednesday               287.976744
    4    Tuesday               344.428571
    5     Monday               481.309524
    6     Sunday               477.071429




![png](../images/health_post/support_post_health_23_2.png)


## Explain Box and Whiskers Plot


```python
df_dow = df_master[(df_master.date>'2019-01-01')&(df_master.totalSleepMinutesAsleep>0)&(df_master.totalSleep_light_mins>0)].groupby([pd.Grouper(key='date', freq='M'),'day'])['totalSleepMinutesAsleep','totalSleep_deep_mins'].agg('mean')
df_dow.reset_index(inplace=True)
df_dow['date']=df_dow['date'].dt.strftime('%Y-%m-%d')
g = sns.catplot(x="day", y="totalSleepMinutesAsleep", kind="box", data=df_dow, 
            order=['Sunday','Monday','Tuesday','Wednesday','Thursday', 'Friday','Saturday'])
g.fig.set_size_inches(14,10)
```


![png](../images/health_post/support_post_health_25_0.png)


A box plot perfectly illustrates what we can do with basic statistical features:
•When the box plot is short it implies that much of your data points are similar, since there are many values in a small range
•When the box plot is tall it implies that much of your data points are quite different, since the values are spread over a wide range
•If the median value is closer to the bottom then we know that most of the data has lower values. If the median value is closer to the top then we know that most of the data has higher values. Basically, if the median line is not in the middle of the box then it is an indication of skewed data.
•Are the whiskers very long? That means your data has a high standard deviation and variance i.e the values are spread out and highly varying. If you have long whiskers on one side of the box but not the other, then your data may be highly varying only in one direction.




## Next Let's Look At Some Vitals and Biomarkers!



    [<matplotlib.lines.Line2D at 0x1a8ef04be0>]




![png](../images/health_post/support_post_health_44_1.png)




![png](../images/health_post/support_post_health_45_1.png)




![png](../images/health_post/support_post_health_46_1.png)





![png](../images/health_post/support_post_health_47_1.png)




![png](../images/health_post/support_post_health_48_0.png)



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



