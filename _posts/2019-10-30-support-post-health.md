
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


![png](../images/health_post/support_post_health_8_0.png)




![png](../images/health_post/support_post_health_9_0.png)




![png](../images/health_post/support_post_health_10_0.png)




![png](../images/health_post/support_post_health_11_0.png)


## Dive Into Sleep




![png](../images/health_post/support_post_health_14_1.png)


## Sleep Efficiency Is A Blackbox Number, Disregarding




![png](../images/health_post/support_post_health_16_1.png)




![png](../images/health_post/support_post_health_18_1.png)



![png](../images/health_post/support_post_health_19_1.png)


## Now we got a broad view of things improving, albeit still very far from ideal!
### Let's dig a bit deeper




![png](../images/health_post/support_post_health_21_1.png)





![png](../images/health_post/support_post_health_23_2.png)

The above doesn't really give us any idea of the variance aside from the means across our weekdays.


![png](../images/health_post/support_post_health_25_0.png)

That's better, the box plot is great here as if the box plot is tall then that implies that our data is pretty spread out.  Should the median value be closer to the bottom, then we know the distribution is mostly towards lower values and the whiskers really give us a much better idea of the variance and standard deviations.



## Next Let's Look At Some Vitals and Biomarkers!



    [<matplotlib.lines.Line2D at 0x1a8ef04be0>]




![png](../images/health_post/support_post_health_44_1.png)




![png](../images/health_post/support_post_health_45_1.png)




![png](../images/health_post/support_post_health_46_1.png)





![png](../images/health_post/support_post_health_47_1.png)




![png](../images/health_post/support_post_health_48_0.png)



# What Ranges?
The big metrics to focus on are:
calc_active_mins, mins_sedant, hr_total_calories, steps, and sleep




