---
layout: post
title: Healthy Data Wrangling
date: '2019-10-28 12:07:25 +0000'
categories:
  - data
published: true
---

In the past year, I have really come to appreciate the rollercoaster of health both personally and how it relates to the people we care about.  All that is a story for another day, but building a data-driven foundation and weaving it into the DNA of an organization is my background.  Soup to nuts, starting with infrastructure and developing baselines, mining for insights, to eventual experimentation, and then bringing it all together in order to drive transformational change is something I'm both good at and enjoy greatly.  You see where I'm going with this?  

Given some health setbacks, even with the limited data I currently had, it seemed pretty obvious that things had gone off the rails.

![png](../images/health_post/support_post_health_1.png)

If I could swap out the business for my body, would the same rules apply?  First rule is that it is much easier said than done....enjoy!

## Monthly Cumulative - Weekdays


```python
df_monthly[['calories', 'steps', 'dist', 'mins_sedant', 'hr_total_calories', 'mins_active_light', 'calc_active_mins', 'totalSleepMinutesAsleep', 'totalSleep_deep_mins', 'totalSleep_rem_mins', 'totalSleep_wake_mins', 'totalSleepTimeInBed']][df_monthly.index > '2017-07-31'].tail(10)
df_monthly.reset_index(inplace=True)
df_monthly[['date','calories', 'steps', 'dist']][df_monthly.date>'2017-11-01'].plot(subplots=True,x='date',figsize=(12,9), sharex=True, legend=True,title='Monthly Cumulative Calories, Steps, Distances')
```
After hooking into a few apis and a lot of data wrangling, which took awhile, we're finally in a place to get started to see what we're working with.  

Some quick plots around my wearable shows the increase in activity the past 3 months, but it (I) bounce around.

![png](../images/health_post/support_post_health_6_1.png)

That's all well and good, but cumulative monthly values are difficult for me to wrap my head around and bring into my day-to-day.  Perhaps averages may be more interesting and I'll scrub out my weekends, as my weekends tend to be rather varied from week to week.

## Monthly Averages - Weekdays

![png](../images/health_post/support_post_health_8_0.png)

This was rather promising to see, as I've really committed to improving my cardiovascular health due to all the heart disease in my family and I'm finally starting to see some progress which is extremely motivating!

Now I've hooked into quite a few disparate data platforms and munged them together.  I think a correlation plot, although visually intimidating at first, may be a good way start off developing an underestanding of this unfamiliar dataset and the inter-relationships among features.


![png](../images/health_post/support_post_health_9_0.png)

There is a lot of good stuff here, what is interesting to see here is that sendentary minutes in addition to active minutes seem pretty correlated to my resting heart rate.

These active and sedentary minutes below exhibit the inverse relationship that one would expect, which is great to see as then that means we can start to trust this data a bit more (that's not always the case and frequently cause a lot of headaches for data scientists later on).

![png](../images/health_post/support_post_health_10_0.png)

![png](../images/health_post/support_post_health_11_0.png)


### Let's Dive Into Sleep, What is Sleep Efficiency? 

![png](../images/health_post/support_post_health_14_1.png)

As you can see above, the sleep efficiency metric doesn't appear too correlated to sleep minutes.  This number is a bit of a black box and due to insufficient information on the calculation I'm going to disregard it for now.

![png](../images/health_post/support_post_health_16_1.png)

I have always suffered from terrible insomnia, and no matter how much sleep hygiene and tricks I still haven't felt I've made too much progress. <b>BUT</b> at first glance, it does seem to be stabilizing and sloping upward.  Think we need to take a closer view.


![png](../images/health_post/support_post_health_18_1.png)

If we zoom in a bit closer, I think we can see how much 'opportunity' there is; look at that daunting dotted line implying 8 hours!  I'm still no way near 8 hours (480 mins), which is quite troubling actually as there is more and more research out on the hazards to heath caused by sleep deprivation. 

This isn't very promising, but let's see if we can smooth it out over quarters. Now that's a bit more motivating!  I can take some solace in that there is some upward movement.  Hopefully this continues as I have a large hike up to 8 hours!  

Broadly things seem to improving, hopefully I can increase that slope quite a bit. 


![png](../images/health_post/support_post_health_19_1.png)



Let's dig a bit deeper and see if we can get a bit more granular, particularly the <u>quality of sleep</u> as opposed to simply the total time.


![png](../images/health_post/support_post_health_21_1.png)

The above gave us a weekly view, but I wonder if there is some variability within the week?

![png](../images/health_post/support_post_health_23_2.png)

We can see a bit of variability, but the above doesn't really give us any idea of the variance aside from the means across our weekdays.

This this calls for a box plot, a favorite of mine.


![png](../images/health_post/support_post_health_25_0.png)

That's better, the box plot is great here as if the box plot is short then that implies that our data is not very spread out (Wednesday).  Should the median value be closer to the bottom, then we know the distribution is mostly towards lower values and the whiskers really give us a much better idea of the variance and standard deviations.



## Next Let's Look At Some Vitals and Biomarkers!

    [<matplotlib.lines.Line2D at 0x1a8ef04be0>]


I have modified my diet a bit over the past 3 months and included a b-complex supplement, which appears to be helping here:

![png](../images/health_post/support_post_health_44_1.png)


Now let's get to the nitty gritty, glucose and lipid panels.


![png](../images/health_post/support_post_health_45_1.png)

Taking a closer look, it seems like the glucose has improved based on my diet and exercise modifications!


![png](../images/health_post/support_post_health_46_1.png)


Unfortunately my cholesterol seems to be getting worse, which is not good due to the high risk of cardiovascular disease in my family.


![png](../images/health_post/support_post_health_47_1.png)

Yikes....that LDL is no bueno

![png](../images/health_post/support_post_health_48_0.png)


Well after all of that, I don't have a conclusion rather than a few more To-Dos :) 

That includes talking to my doctor particularly about my LDL levels and evaluating if I need to run some more tests (I hear the LDL-p is a good number)!  

In addition, my wife is going to ensure I go to a sleep doctor and has threatened to drag me there herself if I don't make that appointment ASAP.

I did also gain a set of average metrics across week, particularly active minutes and hr_calories which can ensure I'm at least maintaining my momentum as I don't want that resting heart rate to start creeping back up.

But this was both a painful exercise from a data munging perspective, but I think the outcome is particularly motivating as it's great to see progress as well as see things that don't "budge" as then you can run anotehr experiment or switch tactics.  

P.S. 
These apis and 3rd party data sources were particularly messy, so I need to modularize my code a bit before posting in the repo, so it will be in there in a week or so!




