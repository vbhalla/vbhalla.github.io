---
layout: post
title: We're up and running!
published: true
---

Around 3 years ago I was looking for a fun supervised learning problem where I could familiarize myself with the popular python machine learning library [scikit-learn](https://scikit-learn.org/stable/).  I found a dataset on Kaggle that had Lending Club's historical default rates on 2014 and 2015 vintages and thought that it may be a fun exercise to apply some machine learning methods and see if I can beat the performance of a portfolio I constructed manually.

Now if I were to do this again, I'd definitely approach the feature extraction much differently as I've learned a lot in the past 3 years. 

Now this comparison isn't exactly apples to apples in terms of loan characteristics and seasoning across both portfolios. Mostly because I already had my manually constructed portfolio a few months earlier.  Again if I were to do this again, I also would have documented the portfolio selection below much more thoroughly :-)   

But after a few years:

| Portfolio | Charged Off | Current + Fully Paid | Default Rate |
| :---  | :---: | :---: | :---: | 
| Manually Constructed | 30 | 197 | **13.22%** |
| ML Algorithm | 6 | 55 | **9.84%** |

Not too bad, although definitely not my bestwork.  When I get some time, I'd like to revisit and backtest this with some new methods, perhaps a Nueral Network.....

Old Project below:

[Link : 2016 Loan Defaults Project](https://github.com/vbhalla/Projects/tree/master/102016_loan_defaults)
