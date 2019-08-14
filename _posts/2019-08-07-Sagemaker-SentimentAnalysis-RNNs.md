---
layout: post
title: AWS Sagemaker and Sentiment Analysis 
date: '2019-08-07 12:07:25 +0000'
categories:
  - data
published: true
---

# AWS Sagemaker and Sentiment Analysis 
My goal this summer of exploring some of the new data science technologies wouldn't be complete without looking at some ways to deploy productionized models.  I already have had the experience of running my models on an EC2 instance a couple of years ago, which was not an ideal experience.  To be fair I was seeking an alternative to running my model locally not necessarily a scaleable solution.  I figured it would be great to see all the progress using some of the new technologies out there.  I started looking at AWS Sagemaker which allows you to build and deploy machine learning models and they also have some good [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) filled with examples.

One of the relevant data sets in the examples is the [IMDb Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/), so best if we just use that.  

## Get and Prepare the Data
This data contains a critic's review and an associated text label which indicates a positive or negative review. I'll have to process the data a bit converting the labels, breaking them out into train and test sets and shuffling them around (based on feedback I'll focus more on brevity on this post), but full code is on my github here.



```python
train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
```

    IMDb reviews (combined): train = 25000, test = 25000


Now that we have our train and test sets, let's take a quick look at the data below.  


```python
print(train_X[101])
print(train_y[101])
```

    When a movie shocks you with it's disturbing, brooding atmosphere, and grabs you by the throat with it's stunning cinematography, you just know that you have stumbled upon a treat, masterpiece of a film. Although with most modern movies, extremely enjoyable as some are, those that really shock you into focus are the strongest, and are the ones that are most critically acclaimed and mostly, stick with you for a life time. I say, proudly, that I am a fan of movies that disturb, not just horror movies, but those that send a vibe laden with foreboding. Movies like Breakdown and The Missing, etc etc etc et.
    1


Now that we have our data, I'll quickly tokenize the reviews using NTLK and Beautiful soup in the below function. Aside from tokenizing, I converted to lower case, removed stopwords, and stemmed the words. If you're unfamiliar, it just isolates the variants off a root words i.e. liked, likes, liking, etc.

Here it is below:


```python
print(words_from_review(train_X[101]))
```

    ['movi', 'shock', 'disturb', 'brood', 'atmospher', 'grab', 'throat', 'stun', 'cinematographi', 'know', 'stumbl', 'upon', 'treat', 'masterpiec', 'film', 'although', 'modern', 'movi', 'extrem', 'enjoy', 'realli', 'shock', 'focu', 'strongest', 'one', 'critic', 'acclaim', 'mostli', 'stick', 'life', 'time', 'say', 'proudli', 'fan', 'movi', 'disturb', 'horror', 'movi', 'send', 'vibe', 'laden', 'forebod', 'movi', 'like', 'breakdown', 'miss', 'send', 'chill', 'spine', 'make', 'think', 'holi', 'crap', 'could', 'happen', 'visual', 'entic', 'favorit', 'aspect', 'movi', '21', 'grow', 'actor', 'like', 'burt', 'renyold', 'jon', 'voight', 'ned', 'beatti', 'albeit', 'familiar', 'watch', 'grow', 'proceed', 'actor', 'oppos', 'actor', 'like', 'shia', 'labouf', 'justin', 'long', 'must', 'say', 'long', 'hype', 'wit', 'deliver', 'first', 'time', 'admir', 'veteran', 'actor', 'movi', 'made', '30', 'year', 'ago', 'still', 'live', 'terror', 'competit', 'modern', 'movi', 'burt', 'renyold', 'play', 'lewi', 'macho', 'self', 'appoint', 'leader', 'group', 'four', 'friend', 'cano', 'trip', 'fictiti', 'river', 'abdridged']



```python
# Convert each review and cache the file
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
```

    Wrote preprocessed data to cache file: preprocessed_data.pkl


## Transform the data


Now we from our reviews we have turned them into words (getting closer to the fun stuff).  Next we need to turn these words into unique numbers to correctly identify them, which we can do by building a word dictionary.


```python
import numpy as np
import re
from collections import Counter

def build_dict(data, vocab_size = 5000):
    # We want to map only our most frequently occuring words and we'll leave 0 and 1 
    # for no words (short reviews) and infrequent words 
    # we'll have a vocabulary of vocab_size - 2

    word_count = Counter(np.concatenate(data, axis=0 ))
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} 
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 for infrequent labels and no words
        word_dict[word] = idx + 2                              
        
    return word_dict
```


```python
word_dict = build_dict(train_X)
print(word_dict)
```

    {'movi': 2, 'film': 3, 'one': 4, 'like': 5, 'time': 6, 'good': 7, 'make': 8, 'charact': 9, 'get': 10, 'see': 11, 'watch': 12, 'stori': 13, 'even': 14, 'would': 15, 'realli': 16, 'well': 17, 'scene': 18, 'look': 19, 'show': 20, 'much': 21, 'end': 22, 'peopl': 23, 'bad': 24, 'go': 25, 'great': 26, 'also': 27, 'first': 28, 'love': 29, 'think': 30, 'way': 31, 'act': 32, 'play': 33, 'made': 34, 'thing': 35, 'could': 36, 'know': 37,'example':100000', 'masterson': 4994, 'weaker': 4995, 'rapidli': 4996, 'drone': 4997, 'turtl': 4998, 'frontal': 4999}


### Transform the reviews

Now that we have our word dictionary, we need to apply it to our reviews.  This essentially turns our reviews into numerical representations and in order to keep things uniform we have to ensure that each numerically transformed review are all a uniform size (500 in this case). As you can see below, we now have a numerical representation of one review.


```python
train_X[0][:88]
```




    array([  12,  857,    6,  664,  552,  312,  285,   29, 1496, 2491,  105,
            148,  820,   77,   29, 2055,    3,    4,   26,   41,  486,   84,
            260,    3,  241,    1,  157, 1974,  471,  429,  124,    1,    8,
              3,   69,   12,   28,   11, 1974,  257,    1, 1047,  255,  841,
             49,  580,    2,  190,   59,   26,  476,  215,   12,    3, 4633,
              1, 1986, 4633,    1, 1986,   84, 1170,  121,  691, 1261,  937,
             12,  448,   48, 2617,  101,    6,  783,   25,   12,    1,  838,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0])



## Save the data locally, preparing to upload to S3

We will need to upload the training dataset to the default SageMaker S3 bucket in order for our training code to access it. 


```python
import pandas as pd
import sagemaker

pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'

role = sagemaker.get_execution_role()
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
```

## Training the PyTorch Model

What was interesting was that a model in the SageMaker framework comprises of three things: Model Artifacts, Training Code, and Inference Code.  

I was working on a LSTM model prior as I was exploring [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) and grabbed the simple RNN model below for sentiment analysis.

import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())



We need to provide SageMaker with a training script in order to construct a pyTorch model.


```python
def train(model, train_loader, epochs, optimizer, loss_fn, device):    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model.forward(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
```

Put the above function in `train.py` in the train folder.


```python
import torch.optim as optim
from train.model import LSTMClassifier

from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })
```


```python
estimator.fit({'training': input_data})
```

    2019-08-10 23:31:15 Starting - Starting the training job...
    2019-08-10 23:31:17 Starting - Launching requested ML instances......
    2019-08-10 23:32:45 Starting - Preparing the instances for training.........
    2019-08-10 23:34:07 Downloading - Downloading input data...
    2019-08-10 23:34:34 Training - Downloading the training image..
    
    [31mInvoking script with the following command:
    [0m
    [31m/usr/bin/python -m train --epochs 10 --hidden_dim 200
    
    [0m
    [31mUsing device cuda.[0m
    [31mGet train data loader.[0m
    [31mModel loaded with embedding_dim 32, hidden_dim 200, vocab_size 5000.[0m
    [31mEpoch: 1, BCELoss: 0.671973701642484[0m
    [31mEpoch: 2, BCELoss: 0.6233081695984821[0m
    [31mEpoch: 3, BCELoss: 0.5193899894247249[0m
    [31mEpoch: 4, BCELoss: 0.43153597079977696[0m
    [31mEpoch: 5, BCELoss: 0.37278965480473575[0m
    [31mEpoch: 6, BCELoss: 0.3399012107021955[0m
    [31mEpoch: 7, BCELoss: 0.3150354915735673[0m
    [31mEpoch: 8, BCELoss: 0.29557760151065127[0m
    [31mEpoch: 9, BCELoss: 0.30807020956156206[0m
    
    2019-08-10 23:38:18 Uploading - Uploading generated training model[31mEpoch: 10, BCELoss: 0.3036571315356663[0m
    [31m2019-08-10 23:38:13,465 sagemaker-containers INFO     Reporting training SUCCESS[0m
    
    2019-08-10 23:38:23 Completed - Training job completed
    Billable seconds: 257


## Deploy and Test Our Model

Now we can deploy and test our model, note you should close all endpoints or you will be charged.  It's important to be aware every time you create an endpoint.



```python
# Deploy our model
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

    ----------------------------------------------------------------------------------------------------!


```python
# Chunking the data and adding results.
test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)
def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, predictor.predict(array))
    
    return predictions

predictions = predict(test_X.values)
predictions = [round(num) for num in predictions]
```


```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```




    0.8372



**Not bad:** <br>
If I were more interested in the model accuracy than learning the tools, I would likely try optimizing across one or all the parameters below:
- the embedding dimension, 
- hidden dimension, 
- and/or size of the vocabulary) 

As Deep Learning Models are senstive and require some tuning, but that should result in significant performance improvements.

## Web App For Our Little Model

We've now created our endpoint, how would we create a webapp that can submit our data to our model in Sagemaker?  I found the image below on Amazon's website.


![Drag Racing](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2018/07/18/sagemaker-endpoint-1.gif)


```python
predictor.endpoint
```




    'sagemaker-pytorch-2019-08-12-16-22-32-575'



### In order to get an example webapp set up, we need to do the following steps:
    1. Create an IAM role for the lambda function
    2. Create the lambda function and associate it with our model using the endpoint listed above
    3. Create an API Gateway and retrieve a url for our API

From API Gateway, we retrieve a url and then you can just create a local webpage with the follow:

User Prompt: "Enter your review below and click submit"
<u>Code Snippet</u>


```python
<form method="POST" action="https://yourAPIURL.com" onsubmit="return submitForm(this);" >       
```

Some Sample Reviews:

Input: This movie started off with so much potential and then disappointed terribly.
<br>Output: Your review was NEGATIVE!

I then grabbed a handful of reviews from Rotten Tomatoes from the L and surprisingly did pretty well<br>

Input: Beautiful but disappointing. The images feel very real at times, but they lack the character and heart of both truly real animals and traditional animation.<br>
Output: Your review was NEGATIVE!

Input: Despite the superstar talent of the cast and the stunning presentation, it misses some of the heart that placed the original securely in the pop culture canon.<br>
Output: Your review was POSITIVE!



Then I just deleted my endpoint  with predictor.delete_endpoint()
