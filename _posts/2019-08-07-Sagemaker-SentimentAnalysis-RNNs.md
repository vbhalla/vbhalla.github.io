---
published: false
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
    [31mbash: cannot set terminal process group (-1): Inappropriate ioctl for device[0m
    [31mbash: no job control in this shell[0m
    [31m2019-08-10 23:34:55,814 sagemaker-containers INFO     Imported framework sagemaker_pytorch_container.training[0m
    [31m2019-08-10 23:34:55,850 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.[0m
    [31m2019-08-10 23:34:57,266 sagemaker_pytorch_container.training INFO     Invoking user training script.[0m
    [31m2019-08-10 23:34:57,528 sagemaker-containers INFO     Module train does not provide a setup.py. [0m
    [31mGenerating setup.py[0m
    [31m2019-08-10 23:34:57,528 sagemaker-containers INFO     Generating setup.cfg[0m
    [31m2019-08-10 23:34:57,528 sagemaker-containers INFO     Generating MANIFEST.in[0m
    [31m2019-08-10 23:34:57,528 sagemaker-containers INFO     Installing module with the following command:[0m
    [31m/usr/bin/python -m pip install -U . -r requirements.txt[0m
    [31mProcessing /opt/ml/code[0m
    [31mCollecting pandas (from -r requirements.txt (line 1))
      Downloading https://files.pythonhosted.org/packages/74/24/0cdbf8907e1e3bc5a8da03345c23cbed7044330bb8f73bb12e711a640a00/pandas-0.24.2-cp35-cp35m-manylinux1_x86_64.whl (10.0MB)[0m
    [31mCollecting numpy (from -r requirements.txt (line 2))
      Downloading https://files.pythonhosted.org/packages/69/25/eef8d362bd216b11e7d005331a3cca3d19b0aa57569bde680070109b745c/numpy-1.17.0-cp35-cp35m-manylinux1_x86_64.whl (20.2MB)[0m
    [31mCollecting nltk (from -r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/87/16/4d247e27c55a7b6412e7c4c86f2500ae61afcbf5932b9e3491f8462f8d9e/nltk-3.4.4.zip (1.5MB)[0m
    [31mCollecting beautifulsoup4 (from -r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/1a/b7/34eec2fe5a49718944e215fde81288eec1fa04638aa3fb57c1c6cd0f98c3/beautifulsoup4-4.8.0-py3-none-any.whl (97kB)[0m
    [31mCollecting html5lib (from -r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/a5/62/bbd2be0e7943ec8504b517e62bab011b4946e1258842bc159e5dfde15b96/html5lib-1.0.1-py2.py3-none-any.whl (117kB)[0m
    [31mRequirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /usr/local/lib/python3.5/dist-packages (from pandas->-r requirements.txt (line 1)) (2.7.5)[0m
    [31mCollecting pytz>=2011k (from pandas->-r requirements.txt (line 1))
      Downloading https://files.pythonhosted.org/packages/87/76/46d697698a143e05f77bec5a526bf4e56a0be61d63425b68f4ba553b51f2/pytz-2019.2-py2.py3-none-any.whl (508kB)[0m
    [31mRequirement already satisfied, skipping upgrade: six in /usr/local/lib/python3.5/dist-packages (from nltk->-r requirements.txt (line 3)) (1.11.0)[0m
    [31mCollecting soupsieve>=1.2 (from beautifulsoup4->-r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/35/e3/25079e8911085ab76a6f2facae0771078260c930216ab0b0c44dc5c9bf31/soupsieve-1.9.2-py2.py3-none-any.whl[0m
    [31mCollecting webencodings (from html5lib->-r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/f4/24/2a3e3df732393fed8b3ebf2ec078f05546de641fe1b667ee316ec1dcf3b7/webencodings-0.5.1-py2.py3-none-any.whl[0m
    [31mBuilding wheels for collected packages: nltk, train
      Running setup.py bdist_wheel for nltk: started[0m
    [31m  Running setup.py bdist_wheel for nltk: finished with status 'done'
      Stored in directory: /root/.cache/pip/wheels/41/c8/31/48ace4468e236e0e8435f30d33e43df48594e4d53e367cf061
      Running setup.py bdist_wheel for train: started
      Running setup.py bdist_wheel for train: finished with status 'done'
      Stored in directory: /tmp/pip-ephem-wheel-cache-8p45ww6s/wheels/35/24/16/37574d11bf9bde50616c67372a334f94fa8356bc7164af8ca3[0m
    [31mSuccessfully built nltk train[0m
    
    2019-08-10 23:34:55 Training - Training image download completed. Training in progress.[31mInstalling collected packages: numpy, pytz, pandas, nltk, soupsieve, beautifulsoup4, webencodings, html5lib, train
      Found existing installation: numpy 1.15.4
        Uninstalling numpy-1.15.4:
          Successfully uninstalled numpy-1.15.4[0m
    [31mSuccessfully installed beautifulsoup4-4.8.0 html5lib-1.0.1 nltk-3.4.4 numpy-1.17.0 pandas-0.24.2 pytz-2019.2 soupsieve-1.9.2 train-1.0.0 webencodings-0.5.1[0m
    [31mYou are using pip version 18.1, however version 19.2.1 is available.[0m
    [31mYou should consider upgrading via the 'pip install --upgrade pip' command.[0m
    [31m2019-08-10 23:35:09,587 sagemaker-containers INFO     Invoking user script
    [0m
    [31mTraining Env:
    [0m
    [31m{
        "num_cpus": 4,
        "additional_framework_parameters": {},
        "log_level": 20,
        "input_config_dir": "/opt/ml/input/config",
        "num_gpus": 1,
        "job_name": "sagemaker-pytorch-2019-08-10-23-31-15-121",
        "output_dir": "/opt/ml/output",
        "model_dir": "/opt/ml/model",
        "user_entry_point": "train.py",
        "output_data_dir": "/opt/ml/output/data",
        "current_host": "algo-1",
        "channel_input_dirs": {
            "training": "/opt/ml/input/data/training"
        },
        "hosts": [
            "algo-1"
        ],
        "module_dir": "s3://sagemaker-us-east-1-545961142568/sagemaker-pytorch-2019-08-10-23-31-15-121/source/sourcedir.tar.gz",
        "hyperparameters": {
            "epochs": 10,
            "hidden_dim": 200
        },
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "network_interface_name": "eth0",
        "framework_module": "sagemaker_pytorch_container.training:main",
        "module_name": "train",
        "input_data_config": {
            "training": {
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None",
                "TrainingInputMode": "File"
            }
        },
        "resource_config": {
            "current_host": "algo-1",
            "network_interface_name": "eth0",
            "hosts": [
                "algo-1"
            ]
        },
        "input_dir": "/opt/ml/input"[0m
    [31m}
    [0m
    [31mEnvironment variables:
    [0m
    [31mSM_NETWORK_INTERFACE_NAME=eth0[0m
    [31mSM_INPUT_CONFIG_DIR=/opt/ml/input/config[0m
    [31mSM_FRAMEWORK_PARAMS={}[0m
    [31mPYTHONPATH=/usr/local/bin:/usr/lib/python35.zip:/usr/lib/python3.5:/usr/lib/python3.5/plat-x86_64-linux-gnu:/usr/lib/python3.5/lib-dynload:/usr/local/lib/python3.5/dist-packages:/usr/lib/python3/dist-packages[0m
    [31mSM_MODEL_DIR=/opt/ml/model[0m
    [31mSM_USER_ENTRY_POINT=train.py[0m
    [31mSM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main[0m
    [31mSM_HOSTS=["algo-1"][0m
    [31mSM_MODULE_DIR=s3://sagemaker-us-east-1-545961142568/sagemaker-pytorch-2019-08-10-23-31-15-121/source/sourcedir.tar.gz[0m
    [31mSM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}[0m
    [31mSM_HPS={"epochs":10,"hidden_dim":200}[0m
    [31mSM_INPUT_DIR=/opt/ml/input[0m
    [31mSM_NUM_GPUS=1[0m
    [31mSM_OUTPUT_DATA_DIR=/opt/ml/output/data[0m
    [31mSM_LOG_LEVEL=20[0m
    [31mSM_NUM_CPUS=4[0m
    [31mSM_MODULE_NAME=train[0m
    [31mSM_CHANNEL_TRAINING=/opt/ml/input/data/training[0m
    [31mSM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}[0m
    [31mSM_HP_HIDDEN_DIM=200[0m
    [31mSM_OUTPUT_DIR=/opt/ml/output[0m
    [31mSM_CURRENT_HOST=algo-1[0m
    [31mSM_HP_EPOCHS=10[0m
    [31mSM_USER_ARGS=["--epochs","10","--hidden_dim","200"][0m
    [31mSM_CHANNELS=["training"][0m
    [31mSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training"},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"epochs":10,"hidden_dim":200},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","job_name":"sagemaker-pytorch-2019-08-10-23-31-15-121","log_level":20,"model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-east-1-545961142568/sagemaker-pytorch-2019-08-10-23-31-15-121/source/sourcedir.tar.gz","module_name":"train","network_interface_name":"eth0","num_cpus":4,"num_gpus":1,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"train.py"}[0m
    [31mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
    [0m
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
