
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

    When a movie shocks you with it's disturbing, brooding atmosphere, and grabs you by the throat with it's stunning cinematography, you just know that you have stumbled upon a treat, masterpiece of a film. Although with most modern movies, extremely enjoyable as some are, those that really shock you into focus are the strongest, and are the ones that are most critically acclaimed and mostly, stick with you for a life time. I say, proudly, that I am a fan of movies that disturb, not just horror movies, but those that send a vibe laden with foreboding. Movies like Breakdown and The Missing, that send a chill down your spine, making you think "holy crap, that could happen to me", and visually entice you, are up there with some of my favorite aspects in a movie. Because I am only 21, I did not grow up with actors like Burt Renyolds, Jon Voight and Ned Beatty, albeit I am familiar with them, I didn't watch them grow and proceed as actors, as opposed to actors now like Shia LaBouf and Justin Long. I must say, after the long hype and witnessing Deliverance for the first time, I was so admired by these veteran actors in a movie made more than 30 years ago, and still lives it's terror up in competition to modern movies. Burt Renyolds plays Lewis, the macho self appointed leader of a group of four friends on a canoe trip down a fictitious river before a dam is made, filling the whole wilderness in water. Renyolds' character is an experienced adventurer, sort of no nonsense, and filled with machismo. Witnessing him portray the tough guy, made me think differently about him as an actor, as i have only seen him as a seedy old guy or an angry politician. The dialog the director provides for his character gives him enough malice to be proved as a strong and even intimidating leader. Ronny Cox and Ned Beatty play as the novice adventurers, Drew and Bob respectively, joining in for the fun of a canoe trip. The actor that i thoroughly enjoyed watching was Jon Voight, once again I have only seen him as an older actor, however, unlike Renyolds, I have quite liked Voight's acting (and i don't regard Anaconda when I say that), for example the national treasure movies. Voight plays Ed, whom, like Lewis, is experienced in adventuring but is seen as a more reserved character, a reluctant hero/ leader and definitely lacks Lewis' machismo. The film basically opens up with the four driving into a small town while asking to find someone to drive their cars to the bottom of the river whilst they canoe the rapids and camp along the riverside. You immediately get a creepy vibe from the hillbilly characters we are introduced to, like the imbred kid who plays the infamous "Duelling Banjo's" at the start of the film with Ronny Cox's character Drew; and more so the two mountain men in the films pivotal and disturbing rape scene. As with all atmospheric movies, from this moment on, dread and confusion fills the characters as well as the audience and it is here we see the characters take shape and change form. The canoe trip that follows is expertly shot and it is from here the men fight against both human and nature's odds for survival. The film's cinematics do not let up, and I back that comment up with the scene in which Ed fights one of the rapist mountain men with a composite bow. As Ed falls on to one of his arrows and notices his enemy approaching him, cocks his rifle, only to shoot the floor as he falls with an arrow in his neck; was possible the greatest piece of cinematic shooting I have seen in a film. In wrapping up, Deliverance is one film, who's dread and atmosphere carry the mood across and to this date, remains one of the best films in cinematic history.
    1


Now that we have our data, I'll quickly tokenize the reviews using NTLK and Beautiful soup in the below function. Aside from tokenizing, I converted to lower case, removed stopwords, and stemmed the words. If you're unfamiliar, it just isolates the variants off a root words i.e. liked, likes, liking, etc.

Here it is below:


```python
print(words_from_review(train_X[101]))
```

    ['movi', 'shock', 'disturb', 'brood', 'atmospher', 'grab', 'throat', 'stun', 'cinematographi', 'know', 'stumbl', 'upon', 'treat', 'masterpiec', 'film', 'although', 'modern', 'movi', 'extrem', 'enjoy', 'realli', 'shock', 'focu', 'strongest', 'one', 'critic', 'acclaim', 'mostli', 'stick', 'life', 'time', 'say', 'proudli', 'fan', 'movi', 'disturb', 'horror', 'movi', 'send', 'vibe', 'laden', 'forebod', 'movi', 'like', 'breakdown', 'miss', 'send', 'chill', 'spine', 'make', 'think', 'holi', 'crap', 'could', 'happen', 'visual', 'entic', 'favorit', 'aspect', 'movi', '21', 'grow', 'actor', 'like', 'burt', 'renyold', 'jon', 'voight', 'ned', 'beatti', 'albeit', 'familiar', 'watch', 'grow', 'proceed', 'actor', 'oppos', 'actor', 'like', 'shia', 'labouf', 'justin', 'long', 'must', 'say', 'long', 'hype', 'wit', 'deliver', 'first', 'time', 'admir', 'veteran', 'actor', 'movi', 'made', '30', 'year', 'ago', 'still', 'live', 'terror', 'competit', 'modern', 'movi', 'burt', 'renyold', 'play', 'lewi', 'macho', 'self', 'appoint', 'leader', 'group', 'four', 'friend', 'cano', 'trip', 'fictiti', 'river', 'dam', 'made', 'fill', 'whole', 'wilder', 'water', 'renyold', 'charact', 'experienc', 'adventur', 'sort', 'nonsens', 'fill', 'machismo', 'wit', 'portray', 'tough', 'guy', 'made', 'think', 'differ', 'actor', 'seen', 'seedi', 'old', 'guy', 'angri', 'politician', 'dialog', 'director', 'provid', 'charact', 'give', 'enough', 'malic', 'prove', 'strong', 'even', 'intimid', 'leader', 'ronni', 'cox', 'ned', 'beatti', 'play', 'novic', 'adventur', 'drew', 'bob', 'respect', 'join', 'fun', 'cano', 'trip', 'actor', 'thoroughli', 'enjoy', 'watch', 'jon', 'voight', 'seen', 'older', 'actor', 'howev', 'unlik', 'renyold', 'quit', 'like', 'voight', 'act', 'regard', 'anaconda', 'say', 'exampl', 'nation', 'treasur', 'movi', 'voight', 'play', 'ed', 'like', 'lewi', 'experienc', 'adventur', 'seen', 'reserv', 'charact', 'reluct', 'hero', 'leader', 'definit', 'lack', 'lewi', 'machismo', 'film', 'basic', 'open', 'four', 'drive', 'small', 'town', 'ask', 'find', 'someon', 'drive', 'car', 'bottom', 'river', 'whilst', 'cano', 'rapid', 'camp', 'along', 'riversid', 'immedi', 'get', 'creepi', 'vibe', 'hillbilli', 'charact', 'introduc', 'like', 'imbr', 'kid', 'play', 'infam', 'duell', 'banjo', 'start', 'film', 'ronni', 'cox', 'charact', 'drew', 'two', 'mountain', 'men', 'film', 'pivot', 'disturb', 'rape', 'scene', 'atmospher', 'movi', 'moment', 'dread', 'confus', 'fill', 'charact', 'well', 'audienc', 'see', 'charact', 'take', 'shape', 'chang', 'form', 'cano', 'trip', 'follow', 'expertli', 'shot', 'men', 'fight', 'human', 'natur', 'odd', 'surviv', 'film', 'cinemat', 'let', 'back', 'comment', 'scene', 'ed', 'fight', 'one', 'rapist', 'mountain', 'men', 'composit', 'bow', 'ed', 'fall', 'one', 'arrow', 'notic', 'enemi', 'approach', 'cock', 'rifl', 'shoot', 'floor', 'fall', 'arrow', 'neck', 'possibl', 'greatest', 'piec', 'cinemat', 'shoot', 'seen', 'film', 'wrap', 'deliver', 'one', 'film', 'dread', 'atmospher', 'carri', 'mood', 'across', 'date', 'remain', 'one', 'best', 'film', 'cinemat', 'histori']



```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(words_from_review, data_train))
        #words_test = list(map(words_from_review, data_test))
        words_train = [words_from_review(review) for review in data_train]
        words_test = [words_from_review(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test
```


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

    {'movi': 2, 'film': 3, 'one': 4, 'like': 5, 'time': 6, 'good': 7, 'make': 8, 'charact': 9, 'get': 10, 'see': 11, 'watch': 12, 'stori': 13, 'even': 14, 'would': 15, 'realli': 16, 'well': 17, 'scene': 18, 'look': 19, 'show': 20, 'much': 21, 'end': 22, 'peopl': 23, 'bad': 24, 'go': 25, 'great': 26, 'also': 27, 'first': 28, 'love': 29, 'think': 30, 'way': 31, 'act': 32, 'play': 33, 'made': 34, 'thing': 35, 'could': 36, 'know': 37, 'say': 38, 'seem': 39, 'work': 40, 'plot': 41, 'two': 42, 'actor': 43, 'year': 44, 'come': 45, 'mani': 46, 'seen': 47, 'take': 48, 'life': 49, 'want': 50, 'never': 51, 'littl': 52, 'best': 53, 'tri': 54, 'man': 55, 'ever': 56, 'give': 57, 'better': 58, 'still': 59, 'perform': 60, 'find': 61, 'feel': 62, 'part': 63, 'back': 64, 'use': 65, 'someth': 66, 'director': 67, 'actual': 68, 'interest': 69, 'lot': 70, 'real': 71, 'old': 72, 'cast': 73, 'though': 74, 'live': 75, 'star': 76, 'enjoy': 77, 'guy': 78, 'anoth': 79, 'new': 80, 'role': 81, 'noth': 82, '10': 83, 'funni': 84, 'music': 85, 'point': 86, 'start': 87, 'set': 88, 'girl': 89, 'origin': 90, 'day': 91, 'world': 92, 'everi': 93, 'believ': 94, 'turn': 95, 'quit': 96, 'direct': 97, 'us': 98, 'thought': 99, 'fact': 100, 'minut': 101, 'horror': 102, 'kill': 103, 'action': 104, 'comedi': 105, 'pretti': 106, 'young': 107, 'wonder': 108, 'happen': 109, 'around': 110, 'got': 111, 'effect': 112, 'right': 113, 'long': 114, 'howev': 115, 'big': 116, 'line': 117, 'famili': 118, 'enough': 119, 'seri': 120, 'may': 121, 'need': 122, 'fan': 123, 'bit': 124, 'script': 125, 'beauti': 126, 'person': 127, 'becom': 128, 'without': 129, 'must': 130, 'alway': 131, 'friend': 132, 'tell': 133, 'reason': 134, 'saw': 135, 'last': 136, 'final': 137, 'kid': 138, 'almost': 139, 'put': 140, 'least': 141, 'sure': 142, 'done': 143, 'whole': 144, 'place': 145, 'complet': 146, 'kind': 147, 'differ': 148, 'expect': 149, 'shot': 150, 'far': 151, 'mean': 152, 'anyth': 153, 'book': 154, 'laugh': 155, 'might': 156, 'name': 157, 'sinc': 158, 'begin': 159, '2': 160, 'probabl': 161, 'woman': 162, 'help': 163, 'entertain': 164, 'let': 165, 'screen': 166, 'call': 167, 'tv': 168, 'moment': 169, 'away': 170, 'read': 171, 'yet': 172, 'rather': 173, 'worst': 174, 'run': 175, 'fun': 176, 'lead': 177, 'hard': 178, 'audienc': 179, 'idea': 180, 'anyon': 181, 'episod': 182, 'american': 183, 'found': 184, 'appear': 185, 'bore': 186, 'especi': 187, 'although': 188, 'hope': 189, 'cours': 190, 'keep': 191, 'anim': 192, 'job': 193, 'goe': 194, 'move': 195, 'sens': 196, 'dvd': 197, 'version': 198, 'war': 199, 'money': 200, 'someon': 201, 'mind': 202, 'mayb': 203, 'problem': 204, 'true': 205, 'hous': 206, 'everyth': 207, 'nice': 208, 'second': 209, 'rate': 210, 'three': 211, 'night': 212, 'follow': 213, 'face': 214, 'recommend': 215, 'main': 216, 'product': 217, 'worth': 218, 'leav': 219, 'human': 220, 'special': 221, 'excel': 222, 'togeth': 223, 'wast': 224, 'everyon': 225, 'sound': 226, 'john': 227, 'hand': 228, '1': 229, 'father': 230, 'later': 231, 'eye': 232, 'said': 233, 'view': 234, 'instead': 235, 'review': 236, 'boy': 237, 'high': 238, 'hour': 239, 'miss': 240, 'talk': 241, 'classic': 242, 'wife': 243, 'understand': 244, 'left': 245, 'care': 246, 'black': 247, 'death': 248, 'open': 249, 'murder': 250, 'write': 251, 'half': 252, 'head': 253, 'rememb': 254, 'chang': 255, 'viewer': 256, 'fight': 257, 'gener': 258, 'surpris': 259, 'short': 260, 'includ': 261, 'die': 262, 'fall': 263, 'less': 264, 'els': 265, 'entir': 266, 'piec': 267, 'involv': 268, 'pictur': 269, 'simpli': 270, 'top': 271, 'home': 272, 'power': 273, 'total': 274, 'usual': 275, 'budget': 276, 'attempt': 277, 'suppos': 278, 'releas': 279, 'hollywood': 280, 'terribl': 281, 'song': 282, 'men': 283, 'possibl': 284, 'featur': 285, 'portray': 286, 'disappoint': 287, 'poor': 288, '3': 289, 'coupl': 290, 'stupid': 291, 'camera': 292, 'dead': 293, 'wrong': 294, 'produc': 295, 'low': 296, 'either': 297, 'video': 298, 'aw': 299, 'definit': 300, 'except': 301, 'rest': 302, 'given': 303, 'absolut': 304, 'women': 305, 'lack': 306, 'word': 307, 'writer': 308, 'titl': 309, 'talent': 310, 'decid': 311, 'full': 312, 'perfect': 313, 'along': 314, 'style': 315, 'close': 316, 'truli': 317, 'school': 318, 'emot': 319, 'save': 320, 'age': 321, 'sex': 322, 'next': 323, 'bring': 324, 'mr': 325, 'case': 326, 'killer': 327, 'heart': 328, 'comment': 329, 'sort': 330, 'creat': 331, 'perhap': 332, 'came': 333, 'brother': 334, 'sever': 335, 'joke': 336, 'art': 337, 'dialogu': 338, 'game': 339, 'small': 340, 'base': 341, 'flick': 342, 'written': 343, 'sequenc': 344, 'meet': 345, 'earli': 346, 'often': 347, 'other': 348, 'mother': 349, 'develop': 350, 'humor': 351, 'actress': 352, 'consid': 353, 'dark': 354, 'guess': 355, 'amaz': 356, 'unfortun': 357, 'lost': 358, 'light': 359, 'exampl': 360, 'cinema': 361, 'drama': 362, 'ye': 363, 'white': 364, 'experi': 365, 'imagin': 366, 'mention': 367, 'stop': 368, 'natur': 369, 'forc': 370, 'manag': 371, 'felt': 372, 'present': 373, 'cut': 374, 'children': 375, 'fail': 376, 'son': 377, 'support': 378, 'qualiti': 379, 'car': 380, 'ask': 381, 'hit': 382, 'side': 383, 'voic': 384, 'extrem': 385, 'impress': 386, 'evil': 387, 'wors': 388, 'stand': 389, 'went': 390, 'certainli': 391, 'basic': 392, 'oh': 393, 'overal': 394, 'favorit': 395, 'horribl': 396, 'mysteri': 397, 'number': 398, 'type': 399, 'danc': 400, 'wait': 401, 'hero': 402, '5': 403, 'alreadi': 404, 'learn': 405, 'matter': 406, '4': 407, 'michael': 408, 'genr': 409, 'fine': 410, 'despit': 411, 'throughout': 412, 'walk': 413, 'success': 414, 'histori': 415, 'question': 416, 'zombi': 417, 'town': 418, 'realiz': 419, 'relationship': 420, 'child': 421, 'past': 422, 'daughter': 423, 'late': 424, 'b': 425, 'wish': 426, 'hate': 427, 'credit': 428, 'event': 429, 'theme': 430, 'touch': 431, 'citi': 432, 'today': 433, 'sometim': 434, 'behind': 435, 'god': 436, 'twist': 437, 'sit': 438, 'stay': 439, 'annoy': 440, 'deal': 441, 'abl': 442, 'rent': 443, 'pleas': 444, 'edit': 445, 'blood': 446, 'deserv': 447, 'anyway': 448, 'comic': 449, 'appar': 450, 'soon': 451, 'gave': 452, 'etc': 453, 'level': 454, 'slow': 455, 'chanc': 456, 'score': 457, 'bodi': 458, 'brilliant': 459, 'incred': 460, 'figur': 461, 'situat': 462, 'self': 463, 'major': 464, 'stuff': 465, 'decent': 466, 'element': 467, 'dream': 468, 'return': 469, 'obvious': 470, 'order': 471, 'continu': 472, 'pace': 473, 'ridicul': 474, 'happi': 475, 'highli': 476, 'group': 477, 'add': 478, 'thank': 479, 'ladi': 480, 'novel': 481, 'pain': 482, 'speak': 483, 'career': 484, 'shoot': 485, 'strang': 486, 'heard': 487, 'sad': 488, 'husband': 489, 'polic': 490, 'import': 491, 'break': 492, 'took': 493, 'cannot': 494, 'strong': 495, 'robert': 496, 'predict': 497, 'violenc': 498, 'hilari': 499, 'recent': 500, 'countri': 501, 'known': 502, 'particularli': 503, 'pick': 504, 'documentari': 505, 'season': 506, 'critic': 507, 'jame': 508, 'compar': 509, 'obviou': 510, 'alon': 511, 'told': 512, 'state': 513, 'rock': 514, 'visual': 515, 'theater': 516, 'exist': 517, 'offer': 518, 'opinion': 519, 'gore': 520, 'crap': 521, 'hold': 522, 'result': 523, 'realiti': 524, 'room': 525, 'hear': 526, 'clich': 527, 'effort': 528, 'thriller': 529, 'caus': 530, 'serious': 531, 'explain': 532, 'sequel': 533, 'king': 534, 'local': 535, 'ago': 536, 'hell': 537, 'none': 538, 'note': 539, 'allow': 540, 'sister': 541, 'david': 542, 'simpl': 543, 'femal': 544, 'deliv': 545, 'ok': 546, 'class': 547, 'convinc': 548, 'check': 549, 'suspens': 550, 'win': 551, 'buy': 552, 'oscar': 553, 'huge': 554, 'valu': 555, 'sexual': 556, 'cool': 557, 'scari': 558, 'excit': 559, 'similar': 560, 'apart': 561, 'provid': 562, 'exactli': 563, 'avoid': 564, 'shown': 565, 'seriou': 566, 'english': 567, 'whose': 568, 'taken': 569, 'cinematographi': 570, 'shock': 571, 'polit': 572, 'spoiler': 573, 'offic': 574, 'across': 575, 'middl': 576, 'pass': 577, 'street': 578, 'messag': 579, 'silli': 580, 'somewhat': 581, 'charm': 582, 'modern': 583, 'confus': 584, 'filmmak': 585, 'form': 586, 'tale': 587, 'singl': 588, 'jack': 589, 'mostli': 590, 'attent': 591, 'william': 592, 'carri': 593, 'sing': 594, 'five': 595, 'subject': 596, 'richard': 597, 'prove': 598, 'team': 599, 'stage': 600, 'unlik': 601, 'cop': 602, 'georg': 603, 'televis': 604, 'monster': 605, 'earth': 606, 'villain': 607, 'cover': 608, 'pay': 609, 'marri': 610, 'toward': 611, 'build': 612, 'pull': 613, 'parent': 614, 'due': 615, 'fill': 616, 'respect': 617, 'dialog': 618, 'four': 619, 'remind': 620, 'futur': 621, 'typic': 622, 'weak': 623, '7': 624, 'cheap': 625, 'intellig': 626, 'british': 627, 'atmospher': 628, 'clearli': 629, '80': 630, 'paul': 631, 'non': 632, 'dog': 633, '8': 634, 'knew': 635, 'artist': 636, 'fast': 637, 'crime': 638, 'easili': 639, 'escap': 640, 'adult': 641, 'doubt': 642, 'detail': 643, 'date': 644, 'member': 645, 'fire': 646, 'romant': 647, 'gun': 648, 'drive': 649, 'straight': 650, 'fit': 651, 'beyond': 652, 'attack': 653, 'imag': 654, 'upon': 655, 'posit': 656, 'whether': 657, 'fantast': 658, 'peter': 659, 'captur': 660, 'appreci': 661, 'aspect': 662, 'ten': 663, 'plan': 664, 'discov': 665, 'remain': 666, 'period': 667, 'near': 668, 'air': 669, 'realist': 670, 'mark': 671, 'red': 672, 'dull': 673, 'adapt': 674, 'within': 675, 'spend': 676, 'lose': 677, 'color': 678, 'materi': 679, 'chase': 680, 'mari': 681, 'storylin': 682, 'forget': 683, 'bunch': 684, 'clear': 685, 'lee': 686, 'victim': 687, 'nearli': 688, 'box': 689, 'york': 690, 'match': 691, 'inspir': 692, 'finish': 693, 'mess': 694, 'standard': 695, 'easi': 696, 'truth': 697, 'busi': 698, 'suffer': 699, 'dramat': 700, 'bill': 701, 'space': 702, 'western': 703, 'e': 704, 'list': 705, 'battl': 706, 'notic': 707, 'de': 708, 'french': 709, 'ad': 710, '9': 711, 'tom': 712, 'larg': 713, 'among': 714, 'eventu': 715, 'accept': 716, 'train': 717, 'agre': 718, 'spirit': 719, 'soundtrack': 720, 'third': 721, 'teenag': 722, 'adventur': 723, 'soldier': 724, 'famou': 725, 'drug': 726, 'sorri': 727, 'suggest': 728, 'babi': 729, 'normal': 730, 'cri': 731, 'ultim': 732, 'troubl': 733, 'contain': 734, 'certain': 735, 'cultur': 736, 'romanc': 737, 'rare': 738, 'lame': 739, 'somehow': 740, 'disney': 741, 'mix': 742, 'gone': 743, 'cartoon': 744, 'student': 745, 'fear': 746, 'reveal': 747, 'kept': 748, 'suck': 749, 'attract': 750, 'appeal': 751, 'premis': 752, 'design': 753, 'secret': 754, 'greatest': 755, 'shame': 756, 'throw': 757, 'copi': 758, 'scare': 759, 'wit': 760, 'admit': 761, 'america': 762, 'relat': 763, 'particular': 764, 'brought': 765, 'screenplay': 766, 'whatev': 767, 'pure': 768, '70': 769, 'harri': 770, 'averag': 771, 'master': 772, 'describ': 773, 'male': 774, 'treat': 775, '20': 776, 'fantasi': 777, 'issu': 778, 'warn': 779, 'inde': 780, 'background': 781, 'forward': 782, 'free': 783, 'project': 784, 'memor': 785, 'japanes': 786, 'poorli': 787, 'award': 788, 'locat': 789, 'potenti': 790, 'amus': 791, 'struggl': 792, 'weird': 793, 'magic': 794, 'societi': 795, 'okay': 796, 'accent': 797, 'imdb': 798, 'doctor': 799, 'hot': 800, 'water': 801, '30': 802, 'dr': 803, 'alien': 804, 'express': 805, 'odd': 806, 'choic': 807, 'crazi': 808, 'studio': 809, 'fiction': 810, 'control': 811, 'becam': 812, 'masterpiec': 813, 'fli': 814, 'difficult': 815, 'joe': 816, 'scream': 817, 'costum': 818, 'lover': 819, 'uniqu': 820, 'refer': 821, 'remak': 822, 'vampir': 823, 'girlfriend': 824, 'prison': 825, 'execut': 826, 'wear': 827, 'jump': 828, 'unless': 829, 'wood': 830, 'creepi': 831, 'cheesi': 832, 'superb': 833, 'otherwis': 834, 'parti': 835, 'roll': 836, 'ghost': 837, 'mad': 838, 'public': 839, 'depict': 840, 'earlier': 841, 'moral': 842, 'week': 843, 'jane': 844, 'badli': 845, 'dumb': 846, 'fi': 847, 'flaw': 848, 'grow': 849, 'deep': 850, 'sci': 851, 'maker': 852, 'cat': 853, 'connect': 854, 'footag': 855, 'older': 856, 'plenti': 857, 'bother': 858, 'outsid': 859, 'stick': 860, 'gay': 861, 'catch': 862, 'plu': 863, 'co': 864, 'popular': 865, 'equal': 866, 'social': 867, 'disturb': 868, 'quickli': 869, 'perfectli': 870, 'dress': 871, 'era': 872, '90': 873, 'mistak': 874, 'lie': 875, 'ride': 876, 'previou': 877, 'combin': 878, 'concept': 879, 'band': 880, 'surviv': 881, 'answer': 882, 'rich': 883, 'front': 884, 'sweet': 885, 'christma': 886, 'insid': 887, 'concern': 888, 'eat': 889, 'bare': 890, 'listen': 891, 'ben': 892, 'beat': 893, 'c': 894, 'serv': 895, 'term': 896, 'meant': 897, 'la': 898, 'german': 899, 'stereotyp': 900, 'hardli': 901, 'law': 902, 'innoc': 903, 'desper': 904, 'promis': 905, 'memori': 906, 'intent': 907, 'cute': 908, 'variou': 909, 'inform': 910, 'steal': 911, 'brain': 912, 'post': 913, 'tone': 914, 'island': 915, 'amount': 916, 'compani': 917, 'track': 918, 'nuditi': 919, 'claim': 920, 'store': 921, '50': 922, 'flat': 923, 'hair': 924, 'land': 925, 'univers': 926, 'kick': 927, 'scott': 928, 'danger': 929, 'fairli': 930, 'player': 931, 'plain': 932, 'crew': 933, 'step': 934, 'toni': 935, 'share': 936, 'tast': 937, 'centuri': 938, 'engag': 939, 'achiev': 940, 'cold': 941, 'travel': 942, 'record': 943, 'suit': 944, 'rip': 945, 'sadli': 946, 'manner': 947, 'tension': 948, 'wrote': 949, 'spot': 950, 'intens': 951, 'fascin': 952, 'familiar': 953, 'depth': 954, 'remark': 955, 'burn': 956, 'destroy': 957, 'histor': 958, 'sleep': 959, 'purpos': 960, 'languag': 961, 'ignor': 962, 'ruin': 963, 'delight': 964, 'unbeliev': 965, 'italian': 966, 'abil': 967, 'collect': 968, 'soul': 969, 'clever': 970, 'detect': 971, 'violent': 972, 'rape': 973, 'reach': 974, 'door': 975, 'liter': 976, 'trash': 977, 'scienc': 978, 'caught': 979, 'reveng': 980, 'commun': 981, 'creatur': 982, 'approach': 983, 'trip': 984, 'fashion': 985, 'intrigu': 986, 'paint': 987, 'introduc': 988, 'skill': 989, 'complex': 990, 'channel': 991, 'camp': 992, 'christian': 993, 'extra': 994, 'hole': 995, 'immedi': 996, 'mental': 997, 'ann': 998, 'limit': 999, 'million': 1000, 'slightli': 1001, 'mere': 1002, 'comput': 1003, '6': 1004, 'conclus': 1005, 'slasher': 1006, 'imposs': 1007, 'suddenli': 1008, 'neither': 1009, 'teen': 1010, 'crimin': 1011, 'physic': 1012, 'spent': 1013, 'nation': 1014, 'respons': 1015, 'planet': 1016, 'receiv': 1017, 'fake': 1018, 'blue': 1019, 'sick': 1020, 'bizarr': 1021, 'embarrass': 1022, 'indian': 1023, '15': 1024, 'ring': 1025, 'pop': 1026, 'drop': 1027, 'drag': 1028, 'haunt': 1029, 'pointless': 1030, 'suspect': 1031, 'search': 1032, 'edg': 1033, 'handl': 1034, 'biggest': 1035, 'common': 1036, 'hurt': 1037, 'faith': 1038, 'arriv': 1039, 'technic': 1040, 'angel': 1041, 'dad': 1042, 'genuin': 1043, 'f': 1044, 'awesom': 1045, 'solid': 1046, 'focu': 1047, 'van': 1048, 'colleg': 1049, 'former': 1050, 'count': 1051, 'tear': 1052, 'heavi': 1053, 'rais': 1054, 'wall': 1055, 'younger': 1056, 'visit': 1057, 'laughabl': 1058, 'fair': 1059, 'sign': 1060, 'excus': 1061, 'cult': 1062, 'tough': 1063, 'motion': 1064, 'key': 1065, 'desir': 1066, 'super': 1067, 'stun': 1068, 'addit': 1069, 'exploit': 1070, 'cloth': 1071, 'smith': 1072, 'tortur': 1073, 'race': 1074, 'davi': 1075, 'author': 1076, 'cross': 1077, 'jim': 1078, 'focus': 1079, 'compel': 1080, 'consist': 1081, 'minor': 1082, 'pathet': 1083, 'commit': 1084, 'chemistri': 1085, 'park': 1086, 'obsess': 1087, 'tradit': 1088, 'frank': 1089, 'grade': 1090, '60': 1091, 'asid': 1092, 'brutal': 1093, 'somewher': 1094, 'steve': 1095, 'grant': 1096, 'explor': 1097, 'opportun': 1098, 'u': 1099, 'rule': 1100, 'depress': 1101, 'honest': 1102, 'besid': 1103, 'anti': 1104, 'dub': 1105, 'trailer': 1106, 'intend': 1107, 'bar': 1108, 'longer': 1109, 'west': 1110, 'regard': 1111, 'scientist': 1112, 'judg': 1113, 'decad': 1114, 'silent': 1115, 'creativ': 1116, 'armi': 1117, 'wild': 1118, 'g': 1119, 'south': 1120, 'stewart': 1121, 'draw': 1122, 'road': 1123, 'govern': 1124, 'ex': 1125, 'boss': 1126, 'practic': 1127, 'motiv': 1128, 'club': 1129, 'surprisingli': 1130, 'gang': 1131, 'festiv': 1132, 'page': 1133, 'london': 1134, 'green': 1135, 'redeem': 1136, 'militari': 1137, 'aliv': 1138, 'idiot': 1139, 'display': 1140, 'machin': 1141, 'repeat': 1142, 'thrill': 1143, 'nobodi': 1144, 'yeah': 1145, '100': 1146, 'folk': 1147, '40': 1148, 'journey': 1149, 'garbag': 1150, 'smile': 1151, 'tire': 1152, 'ground': 1153, 'bought': 1154, 'mood': 1155, 'sam': 1156, 'stone': 1157, 'cost': 1158, 'mouth': 1159, 'noir': 1160, 'terrif': 1161, 'agent': 1162, 'requir': 1163, 'utterli': 1164, 'area': 1165, 'honestli': 1166, 'sexi': 1167, 'report': 1168, 'geniu': 1169, 'humour': 1170, 'investig': 1171, 'glad': 1172, 'enter': 1173, 'serial': 1174, 'occasion': 1175, 'passion': 1176, 'narr': 1177, 'marriag': 1178, 'climax': 1179, 'industri': 1180, 'studi': 1181, 'ship': 1182, 'center': 1183, 'demon': 1184, 'charli': 1185, 'nowher': 1186, 'bear': 1187, 'hors': 1188, 'loos': 1189, 'wow': 1190, 'hang': 1191, 'graphic': 1192, 'admir': 1193, 'giant': 1194, 'send': 1195, 'damn': 1196, 'loud': 1197, 'nake': 1198, 'subtl': 1199, 'profession': 1200, 'rel': 1201, 'blow': 1202, 'bottom': 1203, 'insult': 1204, 'batman': 1205, 'r': 1206, 'boyfriend': 1207, 'doubl': 1208, 'kelli': 1209, 'initi': 1210, 'frame': 1211, 'gem': 1212, 'opera': 1213, 'affect': 1214, 'challeng': 1215, 'drawn': 1216, 'church': 1217, 'cinemat': 1218, 'evid': 1219, 'seek': 1220, 'fulli': 1221, 'l': 1222, 'nightmar': 1223, 'j': 1224, 'essenti': 1225, 'conflict': 1226, 'arm': 1227, 'christoph': 1228, 'henri': 1229, 'wind': 1230, 'grace': 1231, 'narrat': 1232, 'witch': 1233, 'assum': 1234, 'hunt': 1235, 'push': 1236, 'chri': 1237, 'wise': 1238, 'repres': 1239, 'nomin': 1240, 'month': 1241, 'affair': 1242, 'hide': 1243, 'avail': 1244, 'sceneri': 1245, 'justic': 1246, 'smart': 1247, 'bond': 1248, 'thu': 1249, 'interview': 1250, 'outstand': 1251, 'flashback': 1252, 'satisfi': 1253, 'constantli': 1254, 'presenc': 1255, 'central': 1256, 'bed': 1257, 'iron': 1258, 'sell': 1259, 'content': 1260, 'everybodi': 1261, 'gag': 1262, 'slowli': 1263, 'hotel': 1264, 'hire': 1265, 'system': 1266, 'thrown': 1267, 'adam': 1268, 'charl': 1269, 'individu': 1270, 'hey': 1271, 'allen': 1272, 'jone': 1273, 'mediocr': 1274, 'lesson': 1275, 'billi': 1276, 'ray': 1277, 'photographi': 1278, 'cameo': 1279, 'fellow': 1280, 'pari': 1281, 'strike': 1282, 'absurd': 1283, 'brief': 1284, 'independ': 1285, 'rise': 1286, 'neg': 1287, 'impact': 1288, 'phone': 1289, 'model': 1290, 'born': 1291, 'ill': 1292, 'angl': 1293, 'fresh': 1294, 'spoil': 1295, 'abus': 1296, 'likabl': 1297, 'hill': 1298, 'discuss': 1299, 'sight': 1300, 'ahead': 1301, 'sent': 1302, 'photograph': 1303, 'blame': 1304, 'occur': 1305, 'shine': 1306, 'logic': 1307, 'mainli': 1308, 'bruce': 1309, 'commerci': 1310, 'skip': 1311, 'forev': 1312, 'surround': 1313, 'held': 1314, 'teacher': 1315, 'segment': 1316, 'zero': 1317, 'blond': 1318, 'satir': 1319, 'resembl': 1320, 'trap': 1321, 'summer': 1322, 'ball': 1323, 'queen': 1324, 'fool': 1325, 'six': 1326, 'tragedi': 1327, 'twice': 1328, 'sub': 1329, 'pack': 1330, 'reaction': 1331, 'bomb': 1332, 'hospit': 1333, 'protagonist': 1334, 'will': 1335, 'sport': 1336, 'mile': 1337, 'jerri': 1338, 'trust': 1339, 'drink': 1340, 'mom': 1341, 'vote': 1342, 'encount': 1343, 'plane': 1344, 'program': 1345, 'station': 1346, 'current': 1347, 'al': 1348, 'celebr': 1349, 'martin': 1350, 'choos': 1351, 'join': 1352, 'tragic': 1353, 'favourit': 1354, 'lord': 1355, 'round': 1356, 'field': 1357, 'vision': 1358, 'jean': 1359, 'robot': 1360, 'tie': 1361, 'arthur': 1362, 'random': 1363, 'fortun': 1364, 'roger': 1365, 'intern': 1366, 'dread': 1367, 'psycholog': 1368, 'improv': 1369, 'nonsens': 1370, 'prefer': 1371, 'epic': 1372, 'highlight': 1373, 'legend': 1374, 'formula': 1375, 'pleasur': 1376, 'tape': 1377, '11': 1378, 'dollar': 1379, 'thin': 1380, 'porn': 1381, 'gorgeou': 1382, 'fox': 1383, 'wide': 1384, 'object': 1385, 'influenc': 1386, 'buddi': 1387, 'ugli': 1388, 'ii': 1389, 'nasti': 1390, 'prepar': 1391, 'progress': 1392, 'reflect': 1393, 'supposedli': 1394, 'warm': 1395, 'youth': 1396, 'worthi': 1397, 'latter': 1398, 'unusu': 1399, 'length': 1400, 'crash': 1401, 'seven': 1402, 'childhood': 1403, 'superior': 1404, 'shop': 1405, 'theatr': 1406, 'remot': 1407, 'disgust': 1408, 'funniest': 1409, 'paid': 1410, 'pilot': 1411, 'convers': 1412, 'castl': 1413, 'fell': 1414, 'trick': 1415, 'gangster': 1416, 'disast': 1417, 'establish': 1418, 'rob': 1419, 'heaven': 1420, 'disappear': 1421, 'mine': 1422, 'suicid': 1423, 'ident': 1424, 'singer': 1425, 'forgotten': 1426, 'tend': 1427, 'heroin': 1428, 'decis': 1429, 'mask': 1430, 'partner': 1431, 'brian': 1432, 'recogn': 1433, 'alan': 1434, 'desert': 1435, 'p': 1436, 'sky': 1437, 'ms': 1438, 'thoroughli': 1439, 'stuck': 1440, 'accur': 1441, 'replac': 1442, 'market': 1443, 'danni': 1444, 'eddi': 1445, 'uncl': 1446, 'commentari': 1447, 'seemingli': 1448, 'andi': 1449, 'clue': 1450, 'jackson': 1451, 'devil': 1452, 'therefor': 1453, 'that': 1454, 'refus': 1455, 'pair': 1456, 'accid': 1457, 'ed': 1458, 'river': 1459, 'fault': 1460, 'unit': 1461, 'fate': 1462, 'afraid': 1463, 'tune': 1464, 'hidden': 1465, 'clean': 1466, 'russian': 1467, 'stephen': 1468, 'convey': 1469, 'instanc': 1470, 'quick': 1471, 'captain': 1472, 'irrit': 1473, 'test': 1474, 'readi': 1475, 'european': 1476, 'insan': 1477, 'frustrat': 1478, 'daniel': 1479, '1950': 1480, 'wed': 1481, 'chines': 1482, 'rescu': 1483, 'food': 1484, 'lock': 1485, 'angri': 1486, 'dirti': 1487, 'joy': 1488, 'steven': 1489, 'price': 1490, 'bland': 1491, 'cage': 1492, 'rang': 1493, 'anymor': 1494, 'wooden': 1495, 'jason': 1496, 'n': 1497, 'rush': 1498, 'news': 1499, '12': 1500, 'twenti': 1501, 'martial': 1502, 'worri': 1503, 'board': 1504, 'led': 1505, 'transform': 1506, 'cgi': 1507, 'hunter': 1508, 'symbol': 1509, 'sentiment': 1510, 'x': 1511, 'piti': 1512, 'onto': 1513, 'invent': 1514, 'johnni': 1515, 'process': 1516, 'explan': 1517, 'attitud': 1518, 'awar': 1519, 'owner': 1520, 'aim': 1521, 'floor': 1522, 'target': 1523, 'energi': 1524, 'necessari': 1525, 'favor': 1526, 'opposit': 1527, 'religi': 1528, 'insight': 1529, 'chick': 1530, 'window': 1531, 'blind': 1532, 'movement': 1533, 'deepli': 1534, 'research': 1535, 'possess': 1536, 'mountain': 1537, 'comparison': 1538, 'comed': 1539, 'rain': 1540, 'grand': 1541, 'whatsoev': 1542, 'began': 1543, 'mid': 1544, 'shadow': 1545, 'bank': 1546, 'parodi': 1547, 'princ': 1548, 'friendship': 1549, 'pre': 1550, 'weapon': 1551, 'taylor': 1552, 'credibl': 1553, 'dougla': 1554, 'teach': 1555, 'flesh': 1556, 'hint': 1557, 'terror': 1558, 'bloodi': 1559, 'protect': 1560, 'marvel': 1561, 'anybodi': 1562, 'drunk': 1563, 'superman': 1564, 'accord': 1565, 'leader': 1566, 'load': 1567, 'watchabl': 1568, 'brown': 1569, 'freddi': 1570, 'jeff': 1571, 'seat': 1572, 'tim': 1573, 'hitler': 1574, 'appropri': 1575, 'knock': 1576, 'charg': 1577, 'unknown': 1578, 'keaton': 1579, 'villag': 1580, 'media': 1581, 'england': 1582, 'unnecessari': 1583, 'empti': 1584, 'enemi': 1585, 'strength': 1586, 'wave': 1587, 'dare': 1588, 'craft': 1589, 'buck': 1590, 'perspect': 1591, 'utter': 1592, 'correct': 1593, 'kiss': 1594, 'ford': 1595, 'nativ': 1596, 'contrast': 1597, 'distract': 1598, 'magnific': 1599, 'chill': 1600, 'soap': 1601, 'anywher': 1602, 'nazi': 1603, 'knowledg': 1604, 'speed': 1605, '1980': 1606, 'ice': 1607, 'breath': 1608, 'mission': 1609, 'fred': 1610, 'crowd': 1611, 'moon': 1612, 'joan': 1613, 'jr': 1614, 'soft': 1615, 'frighten': 1616, '000': 1617, 'kate': 1618, 'dick': 1619, 'nick': 1620, 'hundr': 1621, 'dan': 1622, 'simon': 1623, 'radio': 1624, 'dozen': 1625, 'somebodi': 1626, 'academi': 1627, 'loss': 1628, 'shakespear': 1629, 'andrew': 1630, 'thousand': 1631, 'account': 1632, 'vehicl': 1633, 'quot': 1634, 'root': 1635, 'sum': 1636, '1970': 1637, 'convent': 1638, 'leg': 1639, 'behavior': 1640, 'gold': 1641, 'regular': 1642, 'demand': 1643, 'compet': 1644, 'pretenti': 1645, 'worker': 1646, 'lynch': 1647, 'candi': 1648, 'interpret': 1649, 'stretch': 1650, 'privat': 1651, 'explos': 1652, 'notabl': 1653, 'japan': 1654, 'debut': 1655, 'constant': 1656, 'tarzan': 1657, 'revolv': 1658, 'translat': 1659, 'spi': 1660, 'sea': 1661, 'prais': 1662, 'franc': 1663, 'jesu': 1664, 'technolog': 1665, 'failur': 1666, 'sat': 1667, 'threaten': 1668, 'ass': 1669, 'quiet': 1670, 'met': 1671, 'aid': 1672, 'toy': 1673, 'punch': 1674, 'kevin': 1675, 'higher': 1676, 'vh': 1677, 'interact': 1678, 'abandon': 1679, 'mike': 1680, 'separ': 1681, 'confront': 1682, 'command': 1683, 'bet': 1684, 'techniqu': 1685, 'gotten': 1686, 'site': 1687, 'servic': 1688, 'stunt': 1689, 'belong': 1690, 'recal': 1691, 'foot': 1692, 'freak': 1693, 'cabl': 1694, 'bug': 1695, 'bright': 1696, 'fu': 1697, 'capabl': 1698, 'african': 1699, 'jimmi': 1700, 'clark': 1701, 'stock': 1702, 'succeed': 1703, 'fat': 1704, 'presid': 1705, 'boat': 1706, 'structur': 1707, 'spanish': 1708, 'gene': 1709, 'kidnap': 1710, 'paper': 1711, 'factor': 1712, 'whilst': 1713, 'belief': 1714, 'realis': 1715, 'complic': 1716, 'educ': 1717, 'realism': 1718, 'witti': 1719, 'tree': 1720, 'bob': 1721, 'attend': 1722, 'finest': 1723, 'assist': 1724, 'broken': 1725, 'santa': 1726, 'v': 1727, 'determin': 1728, 'up': 1729, 'observ': 1730, 'smoke': 1731, 'depart': 1732, 'domin': 1733, 'lewi': 1734, 'hat': 1735, 'routin': 1736, 'fame': 1737, 'rubbish': 1738, 'oper': 1739, 'lone': 1740, 'foreign': 1741, 'safe': 1742, 'kinda': 1743, 'hook': 1744, 'morgan': 1745, 'advanc': 1746, 'rank': 1747, 'numer': 1748, 'washington': 1749, 'rose': 1750, 'vs': 1751, 'shallow': 1752, 'civil': 1753, 'werewolf': 1754, 'shape': 1755, 'gari': 1756, 'morn': 1757, 'kong': 1758, 'ordinari': 1759, 'winner': 1760, 'accomplish': 1761, 'whenev': 1762, 'grab': 1763, 'peac': 1764, 'virtual': 1765, 'h': 1766, 'offens': 1767, 'luck': 1768, 'complain': 1769, 'activ': 1770, 'bigger': 1771, 'patient': 1772, 'unfunni': 1773, 'contriv': 1774, 'welcom': 1775, 'pretend': 1776, 'dimension': 1777, 'trek': 1778, 'con': 1779, 'flash': 1780, 'eric': 1781, 'cain': 1782, 'lesbian': 1783, 'dri': 1784, 'code': 1785, 'wake': 1786, 'dancer': 1787, 'manipul': 1788, 'corrupt': 1789, 'albert': 1790, 'statu': 1791, 'guard': 1792, 'awkward': 1793, 'sourc': 1794, 'context': 1795, 'speech': 1796, 'gain': 1797, 'signific': 1798, 'clip': 1799, 'psycho': 1800, '13': 1801, 'anthoni': 1802, 'sean': 1803, 'corni': 1804, 'theatric': 1805, 'w': 1806, 'religion': 1807, 'advic': 1808, 'reli': 1809, 'priest': 1810, 'curiou': 1811, 'flow': 1812, 'addict': 1813, 'howard': 1814, 'asian': 1815, 'secur': 1816, 'jennif': 1817, 'specif': 1818, 'skin': 1819, 'promot': 1820, 'comfort': 1821, 'core': 1822, 'golden': 1823, 'luke': 1824, 'organ': 1825, 'cash': 1826, 'cheat': 1827, 'lucki': 1828, 'lower': 1829, 'dislik': 1830, 'associ': 1831, 'frequent': 1832, 'spell': 1833, 'degre': 1834, 'frankli': 1835, 'devic': 1836, 'balanc': 1837, 'regret': 1838, 'contribut': 1839, 'wing': 1840, 'print': 1841, 'forgiv': 1842, 'lake': 1843, 'sake': 1844, 'betti': 1845, 'mass': 1846, 'thoma': 1847, 'crack': 1848, 'gordon': 1849, 'unexpect': 1850, 'construct': 1851, 'amateur': 1852, 'invit': 1853, 'unfold': 1854, 'categori': 1855, 'depend': 1856, 'grown': 1857, 'anna': 1858, 'walter': 1859, 'matur': 1860, 'grew': 1861, 'intellectu': 1862, 'honor': 1863, 'condit': 1864, 'spectacular': 1865, 'veteran': 1866, 'mirror': 1867, 'sudden': 1868, 'sole': 1869, 'demonstr': 1870, 'experienc': 1871, 'robin': 1872, 'liner': 1873, 'card': 1874, 'grip': 1875, 'freedom': 1876, 'gift': 1877, 'overli': 1878, 'meanwhil': 1879, 'unabl': 1880, 'theori': 1881, 'oliv': 1882, 'colour': 1883, 'crappi': 1884, 'section': 1885, 'circumst': 1886, 'sheriff': 1887, 'subtitl': 1888, 'drew': 1889, 'brilliantli': 1890, 'altern': 1891, 'sheer': 1892, 'path': 1893, 'pile': 1894, 'cook': 1895, 'laughter': 1896, 'matt': 1897, 'parker': 1898, 'defin': 1899, 'hall': 1900, 'wander': 1901, 'accident': 1902, 'treatment': 1903, 'lawyer': 1904, 'sinatra': 1905, 'relief': 1906, 'captiv': 1907, 'dragon': 1908, 'hank': 1909, 'halloween': 1910, 'gratuit': 1911, 'moor': 1912, 'unintent': 1913, 'kung': 1914, 'wound': 1915, 'broadway': 1916, 'barbara': 1917, 'k': 1918, 'jacki': 1919, 'wayn': 1920, 'cowboy': 1921, 'spoof': 1922, 'statement': 1923, 'winter': 1924, 'canadian': 1925, 'surreal': 1926, 'cheer': 1927, 'compos': 1928, 'gonna': 1929, 'fish': 1930, 'fare': 1931, 'treasur': 1932, 'emerg': 1933, 'woodi': 1934, 'victor': 1935, 'sensit': 1936, 'unrealist': 1937, 'neighbor': 1938, 'ran': 1939, 'driven': 1940, 'sympathet': 1941, 'topic': 1942, 'glass': 1943, 'overlook': 1944, 'expos': 1945, 'menac': 1946, 'authent': 1947, 'handsom': 1948, 'michel': 1949, 'gross': 1950, 'chief': 1951, 'ancient': 1952, 'comedian': 1953, 'stranger': 1954, 'cinderella': 1955, 'feet': 1956, 'built': 1957, 'network': 1958, 'russel': 1959, 'nevertheless': 1960, 'contemporari': 1961, 'pleasant': 1962, 'letter': 1963, 'earn': 1964, 'consider': 1965, 'underr': 1966, 'blockbust': 1967, 'miser': 1968, 'endless': 1969, 'gori': 1970, 'solv': 1971, 'switch': 1972, 'brook': 1973, 'edward': 1974, 'convict': 1975, 'joseph': 1976, 'bullet': 1977, 'virgin': 1978, 'victoria': 1979, 'cynic': 1980, 'chosen': 1981, 'scale': 1982, 'scenario': 1983, 'alex': 1984, '0': 1985, 'com': 1986, 'outrag': 1987, 'gut': 1988, 'curs': 1989, 'sword': 1990, 'screenwrit': 1991, 'monkey': 1992, 'uk': 1993, 'juli': 1994, 'driver': 1995, 'proper': 1996, 'substanc': 1997, 'wrap': 1998, 'indic': 1999, 'par': 2000, 'remov': 2001, 'bird': 2002, 'court': 2003, 'grave': 2004, 'naiv': 2005, 'consequ': 2006, 'rental': 2007, 'loser': 2008, 'advertis': 2009, 'inevit': 2010, 'nanci': 2011, 'roy': 2012, 'fatal': 2013, 'brave': 2014, 'germani': 2015, 'bridg': 2016, 'le': 2017, 'slap': 2018, 'invis': 2019, 'footbal': 2020, 'loui': 2021, 'ador': 2022, 'provok': 2023, 'anger': 2024, 'anderson': 2025, 'alcohol': 2026, 'chan': 2027, 'ryan': 2028, 'willi': 2029, 'stumbl': 2030, 'professor': 2031, 'australian': 2032, 'assassin': 2033, 'bat': 2034, '1930': 2035, 'sharp': 2036, 'patrick': 2037, 'trilog': 2038, 'eight': 2039, 'cell': 2040, 'heck': 2041, 'deni': 2042, 'amateurish': 2043, 'refresh': 2044, 'ape': 2045, 'strongli': 2046, 'lousi': 2047, 'liber': 2048, 'saturday': 2049, 'sin': 2050, 'resid': 2051, 'vagu': 2052, 'justifi': 2053, 'san': 2054, 'indi': 2055, 'terrifi': 2056, 'defeat': 2057, 'creator': 2058, 'mini': 2059, 'reput': 2060, 'sympathi': 2061, 'expert': 2062, 'tabl': 2063, 'endur': 2064, 'prevent': 2065, 'tediou': 2066, 'task': 2067, 'trial': 2068, 'imit': 2069, 'offend': 2070, 'employ': 2071, 'basebal': 2072, 'rival': 2073, 'che': 2074, 'pitch': 2075, 'beach': 2076, 'fairi': 2077, 'complaint': 2078, 'dig': 2079, 'max': 2080, 'europ': 2081, 'weekend': 2082, 'murphi': 2083, 'format': 2084, 'risk': 2085, 'purchas': 2086, 'reminisc': 2087, 'harsh': 2088, 'bite': 2089, 'glimps': 2090, 'titan': 2091, 'tini': 2092, 'hype': 2093, 'nois': 2094, 'powel': 2095, 'till': 2096, 'north': 2097, '14': 2098, 'prime': 2099, 'strip': 2100, 'fals': 2101, 'asleep': 2102, 'revel': 2103, 'texa': 2104, 'africa': 2105, 'descript': 2106, 'destruct': 2107, 'sitcom': 2108, 'spin': 2109, 'arrest': 2110, 'excess': 2111, 'inner': 2112, 'surfac': 2113, 'semi': 2114, 'uninterest': 2115, 'argu': 2116, 'twin': 2117, 'massiv': 2118, 'dinosaur': 2119, 'maintain': 2120, 'hitchcock': 2121, 'controversi': 2122, 'makeup': 2123, 'melodrama': 2124, 'stare': 2125, 'kim': 2126, 'ludicr': 2127, 'insist': 2128, 'ideal': 2129, 'reject': 2130, 'expens': 2131, 'atroci': 2132, 'ala': 2133, 'forest': 2134, 'host': 2135, 'nail': 2136, 'ga': 2137, 'erot': 2138, 'press': 2139, 'subplot': 2140, 'supernatur': 2141, 'columbo': 2142, 'dude': 2143, 'cant': 2144, 'presum': 2145, 'identifi': 2146, 'notch': 2147, 'closer': 2148, 'method': 2149, 'forgett': 2150, 'character': 2151, 'plagu': 2152, 'guest': 2153, 'crude': 2154, 'foster': 2155, 'princess': 2156, 'ear': 2157, 'beast': 2158, 'lion': 2159, 'border': 2160, 'landscap': 2161, 'previous': 2162, 'pacino': 2163, 'urban': 2164, 'aunt': 2165, 'birth': 2166, 'jungl': 2167, 'storytel': 2168, 'bound': 2169, 'accus': 2170, 'damag': 2171, 'chose': 2172, 'doll': 2173, 'thirti': 2174, 'emma': 2175, 'jess': 2176, 'guid': 2177, 'nude': 2178, 'propaganda': 2179, 'whoever': 2180, '25': 2181, 'warrior': 2182, 'mainstream': 2183, 'mate': 2184, 'pet': 2185, 'size': 2186, 'poster': 2187, 'merit': 2188, 'cooper': 2189, 'latest': 2190, 'upset': 2191, 'deadli': 2192, 'gritti': 2193, 'friday': 2194, 'exact': 2195, 'settl': 2196, 'citizen': 2197, 'sun': 2198, 'blend': 2199, 'ton': 2200, 'popul': 2201, 'rough': 2202, 'warner': 2203, 'wilson': 2204, 'corps': 2205, 'contact': 2206, 'buff': 2207, '1990': 2208, 'contest': 2209, 'rat': 2210, 'alic': 2211, 'metal': 2212, 'select': 2213, 'environ': 2214, 'mgm': 2215, 'bu': 2216, 'overcom': 2217, 'widow': 2218, 'pitt': 2219, 'ted': 2220, 'particip': 2221, 'revolut': 2222, 'guilti': 2223, 'link': 2224, 'lift': 2225, 'corpor': 2226, 'exagger': 2227, 'prostitut': 2228, 'accompani': 2229, '1960': 2230, 'johnson': 2231, 'matrix': 2232, 'moron': 2233, 'corner': 2234, 'afternoon': 2235, 'multipl': 2236, 'holm': 2237, 'hood': 2238, 'instal': 2239, 'sincer': 2240, 'doom': 2241, 'leagu': 2242, 'friendli': 2243, 'clair': 2244, 'hip': 2245, 'examin': 2246, 'string': 2247, 'defend': 2248, 'advis': 2249, 'blah': 2250, 'sunday': 2251, 'grim': 2252, 'junk': 2253, 'lugosi': 2254, 'irish': 2255, 'aka': 2256, 'campi': 2257, 'tight': 2258, 'icon': 2259, 'shut': 2260, 'shake': 2261, 'rachel': 2262, 'pro': 2263, 'confid': 2264, 'varieti': 2265, 'attach': 2266, 'denni': 2267, 'medic': 2268, 'directli': 2269, 'sullivan': 2270, 'jaw': 2271, 'goal': 2272, 'mexican': 2273, 'legendari': 2274, 'sarah': 2275, 'courag': 2276, 'prior': 2277, 'dean': 2278, 'breast': 2279, 'duke': 2280, 'terrorist': 2281, 'vietnam': 2282, 'sentenc': 2283, 'bourn': 2284, 'truck': 2285, 'hong': 2286, 'split': 2287, 'entri': 2288, 'nose': 2289, 'yell': 2290, 'behav': 2291, 'proceed': 2292, 'donald': 2293, 'un': 2294, 'everywher': 2295, 'confess': 2296, 'gather': 2297, 'jerk': 2298, 'borrow': 2299, 'buri': 2300, 'lifetim': 2301, 'crush': 2302, 'unconvinc': 2303, 'swim': 2304, 'stolen': 2305, 'forth': 2306, 'concentr': 2307, 'julia': 2308, 'california': 2309, 'turkey': 2310, 'pan': 2311, 'deliveri': 2312, 'spite': 2313, 'lip': 2314, 'reward': 2315, 'proud': 2316, 'freeman': 2317, 'downright': 2318, 'flight': 2319, 'offici': 2320, 'quest': 2321, 'china': 2322, 'hoffman': 2323, 'notori': 2324, 'betray': 2325, 'jail': 2326, 'fade': 2327, 'worthwhil': 2328, 'jon': 2329, 'sir': 2330, 'sink': 2331, 'fabul': 2332, 'encourag': 2333, 'inept': 2334, 'lazi': 2335, 'imageri': 2336, 'susan': 2337, 'branagh': 2338, 'cousin': 2339, 'storm': 2340, 'lisa': 2341, 'survivor': 2342, 'retard': 2343, 'bag': 2344, 'relev': 2345, 'teeth': 2346, 'shower': 2347, 'bell': 2348, 'stab': 2349, 'finger': 2350, 'summari': 2351, 'toler': 2352, 'alright': 2353, 'trade': 2354, 'facial': 2355, 'mexico': 2356, 'tremend': 2357, 'quirki': 2358, 'bride': 2359, 'shark': 2360, 'hugh': 2361, 'hyster': 2362, 'ha': 2363, 'von': 2364, 'blown': 2365, 'bitter': 2366, 'pose': 2367, 'cruel': 2368, 'larri': 2369, 'ron': 2370, 'scheme': 2371, 'ned': 2372, 'afterward': 2373, 'address': 2374, 'christ': 2375, 'bone': 2376, 'pursu': 2377, 'tour': 2378, 'feed': 2379, 'beg': 2380, 'snake': 2381, 'traci': 2382, 'thumb': 2383, 'screw': 2384, 'distinct': 2385, 'swear': 2386, 'occas': 2387, 'chair': 2388, 'stomach': 2389, 'mechan': 2390, 'obscur': 2391, 'raw': 2392, 'photo': 2393, 'heavili': 2394, 'southern': 2395, 'hardi': 2396, 'necessarili': 2397, 'gruesom': 2398, 'holiday': 2399, 'resist': 2400, 'sidney': 2401, 'chain': 2402, 'argument': 2403, 'cabin': 2404, 'render': 2405, 'satan': 2406, 'india': 2407, 'racist': 2408, 'philip': 2409, 'understood': 2410, 'indulg': 2411, 'lay': 2412, 'stalk': 2413, 'belov': 2414, 'outfit': 2415, 'tongu': 2416, 'obnoxi': 2417, 'forgot': 2418, 'midnight': 2419, 'pregnant': 2420, 'fourth': 2421, 'integr': 2422, 'magazin': 2423, 'ticket': 2424, 'deeper': 2425, 'slapstick': 2426, 'garden': 2427, 'inhabit': 2428, 'restor': 2429, 'carol': 2430, '17': 2431, 'incid': 2432, 'shoe': 2433, 'devot': 2434, 'lincoln': 2435, 'brad': 2436, 'underground': 2437, 'benefit': 2438, 'disbelief': 2439, 'divorc': 2440, 'lili': 2441, 'guarante': 2442, 'anticip': 2443, 'elizabeth': 2444, 'maria': 2445, 'sandler': 2446, 'bbc': 2447, 'princip': 2448, 'amazingli': 2449, 'cring': 2450, 'explod': 2451, 'mildli': 2452, 'creation': 2453, 'capit': 2454, 'slave': 2455, 'greater': 2456, 'lesli': 2457, 'extraordinari': 2458, 'halfway': 2459, 'introduct': 2460, 'funnier': 2461, 'enhanc': 2462, 'tap': 2463, 'punish': 2464, 'advantag': 2465, 'text': 2466, 'overwhelm': 2467, 'transfer': 2468, 'wreck': 2469, 'extent': 2470, 'preview': 2471, 'lane': 2472, 'dynam': 2473, 'plant': 2474, 'lo': 2475, 'east': 2476, 'horrif': 2477, 'error': 2478, 'deliber': 2479, 'jessica': 2480, 'miscast': 2481, 'basi': 2482, '2000': 2483, 'vincent': 2484, 'ensu': 2485, 'sophist': 2486, 'miller': 2487, 'appli': 2488, 'homosexu': 2489, 'vacat': 2490, 'steel': 2491, 'via': 2492, 'uncomfort': 2493, 'reed': 2494, 'spoken': 2495, 'elev': 2496, 'measur': 2497, 'sleazi': 2498, 'bollywood': 2499, 'extend': 2500, 'mansion': 2501, 'assign': 2502, 'alter': 2503, 'beer': 2504, 'daili': 2505, 'goofi': 2506, 'stanley': 2507, 'savag': 2508, 'conceiv': 2509, 'mous': 2510, 'hippi': 2511, 'breathtak': 2512, 'melt': 2513, 'fix': 2514, 'cathol': 2515, 'overact': 2516, 'blair': 2517, 'dentist': 2518, 'sacrific': 2519, 'everyday': 2520, 'subsequ': 2521, 'properli': 2522, 'succe': 2523, 'nowaday': 2524, 'inspector': 2525, 'oppos': 2526, 'carpent': 2527, 'burt': 2528, 'neck': 2529, 'massacr': 2530, 'circl': 2531, 'block': 2532, 'laura': 2533, 'pool': 2534, 'fallen': 2535, 'seagal': 2536, 'concert': 2537, 'mob': 2538, 'access': 2539, 'lesser': 2540, 'fay': 2541, 'portrait': 2542, 'grey': 2543, 'christi': 2544, 'sinist': 2545, 'chees': 2546, 'jake': 2547, 'relax': 2548, 'react': 2549, 'competit': 2550, 'usa': 2551, 'jewish': 2552, 'isol': 2553, 'chop': 2554, 'suitabl': 2555, 'nonetheless': 2556, 'nine': 2557, 'immens': 2558, 'lyric': 2559, 'creep': 2560, '2006': 2561, 'appal': 2562, 'ironi': 2563, 'stink': 2564, 'spiritu': 2565, 'retir': 2566, 'user': 2567, 'nut': 2568, 'spring': 2569, 'shirt': 2570, 'needless': 2571, 'rage': 2572, 'luci': 2573, 'adopt': 2574, 'sold': 2575, 'franchis': 2576, 'reduc': 2577, 'showcas': 2578, 'navi': 2579, 'uninspir': 2580, 'per': 2581, 'nurs': 2582, 'bulli': 2583, 'bath': 2584, 'zone': 2585, 'asham': 2586, 'jay': 2587, 'digit': 2588, 'stanwyck': 2589, 'amongst': 2590, 'illustr': 2591, 'broadcast': 2592, '1940': 2593, 'oddli': 2594, 'laid': 2595, '2001': 2596, 'upper': 2597, 'sutherland': 2598, 'disguis': 2599, 'throat': 2600, 'fulfil': 2601, 'baker': 2602, 'stylish': 2603, 'aspir': 2604, 'brando': 2605, 'pride': 2606, 'wanna': 2607, 'em': 2608, 'neighborhood': 2609, 'impli': 2610, '18': 2611, 'endear': 2612, 'nobl': 2613, 'thief': 2614, 'wwii': 2615, 'pound': 2616, '16': 2617, 'cinematograph': 2618, 'dinner': 2619, 'diseas': 2620, 'shift': 2621, 'bo': 2622, 'prop': 2623, 'rochest': 2624, 'tens': 2625, 'dawn': 2626, 'albeit': 2627, 'distribut': 2628, 'shoulder': 2629, 'bett': 2630, 'coher': 2631, 'knife': 2632, 'matthau': 2633, 'function': 2634, 'shout': 2635, 'surf': 2636, 'contract': 2637, 'snow': 2638, 'rebel': 2639, 'silenc': 2640, 'wash': 2641, 'forti': 2642, 'poignant': 2643, 'duti': 2644, 'proof': 2645, 'mindless': 2646, 'internet': 2647, 'height': 2648, 'derek': 2649, 'widmark': 2650, 'henc': 2651, 'cancel': 2652, 'heat': 2653, 'eeri': 2654, 'silver': 2655, 'cannib': 2656, 'horrend': 2657, 'reunion': 2658, 'elvira': 2659, 'instinct': 2660, 'chuck': 2661, 'elvi': 2662, 'glori': 2663, 'absorb': 2664, 'innov': 2665, 'premier': 2666, 'alik': 2667, 'incoher': 2668, 'repetit': 2669, 'pie': 2670, 'greatli': 2671, 'spielberg': 2672, 'etern': 2673, 'torn': 2674, 'neat': 2675, 'musician': 2676, 'mill': 2677, 'announc': 2678, 'nelson': 2679, 'homag': 2680, 'precis': 2681, 'lovabl': 2682, 'infam': 2683, 'wealthi': 2684, 'diamond': 2685, 'fbi': 2686, 'horrifi': 2687, 'crisi': 2688, 'trite': 2689, 'redempt': 2690, 'bang': 2691, 'blank': 2692, 'itali': 2693, 'britain': 2694, 'racism': 2695, 'burton': 2696, 'parallel': 2697, 'ensembl': 2698, 'flop': 2699, 'chaplin': 2700, 'hammer': 2701, 'pat': 2702, 'helen': 2703, 'happili': 2704, 'wilder': 2705, 'dedic': 2706, 'streisand': 2707, 'resolut': 2708, 'disagre': 2709, 'plastic': 2710, 'conclud': 2711, 'st': 2712, 'oil': 2713, 'factori': 2714, 'broke': 2715, 'cube': 2716, 'carter': 2717, 'mar': 2718, 'triumph': 2719, 'weight': 2720, 'vega': 2721, 'own': 2722, 'fighter': 2723, 'row': 2724, 'climb': 2725, 'march': 2726, 'bush': 2727, 'chuckl': 2728, 'rocket': 2729, 'unforgett': 2730, 'mst3k': 2731, 'boot': 2732, 'thug': 2733, 'enorm': 2734, 'kurt': 2735, 'lust': 2736, 'sensibl': 2737, 'meaning': 2738, 'spare': 2739, 'luca': 2740, 'dump': 2741, 'dane': 2742, 'wherea': 2743, 'fifti': 2744, 'engin': 2745, 'dear': 2746, 'bobbi': 2747, 'stress': 2748, 'butt': 2749, 'adequ': 2750, 'brand': 2751, 'caricatur': 2752, 'karloff': 2753, 'threat': 2754, 'difficulti': 2755, 'arnold': 2756, 'rap': 2757, 'elabor': 2758, 'secretari': 2759, 'hamlet': 2760, 'flynn': 2761, 'journalist': 2762, 'polish': 2763, 'arrog': 2764, 'ralph': 2765, 'barri': 2766, 'ego': 2767, 'fest': 2768, 'homeless': 2769, 'swing': 2770, 'conspiraci': 2771, 'puppet': 2772, 'grate': 2773, 'tool': 2774, 'induc': 2775, 'spike': 2776, 'fanci': 2777, 'float': 2778, 'resort': 2779, 'simpson': 2780, 'arrang': 2781, 'unbear': 2782, 'choreograph': 2783, 'exercis': 2784, 'guilt': 2785, 'cruis': 2786, 'basement': 2787, 'pig': 2788, 'tribut': 2789, 'phillip': 2790, 'muppet': 2791, 'boll': 2792, 'editor': 2793, 'slip': 2794, 'item': 2795, 'medium': 2796, 'toilet': 2797, 'puzzl': 2798, 'fianc': 2799, 'babe': 2800, '24': 2801, 'tower': 2802, 'layer': 2803, 'document': 2804, 'file': 2805, 'korean': 2806, 'scarecrow': 2807, 'stan': 2808, 'ward': 2809, 'ham': 2810, 'glover': 2811, 'inexplic': 2812, 'philosoph': 2813, 'transit': 2814, 'assur': 2815, 'orient': 2816, 'doc': 2817, 'territori': 2818, 'persona': 2819, 'slaughter': 2820, 'librari': 2821, 'portion': 2822, 'denzel': 2823, 'catherin': 2824, 'minim': 2825, 'superfici': 2826, 'larger': 2827, 'spark': 2828, 'dorothi': 2829, 'wolf': 2830, 'sneak': 2831, 'pg': 2832, 'jeremi': 2833, 'jet': 2834, 'curti': 2835, 'shi': 2836, 'boredom': 2837, 'owe': 2838, 'ban': 2839, 'walken': 2840, 'financi': 2841, 'profound': 2842, 'hudson': 2843, 'metaphor': 2844, 'multi': 2845, 'backdrop': 2846, 'cusack': 2847, 'ambigu': 2848, 'eleph': 2849, 'whale': 2850, 'hack': 2851, 'elsewher': 2852, 'viru': 2853, 'notion': 2854, 'ultra': 2855, 'union': 2856, '2005': 2857, 'implaus': 2858, 'rave': 2859, 'birthday': 2860, 'stiff': 2861, 'gadget': 2862, 'bibl': 2863, 'hawk': 2864, 'poison': 2865, 'afford': 2866, 'newspap': 2867, 'slight': 2868, 'reader': 2869, 'canada': 2870, '1st': 2871, 'deriv': 2872, 'lloyd': 2873, 'urg': 2874, 'eastwood': 2875, 'eva': 2876, 'squar': 2877, 'pad': 2878, 'disc': 2879, 'distanc': 2880, 'superhero': 2881, 'huh': 2882, 'drown': 2883, 'skit': 2884, 'montag': 2885, 'restaur': 2886, 'cure': 2887, 'sadist': 2888, 'health': 2889, 'charisma': 2890, 'spread': 2891, 'button': 2892, 'essenc': 2893, 'heston': 2894, 'maniac': 2895, 'scoobi': 2896, 'lab': 2897, 'peak': 2898, 'gradual': 2899, 'estat': 2900, 'dealt': 2901, 'invest': 2902, 'godfath': 2903, 'companion': 2904, 'fetch': 2905, 'muslim': 2906, 'cup': 2907, 'kane': 2908, 'ritter': 2909, 'countless': 2910, 'subtleti': 2911, 'gothic': 2912, 'tea': 2913, 'alli': 2914, 'miik': 2915, 'servant': 2916, 'charismat': 2917, 'elect': 2918, 'electr': 2919, 'heroic': 2920, 'iii': 2921, 'briefli': 2922, 'salli': 2923, 'neil': 2924, 'nuanc': 2925, 'cole': 2926, 'admittedli': 2927, 'reel': 2928, 'ingredi': 2929, 'tender': 2930, 'wannab': 2931, 'resourc': 2932, 'toss': 2933, 'grandmoth': 2934, 'bud': 2935, 'stood': 2936, 'carrey': 2937, 'mafia': 2938, 'mild': 2939, 'shall': 2940, 'punk': 2941, 'label': 2942, 'stronger': 2943, 'gate': 2944, 'dawson': 2945, 'pit': 2946, 'reev': 2947, 'kubrick': 2948, 'pauli': 2949, 'poverti': 2950, 'fond': 2951, 'assault': 2952, 'burst': 2953, 'smash': 2954, 'tag': 2955, 'terri': 2956, 'ian': 2957, 'cardboard': 2958, 'cox': 2959, 'useless': 2960, 'easier': 2961, 'updat': 2962, 'smooth': 2963, 'outcom': 2964, 'bakshi': 2965, 'astair': 2966, 'divers': 2967, 'exchang': 2968, 'resolv': 2969, 'vulner': 2970, 'qualifi': 2971, 'vari': 2972, 'rex': 2973, 'coincid': 2974, 'fist': 2975, 'increasingli': 2976, '2002': 2977, 'melodramat': 2978, 'samurai': 2979, 'sketch': 2980, 'luckili': 2981, 'insert': 2982, 'scratch': 2983, 'suspend': 2984, 'brillianc': 2985, 'blast': 2986, 'conveni': 2987, 'reynold': 2988, 'be': 2989, 'templ': 2990, 'tame': 2991, 'pin': 2992, 'ambiti': 2993, 'hamilton': 2994, 'strictli': 2995, 'seventi': 2996, 'gotta': 2997, 'jami': 2998, 'meat': 2999, 'matthew': 3000, 'nuclear': 3001, 'coach': 3002, 'farm': 3003, 'fisher': 3004, 'soprano': 3005, 'walker': 3006, 'joey': 3007, 'timeless': 3008, 'spooki': 3009, 'grasp': 3010, 'butcher': 3011, 'clock': 3012, 'kudo': 3013, 'struck': 3014, 'instantli': 3015, 'convolut': 3016, 'revers': 3017, 'monk': 3018, 'discoveri': 3019, 'ninja': 3020, 'brosnan': 3021, 'recreat': 3022, 'worthless': 3023, 'cave': 3024, 'closet': 3025, 'empir': 3026, 'eccentr': 3027, 'miracl': 3028, 'cliff': 3029, 'pal': 3030, 'declar': 3031, 'sidekick': 3032, 'sloppi': 3033, 'inconsist': 3034, 'seller': 3035, 'bleak': 3036, 'communist': 3037, 'fifteen': 3038, 'selfish': 3039, 'mitchel': 3040, 'norman': 3041, 'partli': 3042, 'evok': 3043, 'wipe': 3044, 'eighti': 3045, 'clown': 3046, 'gray': 3047, 'importantli': 3048, 'australia': 3049, 'stoog': 3050, 'chew': 3051, 'destin': 3052, 'cheek': 3053, 'ho': 3054, 'piano': 3055, 'aforement': 3056, 'superbl': 3057, 'enthusiast': 3058, 'websit': 3059, 'farc': 3060, 'psychiatrist': 3061, 'lifestyl': 3062, 'debat': 3063, '45': 3064, 'flawless': 3065, 'seed': 3066, 'directori': 3067, 'dash': 3068, 'pressur': 3069, 'wick': 3070, 'abc': 3071, 'kitchen': 3072, 'dire': 3073, 'splatter': 3074, 'regardless': 3075, 'incompet': 3076, 'anni': 3077, 'bash': 3078, 'soviet': 3079, 'emili': 3080, 'akshay': 3081, 'slice': 3082, 'wrestl': 3083, 'drivel': 3084, 'judi': 3085, 'increas': 3086, 'mann': 3087, 'curios': 3088, 'helicopt': 3089, 'doo': 3090, 'recov': 3091, 'ken': 3092, 'duo': 3093, 'prize': 3094, 'dave': 3095, 'flower': 3096, 'cia': 3097, 'jar': 3098, 'suppli': 3099, 'cameron': 3100, 'boil': 3101, 'distant': 3102, 'glow': 3103, 'artifici': 3104, 'blob': 3105, 'pleasantli': 3106, 'beaten': 3107, 'seduc': 3108, 'chapter': 3109, 'lou': 3110, 'cagney': 3111, 'goldberg': 3112, 'eleg': 3113, 'web': 3114, 'turner': 3115, 'psychot': 3116, 'favour': 3117, 'laurel': 3118, 'panic': 3119, 'perri': 3120, 'drunken': 3121, 'combat': 3122, 'craig': 3123, 'glenn': 3124, 'splendid': 3125, 'craven': 3126, 'ranger': 3127, 'hop': 3128, 'ellen': 3129, 'francisco': 3130, 'min': 3131, 'hatr': 3132, 'plausibl': 3133, 'flip': 3134, 'greek': 3135, 'fx': 3136, 'rid': 3137, 'philosophi': 3138, 'alexand': 3139, 'shortli': 3140, 'ruth': 3141, 'gentl': 3142, 'slightest': 3143, '20th': 3144, 'falk': 3145, 'modesti': 3146, 'wizard': 3147, 'graduat': 3148, 'gandhi': 3149, 'legal': 3150, 'preciou': 3151, 'manhattan': 3152, 'lend': 3153, 'jealou': 3154, 'futurist': 3155, 'ocean': 3156, 'fund': 3157, 'holi': 3158, 'harm': 3159, 'we': 3160, 'unpleas': 3161, 'knight': 3162, 'felix': 3163, 'dracula': 3164, 'tall': 3165, 'thread': 3166, 'ami': 3167, 'overdon': 3168, 'reviv': 3169, 'childish': 3170, 'forbidden': 3171, 'mock': 3172, 'giallo': 3173, 'digniti': 3174, 'bless': 3175, 'explicit': 3176, 'tank': 3177, 'scientif': 3178, 'nod': 3179, 'awe': 3180, 'unwatch': 3181, 'elderli': 3182, 'broad': 3183, 'torment': 3184, 'fever': 3185, 'nerv': 3186, 'repeatedli': 3187, 'pirat': 3188, 'mel': 3189, 'eve': 3190, 'yesterday': 3191, '99': 3192, 'margaret': 3193, 'awaken': 3194, '2004': 3195, 'thick': 3196, 'verhoeven': 3197, 'acclaim': 3198, 'kay': 3199, 'ah': 3200, 'absenc': 3201, 'timothi': 3202, 'royal': 3203, 'custom': 3204, 'automat': 3205, 'launch': 3206, 'roman': 3207, 'eas': 3208, 'bin': 3209, 'publish': 3210, 'griffith': 3211, 'uniform': 3212, 'romero': 3213, 'politician': 3214, 'rivet': 3215, 'ambit': 3216, 'stiller': 3217, 'lean': 3218, 'bathroom': 3219, 'darker': 3220, 'transport': 3221, 'foul': 3222, 'termin': 3223, 'antic': 3224, 'stinker': 3225, 'phrase': 3226, 'homicid': 3227, 'pierc': 3228, 'pulp': 3229, 'crook': 3230, 'wallac': 3231, 'warren': 3232, 'purpl': 3233, 'gabriel': 3234, 'sunshin': 3235, 'tomato': 3236, 'pray': 3237, 'choreographi': 3238, 'brazil': 3239, 'ought': 3240, 'kenneth': 3241, 'evolv': 3242, 'viciou': 3243, 'sixti': 3244, 'awak': 3245, 'juvenil': 3246, '2003': 3247, 'li': 3248, 'eyr': 3249, 'saint': 3250, 'donna': 3251, 'marin': 3252, 'prom': 3253, 'q': 3254, 'karen': 3255, 'horrid': 3256, 'revolutionari': 3257, 'rambo': 3258, 'contrari': 3259, 'packag': 3260, 'coloni': 3261, 'album': 3262, 'hollow': 3263, 'candid': 3264, 'stole': 3265, 'option': 3266, 'boast': 3267, 'twelv': 3268, 'blade': 3269, 'conserv': 3270, 'ramon': 3271, 'nerd': 3272, 'defi': 3273, 'overr': 3274, 'dose': 3275, 'kapoor': 3276, 'ireland': 3277, 'mummi': 3278, 'beatti': 3279, 'mildr': 3280, 'flame': 3281, 'confirm': 3282, 'funer': 3283, 'trio': 3284, 'jazz': 3285, 'collabor': 3286, 'altman': 3287, 'natali': 3288, 'detract': 3289, 'protest': 3290, 'global': 3291, 'astonish': 3292, 'kirk': 3293, 'fulci': 3294, 'racial': 3295, 'whip': 3296, 'spit': 3297, 'enterpris': 3298, 'nicholson': 3299, 'blake': 3300, 'bottl': 3301, 'mystic': 3302, 'destini': 3303, 'leap': 3304, 'yellow': 3305, 'bull': 3306, 'delici': 3307, 'shade': 3308, 'audio': 3309, 'tommi': 3310, 'bedroom': 3311, 'todd': 3312, 'harder': 3313, 'visibl': 3314, 'enchant': 3315, 'merci': 3316, 'neo': 3317, 'inherit': 3318, 'vivid': 3319, 'threw': 3320, 'meaningless': 3321, 'reunit': 3322, 'adolesc': 3323, 'popcorn': 3324, 'pseudo': 3325, 'altogeth': 3326, 'fonda': 3327, 'staff': 3328, 'swedish': 3329, 'reserv': 3330, 'kennedi': 3331, 'fanat': 3332, 'decor': 3333, 'tip': 3334, 'voight': 3335, 'uneven': 3336, 'respond': 3337, 'exhibit': 3338, 'jew': 3339, 'moodi': 3340, 'lawrenc': 3341, 'bust': 3342, 'synopsi': 3343, 'wire': 3344, 'suspici': 3345, 'befriend': 3346, 'madonna': 3347, 'await': 3348, 'lemmon': 3349, 'atlanti': 3350, 'leonard': 3351, 'edi': 3352, 'crocodil': 3353, 'ruthless': 3354, 'roommat': 3355, 'audit': 3356, 'chao': 3357, 'garner': 3358, 'clumsi': 3359, 'bold': 3360, 'unsettl': 3361, 'clint': 3362, 'centr': 3363, 'bargain': 3364, 'ventur': 3365, 'abysm': 3366, 'holli': 3367, 'incident': 3368, 'carl': 3369, '2007': 3370, 'dimens': 3371, 'palma': 3372, 'voyag': 3373, 'rural': 3374, 'bradi': 3375, 'tiger': 3376, 'lit': 3377, 'elimin': 3378, 'characterist': 3379, 'echo': 3380, 'versu': 3381, 'ant': 3382, 'poetic': 3383, 'hart': 3384, 'troop': 3385, 'neglect': 3386, 'trail': 3387, 'wealth': 3388, 'daddi': 3389, 'nearbi': 3390, 'humili': 3391, 'cd': 3392, '2nd': 3393, 'acknowledg': 3394, 'imperson': 3395, 'cari': 3396, 'immigr': 3397, 'mall': 3398, 'cuba': 3399, 'timon': 3400, 'jeffrey': 3401, 'mickey': 3402, 'pun': 3403, 'marshal': 3404, 'repuls': 3405, 'infect': 3406, 'mistaken': 3407, 'collaps': 3408, 'solo': 3409, 'homer': 3410, 'saga': 3411, 'celluloid': 3412, 'domest': 3413, 'paus': 3414, 'prejudic': 3415, 'interrupt': 3416, 'chest': 3417, 'cake': 3418, 'leon': 3419, 'equip': 3420, 'coat': 3421, 'apolog': 3422, 'inappropri': 3423, 'milk': 3424, 'sore': 3425, '1996': 3426, 'coffe': 3427, 'harvey': 3428, 'assembl': 3429, 'olivi': 3430, 'promin': 3431, 'hbo': 3432, 'gear': 3433, 'undoubtedli': 3434, 'pant': 3435, 'tribe': 3436, 'ginger': 3437, 'inan': 3438, 'humbl': 3439, 'airplan': 3440, 'florida': 3441, 'furthermor': 3442, 'colleagu': 3443, 'colonel': 3444, 'aveng': 3445, 'consum': 3446, 'maggi': 3447, 'brooklyn': 3448, 'exot': 3449, 'primari': 3450, 'devast': 3451, 'institut': 3452, 'polanski': 3453, 'retain': 3454, 'highest': 3455, 'pot': 3456, 'jenni': 3457, 'trace': 3458, 'embrac': 3459, 'solut': 3460, 'pen': 3461, 'vulgar': 3462, 'instant': 3463, 'poke': 3464, 'bowl': 3465, 'dian': 3466, 'smaller': 3467, 'rick': 3468, 'gender': 3469, '3rd': 3470, 'disabl': 3471, 'principl': 3472, 'outer': 3473, 'illog': 3474, 'strain': 3475, 'descend': 3476, 'cope': 3477, 'ya': 3478, 'wive': 3479, 'godzilla': 3480, 'seduct': 3481, 'sale': 3482, '1999': 3483, 'dutch': 3484, 'linda': 3485, 'bubbl': 3486, 'cue': 3487, 'dive': 3488, 'inferior': 3489, 'gloriou': 3490, 'blatant': 3491, 'scope': 3492, 'primarili': 3493, 'lol': 3494, 'yard': 3495, 'predecessor': 3496, 'hal': 3497, 'beneath': 3498, 'secondli': 3499, 'devoid': 3500, 'glamor': 3501, 'gundam': 3502, 'vast': 3503, 'mixtur': 3504, 'dud': 3505, 'rabbit': 3506, 'streep': 3507, 'simplist': 3508, 'z': 3509, 'hideou': 3510, 'aggress': 3511, 'talki': 3512, 'shirley': 3513, 'casual': 3514, 'pearl': 3515, 'alert': 3516, 'alfr': 3517, 'trademark': 3518, 'myer': 3519, 'domino': 3520, 'invas': 3521, 'countrysid': 3522, 'museum': 3523, 'breed': 3524, 'et': 3525, 'april': 3526, 'grinch': 3527, 'shelf': 3528, 'disjoint': 3529, 'senseless': 3530, 'garbo': 3531, 'arab': 3532, 'stack': 3533, 'unhappi': 3534, 'maci': 3535, 'stellar': 3536, 'mail': 3537, 'hardcor': 3538, 'defens': 3539, 'rendit': 3540, 'illeg': 3541, 'boom': 3542, 'loyal': 3543, 'sh': 3544, 'slide': 3545, 'vanish': 3546, 'hopeless': 3547, 'disgrac': 3548, 'applaud': 3549, 'obtain': 3550, 'acid': 3551, 'stir': 3552, 'oz': 3553, 'robinson': 3554, 'khan': 3555, 'mayor': 3556, 'robberi': 3557, 'experiment': 3558, 'uwe': 3559, 'incomprehens': 3560, 'wont': 3561, 'craze': 3562, 'dismiss': 3563, 'grandfath': 3564, 'recruit': 3565, 'tenant': 3566, 'emphasi': 3567, 'tempt': 3568, 'declin': 3569, 'counter': 3570, 'hartley': 3571, 'soccer': 3572, 'spider': 3573, 'diana': 3574, 'fri': 3575, 'topless': 3576, 'psychic': 3577, 'span': 3578, 'amanda': 3579, 'dicken': 3580, 'rifl': 3581, 'blew': 3582, 'scroog': 3583, 'berlin': 3584, 'parad': 3585, 'resurrect': 3586, 'shaw': 3587, 'niro': 3588, 'bitch': 3589, 'ration': 3590, 'ethnic': 3591, 'goer': 3592, 'faster': 3593, 'sympath': 3594, 'intim': 3595, 'trashi': 3596, 'porno': 3597, 'sibl': 3598, 'lumet': 3599, 'shed': 3600, 'riot': 3601, 'justin': 3602, 'woo': 3603, 'revolt': 3604, 'wet': 3605, 'farmer': 3606, 'immort': 3607, 'dealer': 3608, 'nephew': 3609, 'ballet': 3610, 'choru': 3611, 'feminist': 3612, 'mario': 3613, 'wheel': 3614, '00': 3615, 'region': 3616, 'steam': 3617, 'eager': 3618, 'jonathan': 3619, 'hopper': 3620, 'hesit': 3621, 'gap': 3622, 'partial': 3623, 'weakest': 3624, 'enlighten': 3625, 'ensur': 3626, 'unreal': 3627, 'rider': 3628, 'biographi': 3629, 'patriot': 3630, 'honesti': 3631, 'commend': 3632, 'lena': 3633, 'andr': 3634, 'worm': 3635, 'wendi': 3636, 'slick': 3637, 'vice': 3638, 'mutant': 3639, 'prequel': 3640, 'wore': 3641, 'safeti': 3642, 'composit': 3643, 'snap': 3644, 'nostalg': 3645, 'properti': 3646, 'owen': 3647, 'victori': 3648, 'similarli': 3649, 'hung': 3650, 'psychopath': 3651, 'charlott': 3652, 'repress': 3653, 'confin': 3654, 'kingdom': 3655, 'skull': 3656, 'sappi': 3657, 'util': 3658, 'franco': 3659, 'sandra': 3660, 'morri': 3661, 'leo': 3662, 'blunt': 3663, 'macarthur': 3664, 'whoopi': 3665, 'exit': 3666, 'heartbreak': 3667, 'bonu': 3668, 'rope': 3669, 'speci': 3670, 'emperor': 3671, 'rambl': 3672, 'despair': 3673, 'drum': 3674, 'compass': 3675, 'drain': 3676, '1972': 3677, 'dalton': 3678, 'snl': 3679, 'bumbl': 3680, 'miseri': 3681, 'farrel': 3682, 'tail': 3683, 'del': 3684, 'recycl': 3685, 'cg': 3686, 'rocki': 3687, 'acquir': 3688, 'tad': 3689, 'latin': 3690, 'strand': 3691, 'pattern': 3692, 'nervou': 3693, 'bow': 3694, 'repli': 3695, 'montana': 3696, 'valuabl': 3697, 'bergman': 3698, 'campbel': 3699, 'compens': 3700, 'hyde': 3701, 'thru': 3702, 'dust': 3703, 'deed': 3704, 'kyle': 3705, 'rotten': 3706, 'da': 3707, 'percept': 3708, 'slug': 3709, 'contempl': 3710, 'romp': 3711, 'rapist': 3712, 'martian': 3713, 'carradin': 3714, 'bleed': 3715, 'airport': 3716, '35': 3717, 'gal': 3718, 'oppress': 3719, 'wacki': 3720, 'gimmick': 3721, 'pour': 3722, 'downhil': 3723, 'radic': 3724, 'orson': 3725, 'chess': 3726, 'mistress': 3727, 'tonight': 3728, 'olli': 3729, 'roth': 3730, '1983': 3731, 'arguabl': 3732, 'dazzl': 3733, 'mislead': 3734, 'edgar': 3735, 'paltrow': 3736, 'tackl': 3737, 'arc': 3738, 'pervers': 3739, 'tooth': 3740, 'heal': 3741, 'pervert': 3742, 'attorney': 3743, 'shelley': 3744, 'banal': 3745, 'stilt': 3746, 'unpredict': 3747, 'taught': 3748, 'champion': 3749, 'preach': 3750, 'pursuit': 3751, 'slash': 3752, 'programm': 3753, 'belt': 3754, 'melodi': 3755, 'vocal': 3756, 'uplift': 3757, 'tiresom': 3758, 'mesmer': 3759, 'sensat': 3760, 'dixon': 3761, 'marti': 3762, 'duval': 3763, 'closest': 3764, 'rubi': 3765, 'bela': 3766, 'raymond': 3767, 'cleverli': 3768, 'plight': 3769, 'vengeanc': 3770, 'graham': 3771, 'franki': 3772, 'gambl': 3773, 'poem': 3774, 'passeng': 3775, 'chicken': 3776, 'conneri': 3777, 'orang': 3778, 'maid': 3779, 'virginia': 3780, 'employe': 3781, 'sirk': 3782, 'outing': 3783, 'mute': 3784, 'suffic': 3785, 'whine': 3786, 'clone': 3787, 'bay': 3788, '1968': 3789, 'crystal': 3790, 'convincingli': 3791, 'numb': 3792, 'monologu': 3793, 'habit': 3794, 'yawn': 3795, 'giggl': 3796, 'lundgren': 3797, 'engross': 3798, 'gerard': 3799, 'inject': 3800, 'swallow': 3801, 'tube': 3802, 'secretli': 3803, 'quarter': 3804, 'abraham': 3805, 'amitabh': 3806, 'extens': 3807, 'profan': 3808, 'paranoia': 3809, 'climact': 3810, 'volum': 3811, 'calm': 3812, 'iran': 3813, 'scottish': 3814, 'pokemon': 3815, 'septemb': 3816, 'fed': 3817, 'expand': 3818, 'grotesqu': 3819, 'chicago': 3820, 'frankenstein': 3821, 'underst': 3822, 'taxi': 3823, 'austen': 3824, 'trend': 3825, 'surpass': 3826, 'abort': 3827, 'poetri': 3828, 'franci': 3829, 'plod': 3830, 'richardson': 3831, 'dispos': 3832, 'profess': 3833, 'backward': 3834, 'junior': 3835, 'ethan': 3836, 'im': 3837, 'nichola': 3838, 'lowest': 3839, 'bend': 3840, 'earl': 3841, 'spock': 3842, 'linger': 3843, 'meander': 3844, 'sue': 3845, 'rant': 3846, 'greedi': 3847, 'literatur': 3848, 'household': 3849, 'tourist': 3850, 'instrument': 3851, 'waitress': 3852, 'stallon': 3853, 'compliment': 3854, 'mundan': 3855, 'catchi': 3856, 'lure': 3857, 'econom': 3858, 'spoke': 3859, 'myth': 3860, 'simplic': 3861, 'muddl': 3862, 'der': 3863, 'cannon': 3864, 'rubber': 3865, 'hum': 3866, 'eugen': 3867, 'descent': 3868, 'nostalgia': 3869, 'dysfunct': 3870, 'bacal': 3871, 'map': 3872, 'occupi': 3873, 'phoni': 3874, 'hello': 3875, 'coast': 3876, 'carel': 3877, 'eaten': 3878, 'phantom': 3879, 'firstli': 3880, 'flee': 3881, 'cent': 3882, 'equival': 3883, 'randi': 3884, 'deaf': 3885, 'dement': 3886, 'insur': 3887, 'alongsid': 3888, 'damon': 3889, 'lang': 3890, 'mankind': 3891, 'molli': 3892, 'mortal': 3893, 'furi': 3894, 'crucial': 3895, 'omen': 3896, 'irrelev': 3897, 'sissi': 3898, 'duck': 3899, 'recognis': 3900, 'recognit': 3901, 'stale': 3902, 'louis': 3903, 'dictat': 3904, 'june': 3905, 'drake': 3906, 'likewis': 3907, 'wisdom': 3908, 'loyalti': 3909, 'heel': 3910, 'lengthi': 3911, 'bump': 3912, 'twilight': 3913, 'dreari': 3914, 'daisi': 3915, 'freez': 3916, 'cyborg': 3917, 'grayson': 3918, 'distinguish': 3919, 'blackmail': 3920, 'newli': 3921, 'rude': 3922, 'damm': 3923, '1973': 3924, 'bike': 3925, 'rooney': 3926, 'reign': 3927, 'labor': 3928, 'ashley': 3929, 'buffalo': 3930, 'onlin': 3931, 'biko': 3932, 'antwon': 3933, 'attribut': 3934, 'provoc': 3935, 'exposur': 3936, 'proce': 3937, 'worn': 3938, 'tunnel': 3939, 'vein': 3940, 'boxer': 3941, 'chronicl': 3942, 'nineti': 3943, 'inher': 3944, 'startl': 3945, 'keith': 3946, 'ridden': 3947, 'baddi': 3948, 'prey': 3949, 'butler': 3950, 'pink': 3951, 'emphas': 3952, 'incorpor': 3953, 'analysi': 3954, 'interior': 3955, 'unorigin': 3956, 'approv': 3957, 'barrymor': 3958, 'basketbal': 3959, 'sailor': 3960, 'nicol': 3961, 'stalker': 3962, 'walsh': 3963, 'othello': 3964, 'predat': 3965, 'substitut': 3966, 'millionair': 3967, 'er': 3968, 'mormon': 3969, 'degrad': 3970, 'underli': 3971, 'improvis': 3972, 'robbin': 3973, 'bunni': 3974, 'undeni': 3975, 'drift': 3976, 'hypnot': 3977, 'condemn': 3978, 'meg': 3979, 'elm': 3980, 'mighti': 3981, 'julian': 3982, 'simmon': 3983, 'barrel': 3984, 'indiffer': 3985, 'fleet': 3986, 'unrel': 3987, 'belushi': 3988, 'meyer': 3989, 'carla': 3990, 'unawar': 3991, 'warmth': 3992, 'errol': 3993, 'disord': 3994, 'shove': 3995, 'exquisit': 3996, 'roof': 3997, 'agenda': 3998, 'enthusiasm': 3999, 'rukh': 4000, 'firm': 4001, 'alarm': 4002, 'dolph': 4003, 'hay': 4004, 'mtv': 4005, 'edgi': 4006, 'watson': 4007, 'lampoon': 4008, 'nyc': 4009, 'priceless': 4010, 'palac': 4011, 'greed': 4012, 'alison': 4013, 'marion': 4014, 'reid': 4015, '3d': 4016, 'vital': 4017, 'novak': 4018, 'pamela': 4019, 'minimum': 4020, 'thompson': 4021, 'gestur': 4022, 'eastern': 4023, 'preserv': 4024, 'glanc': 4025, 'peril': 4026, 'petti': 4027, 'simultan': 4028, 'unleash': 4029, 'championship': 4030, 'session': 4031, 'distort': 4032, 'crown': 4033, 'profit': 4034, '13th': 4035, 'nun': 4036, 'ponder': 4037, 'testament': 4038, '1933': 4039, 'spain': 4040, 'showdown': 4041, 'campaign': 4042, 'peck': 4043, 'sergeant': 4044, 'cassidi': 4045, 'what': 4046, 'drip': 4047, 'zizek': 4048, 'angela': 4049, 'randomli': 4050, 'israel': 4051, 'beatl': 4052, 'iraq': 4053, 'coup': 4054, 'orlean': 4055, 'valentin': 4056, 'han': 4057, 'bro': 4058, 'unimagin': 4059, 'stake': 4060, 'gentleman': 4061, 'contradict': 4062, 'restrain': 4063, 'rout': 4064, 'crawl': 4065, 'stroke': 4066, 'crow': 4067, 'miyazaki': 4068, 'brenda': 4069, 'exposit': 4070, 'wig': 4071, 'travesti': 4072, 'cooki': 4073, 'climat': 4074, 'mon': 4075, 'sabrina': 4076, 'scotland': 4077, 'empathi': 4078, 'quinn': 4079, 'din': 4080, 'represent': 4081, 'reson': 4082, 'realm': 4083, 'regist': 4084, 'valley': 4085, 'fido': 4086, 'cream': 4087, 'buster': 4088, 'jan': 4089, 'calib': 4090, 'shootout': 4091, 'kurosawa': 4092, 'perpetu': 4093, '1984': 4094, 'meryl': 4095, 'traumat': 4096, 'greg': 4097, 'shaki': 4098, 'cloud': 4099, 'passabl': 4100, 'painter': 4101, 'pole': 4102, 'perceiv': 4103, 'pretens': 4104, 'businessman': 4105, 'dana': 4106, 'femm': 4107, 'warrant': 4108, 'delic': 4109, 'ross': 4110, 'stargat': 4111, 'shoddi': 4112, 'compromis': 4113, 'geek': 4114, '1987': 4115, 'unsatisfi': 4116, 'distress': 4117, 'mclaglen': 4118, 'crawford': 4119, 'tacki': 4120, 'derang': 4121, 'sammi': 4122, 'baldwin': 4123, 'absent': 4124, 'soderbergh': 4125, 'sucker': 4126, 'monoton': 4127, 'fuller': 4128, '1997': 4129, 'wax': 4130, 'abomin': 4131, 'ustinov': 4132, 'unseen': 4133, 'spacey': 4134, 'censor': 4135, 'darren': 4136, 'josh': 4137, 'demis': 4138, 'furiou': 4139, 'sid': 4140, 'wholli': 4141, 'tarantino': 4142, 'judgment': 4143, 'anchor': 4144, 'polici': 4145, 'uncov': 4146, 'primit': 4147, 'norm': 4148, 'austin': 4149, 'expedit': 4150, 'accuraci': 4151, 'fenc': 4152, '1993': 4153, 'tech': 4154, 'unravel': 4155, 'jewel': 4156, 'valid': 4157, 'clash': 4158, 'seal': 4159, 'verbal': 4160, 'deceas': 4161, 'exclus': 4162, 'fog': 4163, 'reluct': 4164, 'dee': 4165, 'nathan': 4166, 'antonioni': 4167, 'kumar': 4168, 'deniro': 4169, 'correctli': 4170, 'click': 4171, 'malon': 4172, 'pocket': 4173, 'sheet': 4174, '3000': 4175, '2008': 4176, 'debt': 4177, 'mode': 4178, 'wang': 4179, 'sustain': 4180, 'temper': 4181, 'sunni': 4182, 'enforc': 4183, 'behold': 4184, 'conduct': 4185, 'logan': 4186, 'hallucin': 4187, 'alec': 4188, 'joel': 4189, 'slam': 4190, 'wretch': 4191, 'seldom': 4192, 'roller': 4193, 'murray': 4194, 'sand': 4195, 'dreck': 4196, 'trait': 4197, '1971': 4198, 'fart': 4199, 'clerk': 4200, 'darn': 4201, 'shanghai': 4202, 'crippl': 4203, 'patienc': 4204, 'tax': 4205, 'fought': 4206, '1995': 4207, 'nicola': 4208, 'unfair': 4209, 'bake': 4210, 'ritual': 4211, 'fabric': 4212, 'vanc': 4213, 'technicolor': 4214, 'shell': 4215, 'schedul': 4216, 'divid': 4217, 'critiqu': 4218, 'conscious': 4219, 'stark': 4220, 'tactic': 4221, 'exhaust': 4222, 'legaci': 4223, 'runner': 4224, 'helpless': 4225, 'scriptwrit': 4226, 'sweep': 4227, 'fundament': 4228, 'phil': 4229, 'pete': 4230, 'bridget': 4231, 'bias': 4232, 'rita': 4233, 'isabel': 4234, 'guitar': 4235, 'squad': 4236, 'soup': 4237, 'outlin': 4238, 'despis': 4239, 'canyon': 4240, 'preposter': 4241, 'stuart': 4242, 'penni': 4243, 'preston': 4244, 'robber': 4245, 'grief': 4246, 'clau': 4247, 'lacklust': 4248, 'inabl': 4249, 'passag': 4250, 'bloom': 4251, 'cigarett': 4252, 'alley': 4253, 'marc': 4254, 'kansa': 4255, 'downey': 4256, 'russia': 4257, 'culmin': 4258, 'sniper': 4259, 'jacket': 4260, 'flair': 4261, 'alicia': 4262, 'sentinel': 4263, 'palanc': 4264, 'boyl': 4265, 'consciou': 4266, 'invad': 4267, 'gregori': 4268, 'jodi': 4269, 'sugar': 4270, 'vomit': 4271, 'connor': 4272, 'unexpectedli': 4273, 'drove': 4274, 'rehash': 4275, 'implic': 4276, 'liberti': 4277, 'restrict': 4278, 'newman': 4279, 'agenc': 4280, 'rear': 4281, 'propos': 4282, 'delv': 4283, 'awhil': 4284, 'asylum': 4285, 'vet': 4286, 'pale': 4287, 'wrench': 4288, 'behaviour': 4289, 'mccoy': 4290, 'aesthet': 4291, 'rod': 4292, 'bacon': 4293, 'ladder': 4294, '1936': 4295, '22': 4296, 'sharon': 4297, 'foxx': 4298, 'delet': 4299, 'lush': 4300, 'tendenc': 4301, 'chainsaw': 4302, 'feat': 4303, 'arrow': 4304, 'improb': 4305, 'horn': 4306, 'karl': 4307, 'cap': 4308, 'yeti': 4309, 'rehears': 4310, 'tripe': 4311, 'rampag': 4312, 'kolchak': 4313, 'spice': 4314, 'sung': 4315, 'stream': 4316, '1920': 4317, 'paramount': 4318, '1988': 4319, 'shortcom': 4320, 'amazon': 4321, 'weav': 4322, 'newcom': 4323, 'thunderbird': 4324, 'tomorrow': 4325, 'rhythm': 4326, 'wildli': 4327, 'prank': 4328, 'globe': 4329, 'hungri': 4330, 'hulk': 4331, 'tasteless': 4332, 'underneath': 4333, 'coaster': 4334, 'filler': 4335, 'visitor': 4336, 'paradis': 4337, 'financ': 4338, 'lurk': 4339, '1978': 4340, 'loneli': 4341, 'suspicion': 4342, 'conscienc': 4343, 'hackney': 4344, 'minu': 4345, 'suffici': 4346, 'elit': 4347, 'rumor': 4348, '19th': 4349, 'el': 4350, 'aristocrat': 4351, 'basing': 4352, 'fright': 4353, 'scoop': 4354, 'wagner': 4355, 'bread': 4356, 'teas': 4357, 'beverli': 4358, 'naughti': 4359, 'curli': 4360, 'quietli': 4361, 'springer': 4362, 'worship': 4363, 'dirt': 4364, 'iv': 4365, 'paxton': 4366, 'impos': 4367, 'straightforward': 4368, 'heist': 4369, 'choppi': 4370, 'literari': 4371, 'leigh': 4372, 'minist': 4373, 'wwe': 4374, 'entranc': 4375, 'couch': 4376, 'hopkin': 4377, 'immers': 4378, 'ingeni': 4379, 'recogniz': 4380, 'penn': 4381, 'posey': 4382, 'secondari': 4383, 'ram': 4384, 'rub': 4385, 'inmat': 4386, 'standout': 4387, 'chavez': 4388, 'counterpart': 4389, 'lectur': 4390, 'abrupt': 4391, 'tierney': 4392, '1939': 4393, 'smell': 4394, 'chamberlain': 4395, '75': 4396, '1989': 4397, 'en': 4398, 'cancer': 4399, 'atroc': 4400, 'grudg': 4401, 'brit': 4402, 'attenborough': 4403, 'laurenc': 4404, 'clan': 4405, 'transcend': 4406, '1986': 4407, 'variat': 4408, 'policeman': 4409, 'skeptic': 4410, 'heartfelt': 4411, 'esther': 4412, 'sublim': 4413, 'enthral': 4414, 'bernard': 4415, 'ace': 4416, 'moreov': 4417, 'watcher': 4418, 'sassi': 4419, 'entitl': 4420, 'yearn': 4421, 'lindsay': 4422, 'net': 4423, 'duel': 4424, 'misguid': 4425, 'nemesi': 4426, 'morbid': 4427, 'convert': 4428, 'injuri': 4429, 'nolan': 4430, 'cattl': 4431, 'ratso': 4432, 'geni': 4433, 'missil': 4434, 'quaid': 4435, 'artsi': 4436, 'steadi': 4437, 'spiral': 4438, 'enabl': 4439, 'diari': 4440, 'puppi': 4441, 'brood': 4442, 'bye': 4443, 'out': 4444, 'graini': 4445, 'moder': 4446, 'mytholog': 4447, 'hopelessli': 4448, 'obstacl': 4449, 'unexplain': 4450, 'setup': 4451, 'youngest': 4452, 'grin': 4453, 'bean': 4454, 'kidman': 4455, 'facil': 4456, 'poe': 4457, 'dont': 4458, 'rosemari': 4459, 'uncut': 4460, '1979': 4461, 'carlito': 4462, 'tyler': 4463, 'reliabl': 4464, 'cruelti': 4465, 'vader': 4466, 'buzz': 4467, 'characteris': 4468, 'hk': 4469, 'kitti': 4470, 'egg': 4471, 'fuel': 4472, 'disastr': 4473, 'exterior': 4474, 'christin': 4475, 'athlet': 4476, 'underworld': 4477, 'martha': 4478, 'spontan': 4479, 'hammi': 4480, 'heap': 4481, 'bewar': 4482, 'gillian': 4483, 'clueless': 4484, 'hain': 4485, '1969': 4486, 'preming': 4487, 'kline': 4488, 'bounc': 4489, 'baffl': 4490, 'niec': 4491, 'despic': 4492, 'weather': 4493, 'narrow': 4494, 'acquaint': 4495, 'decept': 4496, 'brendan': 4497, 'sweat': 4498, 'effici': 4499, 'oblig': 4500, 'bronson': 4501, 'gina': 4502, 'patricia': 4503, 'angst': 4504, 'dilemma': 4505, 'harmless': 4506, 'candl': 4507, '19': 4508, 'analyz': 4509, 'loi': 4510, 'fontain': 4511, 'outlaw': 4512, 'trigger': 4513, 'scar': 4514, 'virtu': 4515, 'mermaid': 4516, 'tick': 4517, 'renaiss': 4518, 'loath': 4519, 'suprem': 4520, 'preachi': 4521, 'taboo': 4522, 'injur': 4523, 'sooner': 4524, 'biker': 4525, 'sleepwalk': 4526, 'viewpoint': 4527, 'insipid': 4528, 'uh': 4529, 'astound': 4530, 'enlist': 4531, 'mayhem': 4532, 'goof': 4533, 'rome': 4534, 'headach': 4535, 'housewif': 4536, 'circu': 4537, 'shatter': 4538, 'lester': 4539, 'dandi': 4540, '73': 4541, 'hepburn': 4542, 'ebert': 4543, 'corbett': 4544, 'glorifi': 4545, 'spade': 4546, 'filth': 4547, 'hooker': 4548, 'whore': 4549, 'claustrophob': 4550, 'dish': 4551, 'fluff': 4552, 'ariel': 4553, 'tripl': 4554, 'salt': 4555, 'immatur': 4556, 'oldest': 4557, 'camcord': 4558, 'slimi': 4559, 'scorses': 4560, 'overlong': 4561, 'boston': 4562, 'amor': 4563, 'bent': 4564, 'macho': 4565, 'surgeri': 4566, 'phenomenon': 4567, 'intric': 4568, 'cassavet': 4569, 'hostag': 4570, 'dismal': 4571, 'zoom': 4572, 'foolish': 4573, 'guin': 4574, 'idol': 4575, 'gere': 4576, 'stair': 4577, 'hokey': 4578, 'stimul': 4579, 'contempt': 4580, 'steer': 4581, 'redund': 4582, 'sox': 4583, 'frantic': 4584, 'trivia': 4585, 'remad': 4586, 'conquer': 4587, 'beard': 4588, 'proport': 4589, 'dwarf': 4590, 'spinal': 4591, 'cow': 4592, '1976': 4593, 'zane': 4594, 'mutual': 4595, 'widescreen': 4596, 'alvin': 4597, 'messi': 4598, 'mount': 4599, 'shield': 4600, 'flag': 4601, '1981': 4602, 'muscl': 4603, 'fascist': 4604, 'margin': 4605, 'cush': 4606, 'preced': 4607, 'antagonist': 4608, 'flashi': 4609, 'flirt': 4610, 'schlock': 4611, 'down': 4612, 'spree': 4613, 'assert': 4614, 'obligatori': 4615, 'perman': 4616, 'rhyme': 4617, 'transplant': 4618, 'harold': 4619, 'faint': 4620, 'joker': 4621, 'gasp': 4622, 'cohen': 4623, 'shred': 4624, 'nolt': 4625, 'strongest': 4626, 'naschi': 4627, 'gabl': 4628, 'corman': 4629, 'keen': 4630, 'radiat': 4631, 'astronaut': 4632, 'www': 4633, 'danish': 4634, 'mol': 4635, 'brush': 4636, 'discern': 4637, '95': 4638, 'departur': 4639, 'resum': 4640, 'barn': 4641, 'archiv': 4642, 'someday': 4643, 'interestingli': 4644, 'fishburn': 4645, 'info': 4646, 'bitten': 4647, 'carey': 4648, 'neurot': 4649, 'wield': 4650, 'bachelor': 4651, 'deer': 4652, 'strive': 4653, 'instruct': 4654, 'triangl': 4655, 'raj': 4656, '1945': 4657, 'sensual': 4658, 'ritchi': 4659, 'off': 4660, 'scandal': 4661, 'hara': 4662, 'persuad': 4663, 'aborigin': 4664, 'inflict': 4665, 'claud': 4666, 'flock': 4667, 'repris': 4668, 'boob': 4669, 'divin': 4670, 'mobil': 4671, '28': 4672, 'vaniti': 4673, 'vibrant': 4674, 'recit': 4675, 'senior': 4676, 'traffic': 4677, 'proclaim': 4678, 'mobster': 4679, 'casino': 4680, 'europa': 4681, 'dim': 4682, 'rot': 4683, 'cycl': 4684, 'biblic': 4685, 'harrison': 4686, 'anton': 4687, 'axe': 4688, 'undermin': 4689, 'colin': 4690, 'artwork': 4691, 'melissa': 4692, 'cliffhang': 4693, 'miracul': 4694, 'hug': 4695, 'heartwarm': 4696, 'luka': 4697, 'pacif': 4698, 'jade': 4699, 'submit': 4700, 'fragil': 4701, 'cb': 4702, 'hapless': 4703, 'earnest': 4704, 'ish': 4705, 'dylan': 4706, 'carlo': 4707, 'cher': 4708, 'frontier': 4709, 'kathryn': 4710, 'prophet': 4711, 'pixar': 4712, 'timberlak': 4713, 'dame': 4714, 'neill': 4715, 'parson': 4716, 'hilar': 4717, 'banter': 4718, 'bate': 4719, 'clad': 4720, 'helm': 4721, 'pickford': 4722, 'loretta': 4723, 'wendigo': 4724, 'bondag': 4725, 'northern': 4726, 'lui': 4727, 'seedi': 4728, 'feast': 4729, 'illus': 4730, 'mason': 4731, 'milo': 4732, 'cerebr': 4733, 'legitim': 4734, 'toronto': 4735, 'aris': 4736, 'sicken': 4737, 'venom': 4738, 'token': 4739, 'choke': 4740, 'jo': 4741, 'rooki': 4742, 'static': 4743, 'trier': 4744, 'http': 4745, 'lucil': 4746, 'misfortun': 4747, 'jordan': 4748, 'blatantli': 4749, 'pc': 4750, 'vile': 4751, 'razor': 4752, 'estrang': 4753, 'electron': 4754, 'holocaust': 4755, 'akin': 4756, 'uma': 4757, 'marlon': 4758, 'redneck': 4759, 'vanessa': 4760, 'flavor': 4761, 'isra': 4762, 'breakfast': 4763, 'orphan': 4764, 'bikini': 4765, 'articl': 4766, 'nope': 4767, 'wardrob': 4768, 'foil': 4769, 'antholog': 4770, 'alexandr': 4771, 'shepherd': 4772, 'eli': 4773, 'winchest': 4774, 'mathieu': 4775, 'glare': 4776, 'audrey': 4777, 'linear': 4778, 'smack': 4779, 'tack': 4780, 'swept': 4781, 'boyer': 4782, 'fifth': 4783, 'styliz': 4784, 'howl': 4785, 'psych': 4786, 'ceremoni': 4787, 'comprehend': 4788, 'dudley': 4789, 'oppon': 4790, 'frog': 4791, 'cartoonish': 4792, 'abund': 4793, 'highway': 4794, 'deem': 4795, 'affleck': 4796, 'gunga': 4797, 'shorter': 4798, 'retriev': 4799, 'outdat': 4800, 'feminin': 4801, 'charlton': 4802, 'disregard': 4803, 'ideolog': 4804, 'knightley': 4805, 'wrestler': 4806, 'clinic': 4807, 'huston': 4808, 'peer': 4809, 'turd': 4810, 'nightclub': 4811, 'magician': 4812, 'leather': 4813, 'gilbert': 4814, 'moe': 4815, 'spawn': 4816, 'spine': 4817, 'cuban': 4818, 'bogu': 4819, 'whack': 4820, '1994': 4821, '1991': 4822, 'potter': 4823, 'evolut': 4824, 'sleaz': 4825, 'uniformli': 4826, 'snatch': 4827, 'summar': 4828, 'lifeless': 4829, 'einstein': 4830, 'newer': 4831, 'energet': 4832, 'monument': 4833, 'toe': 4834, 'chip': 4835, 'lighter': 4836, 'salman': 4837, 'plate': 4838, 'boo': 4839, 'goldsworthi': 4840, 'conrad': 4841, 'tara': 4842, 'bastard': 4843, 'phenomen': 4844, 'senat': 4845, 'breakdown': 4846, 'greet': 4847, 'cemeteri': 4848, 'btw': 4849, 'corn': 4850, 'collector': 4851, '4th': 4852, 'braveheart': 4853, 'client': 4854, 'lavish': 4855, 'durat': 4856, 'mitch': 4857, 'deliver': 4858, 'compris': 4859, 'replay': 4860, 'pronounc': 4861, 'trauma': 4862, 'jam': 4863, 'jule': 4864, 'outright': 4865, 'randolph': 4866, 'occup': 4867, 'wtf': 4868, 'appl': 4869, 'nina': 4870, 'capot': 4871, '1977': 4872, 'undertak': 4873, 'evelyn': 4874, 'ol': 4875, 'kazan': 4876, 'bori': 4877, 'mcqueen': 4878, 'ie': 4879, 'firmli': 4880, 'spectacl': 4881, 'gilliam': 4882, 'fluid': 4883, 'belli': 4884, 'undead': 4885, 'clara': 4886, 'constitut': 4887, 'alleg': 4888, 'liu': 4889, 'healthi': 4890, 'bulk': 4891, 'luxuri': 4892, 'kent': 4893, 'lex': 4894, 'sorrow': 4895, '1974': 4896, 'signal': 4897, 'embark': 4898, 'cecil': 4899, 'jedi': 4900, 'creek': 4901, 'inaccuraci': 4902, 'judd': 4903, 'armstrong': 4904, 'jare': 4905, 'eleven': 4906, 'historian': 4907, 'neatli': 4908, 'lauren': 4909, 'relentless': 4910, 'curtain': 4911, '1985': 4912, 'pioneer': 4913, 'forgiven': 4914, 'unsuspect': 4915, 'poker': 4916, 'palm': 4917, 'comb': 4918, 'kiddi': 4919, 'sidewalk': 4920, 'propheci': 4921, 'truman': 4922, 'id': 4923, 'mum': 4924, 'decapit': 4925, 'miniseri': 4926, 'sacrif': 4927, 'roar': 4928, 'galaxi': 4929, 'vignett': 4930, 'meal': 4931, 'knee': 4932, 'lanc': 4933, 'antonio': 4934, 'walt': 4935, 'spray': 4936, 'subtli': 4937, 'inclus': 4938, 'rosario': 4939, 'vain': 4940, 'blur': 4941, 'tokyo': 4942, 'pepper': 4943, 'conan': 4944, 'fruit': 4945, 'bait': 4946, 'goldblum': 4947, 'genet': 4948, 'paula': 4949, 'unattract': 4950, 'miami': 4951, 'inaccur': 4952, 'cape': 4953, 'abound': 4954, 'bsg': 4955, 'ash': 4956, 'congratul': 4957, 'porter': 4958, 'aussi': 4959, 'carmen': 4960, 'groan': 4961, 'basket': 4962, 'comprehens': 4963, 'profil': 4964, 'playboy': 4965, 'weari': 4966, 'vastli': 4967, 'scarfac': 4968, 'motorcycl': 4969, 'verg': 4970, 'detach': 4971, 'monti': 4972, 'hostil': 4973, 'bravo': 4974, 'casper': 4975, 'jill': 4976, 'sparkl': 4977, 'incorrect': 4978, 'sophi': 4979, 'handicap': 4980, 'omin': 4981, 'reincarn': 4982, 'macabr': 4983, 'bach': 4984, 'growth': 4985, 'spill': 4986, 'optimist': 4987, 'modest': 4988, 'cypher': 4989, 'hackman': 4990, 'scariest': 4991, 'orchestr': 4992, '21st': 4993, 'masterson': 4994, 'weaker': 4995, 'rapidli': 4996, 'drone': 4997, 'turtl': 4998, 'frontal': 4999}


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


```python
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
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())


```python
!pygmentize train/model.py
```

    [34mimport[39;49;00m [04m[36mtorch.nn[39;49;00m [34mas[39;49;00m [04m[36mnn[39;49;00m
    
    [34mclass[39;49;00m [04m[32mLSTMClassifier[39;49;00m(nn.Module):
        [33m"""[39;49;00m
    [33m    This is the simple RNN model we will be using to perform Sentiment Analysis.[39;49;00m
    [33m    """[39;49;00m
    
        [34mdef[39;49;00m [32m__init__[39;49;00m([36mself[39;49;00m, embedding_dim, hidden_dim, vocab_size):
            [33m"""[39;49;00m
    [33m        Initialize the model by settingg up the various layers.[39;49;00m
    [33m        """[39;49;00m
            [36msuper[39;49;00m(LSTMClassifier, [36mself[39;49;00m).[32m__init__[39;49;00m()
    
            [36mself[39;49;00m.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=[34m0[39;49;00m)
            [36mself[39;49;00m.lstm = nn.LSTM(embedding_dim, hidden_dim)
            [36mself[39;49;00m.dense = nn.Linear(in_features=hidden_dim, out_features=[34m1[39;49;00m)
            [36mself[39;49;00m.sig = nn.Sigmoid()
            
            [36mself[39;49;00m.word_dict = [36mNone[39;49;00m
    
        [34mdef[39;49;00m [32mforward[39;49;00m([36mself[39;49;00m, x):
            [33m"""[39;49;00m
    [33m        Perform a forward pass of our model on some input.[39;49;00m
    [33m        """[39;49;00m
            x = x.t()
            lengths = x[[34m0[39;49;00m,:]
            reviews = x[[34m1[39;49;00m:,:]
            embeds = [36mself[39;49;00m.embedding(reviews)
            lstm_out, _ = [36mself[39;49;00m.lstm(embeds)
            out = [36mself[39;49;00m.dense(lstm_out)
            out = out[lengths - [34m1[39;49;00m, [36mrange[39;49;00m([36mlen[39;49;00m(lengths))]
            [34mreturn[39;49;00m [36mself[39;49;00m.sig(out.squeeze())


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

## Web App For My Little Model

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
