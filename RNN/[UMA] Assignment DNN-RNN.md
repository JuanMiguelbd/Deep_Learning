
# Lab assignment: analyzing movie reviews with Recurrent Neural Networks

<img src="img/cinemaReviews.png" style="width:600px;">

In this assignment we will analyze the sentiment, positive or negative, expressed in a set of movie reviews IMDB. To do so we will make use of word embeddings and recurrent neural networks.

## Guidelines

Throughout this notebook you will find empty cells that you will need to fill with your own code. Follow the instructions in the notebook and pay special attention to the following symbols.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>You will need to solve a question by writing your own code or answer in the cell immediately below, or in a different file as instructed.</td></tr>
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>This is a hint or useful observation that can help you solve this assignment. You are not expected to write any solution, but you should pay attention to them to understand the assignment.</td></tr>
 <tr><td width="80"><img src="img/pro.png" style="width:auto;height:auto"></td><td>This is an advanced and voluntary exercise that can help you gain a deeper knowledge into the topic. Good luck!</td></tr>
</table>

During the assigment you will make use of several Python packages that might not be installed in your machine. If that is the case, you can install new Python packages with

    conda install PACKAGENAME
    
if you are using Python Anaconda. Else you should use

    pip install PACKAGENAME

You will need the following packages for this particular assignment. Make sure they are available before proceeding:

* **numpy**
* **keras**
* **matplotlib**

The following code will embed any plots into the notebook instead of generating a new window:


```python
#Importing required packages
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from nltk.corpus import stopwords
from keras.preprocessing import sequence
```

    C:\Users\raul_\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

Lastly, if you need any help on the usage of a Python function you can place the writing cursor over its name and press Caps+Shift to produce a pop-out with related documentation. This will only work inside code cells. 

Let's go!

## The Keras library

In this lab we will make use of the <a href=http://keras.io/>keras</a> Deep Learning library for Python. This library allows building several kinds of shallow and deep networks, following either a sequential or a graph architecture.

## Data loading

We will make use of a part of the IMDB database on movie reviews. IMDB rates movies with a score ranging 0-10, but for simplicity we will consider a dataset of good and bad reviews, where a review has been considered bad with a score smaller than 4, and good if it features a score larger than 7. The data is available under the *data* folder.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Load the data into two variables, a list **text** with each of the movie reviews and a list **y** of the class labels.
 </td></tr>
</table>


```python
IMDB=pd.read_csv("C:\\Users\\raul_\\JupiterNotebooks\\MasterBigData\\data\\datafull.csv",sep='\t',header=0)

```


```python
# Stopwords eliminated using NLTK
stop = stopwords.words('english')
IMDB['text']=IMDB['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
```


```python
print(IMDB.shape)
IMDB.head()
```

    (25000, 2)
    




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
      <th>sentiment</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>I simply cant understand relics Ceausescu era ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Director Raoul Walsh like Michael Bay '40's ye...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>It could better film. It drag points, central ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>It hard rate film. As entertainment value 21st...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>I've read terrible things film, I prepared wor...</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=IMDB["text"]
y=IMDB["sentiment"]
```


```python
# we see,it's balanced the dataset
y.value_counts()
```




    1    12500
    0    12500
    Name: sentiment, dtype: int64



For convenience in what follows we will also split the data into a training and test subsets.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Split the list of texts into **texts_train** and **texts_test** lists, keeping 25% of the texts for test. Split in the same way the labels, obtaining lists **y_train** and **y_test**.
 </td></tr>
</table>


```python
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```


```python
split_IMDB = StratifiedShuffleSplit(n_splits=3, test_size=0.25, random_state=124)

for train_index, test_index in split_IMDB.split(X, y):
     
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
```


```python
print(X_train.head())
print(y_train.head())
```

    12933    ...but working I surprised see many people con...
    8666     As rule, Full Moon production logo warning sig...
    21192    [***POSSIBLE SPOILERS***] This movie's reputat...
    20160    Even 1942 standards movie-making setup HER CAR...
    18376    I've never huge fan Mormon films. Being Mormon...
    Name: text, dtype: object
    12933    0
    8666     1
    21192    0
    20160    0
    18376    0
    Name: sentiment, dtype: int64
    

## Data processing

We can't introduce text directly into the network, so we will have to tranform it to a vector representation. To do so, we will first **tokenize** the text into words (or tokens), and assign a unique identifier to each word found in the text. Doing this will allow us to perform the encoding. We can do this easily by making use of the **Tokenizer** class in keras:


```python
from keras.preprocessing.text import Tokenizer
```

A Tokenizer offers convenient methods to split texts down to tokens. At construction time we need to supply the Tokenizer the maximum number of different words we are willing to represent. If out texts have greater word variety than this number, the least frequent words will be discarded. We will choose a number large enough for our purpose.


```python
maxwords = 1000
tokenizer = Tokenizer(num_words = maxwords, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
```

We now need to **fit** the Tokenizer to the training texts.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Find in the keras documentation the appropriate Tokenizer method to fit the tokenizer on a list of text, then use it to fit it on the training data.
 </td></tr>
</table>


```python
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)
```

If done correctly, the following should show the number of times the tokenizer has found each word in the input texts.


```python
tokenizer.word_counts
```




    OrderedDict([('but', 8813),
                 ('working', 792),
                 ('i', 72637),
                 ('surprised', 802),
                 ('see', 11461),
                 ('many', 6672),
                 ('people', 9103),
                 ('consider', 512),
                 ('good', 15100),
                 ('on', 3422),
                 ('grounds', 62),
                 ('there', 6700),
                 ('loose', 277),
                 ('hints', 103),
                 ('whole', 3078),
                 ('material', 759),
                 ('self', 1184),
                 ('indulgent', 80),
                 ('unconvincing', 186),
                 ("lynch's", 46),
                 ('movies', 7649),
                 ('generally', 468),
                 ('intriguing', 301),
                 ('generate', 50),
                 ('sense', 2322),
                 ('confusion', 163),
                 ('yet', 2748),
                 ('playful', 40),
                 ('that', 5232),
                 ('visual', 522),
                 ('subplots', 87),
                 ('characters', 7055),
                 ('ideas', 593),
                 ('etc', 1212),
                 ('dull', 816),
                 ('yes', 1527),
                 ('pointless', 505),
                 ('because', 688),
                 ('whatever', 729),
                 ('explore', 118),
                 ('either', 1866),
                 ('small', 1641),
                 ('far', 2978),
                 ('fetched', 103),
                 ('simply', 1966),
                 ('told', 1063),
                 ('superior', 310),
                 ('manner', 409),
                 ("it's", 6427),
                 ('lynch', 232),
                 ('exploring', 66),
                 ('dv', 23),
                 ('nothing', 4272),
                 ('treated', 275),
                 ('like', 20272),
                 ('this', 18776),
                 ('1', 2201),
                 ('10', 4330),
                 ('as', 4418),
                 ('rule', 182),
                 ('full', 1772),
                 ('moon', 278),
                 ('production', 1785),
                 ('logo', 31),
                 ('warning', 315),
                 ('sign', 273),
                 ('avoid', 776),
                 ('film', 39098),
                 ("i've", 3342),
                 ('enjoyed', 1245),
                 ('jeffrey', 105),
                 ('combs', 45),
                 ('films', 6868),
                 ('gave', 1217),
                 ('shot', 2047),
                 ('br', 101871),
                 ('bad', 9270),
                 ('not', 4350),
                 ('great', 9045),
                 ("that's", 3410),
                 ('something', 5052),
                 ('else', 1935),
                 ('the', 49126),
                 ('involves', 224),
                 ('struggle', 327),
                 ('mystic', 26),
                 ('evil', 1432),
                 ('brother', 1033),
                 ('wants', 1287),
                 ('dominate', 43),
                 ('worlds', 141),
                 ('title', 1496),
                 ('character', 6704),
                 ('dr', 689),
                 ('mordrid', 25),
                 ('also', 9157),
                 ('deal', 717),
                 ('authorities', 65),
                 ('mundane', 85),
                 ('world', 3686),
                 ('successfully', 161),
                 ('possible', 1000),
                 ('spoilers', 582),
                 ('follow', 782),
                 ('travel', 246),
                 ('dimensions', 27),
                 ('find', 4127),
                 ('companion', 102),
                 ('guarding', 7),
                 ('fortress', 18),
                 ('however', 3537),
                 ('guard', 164),
                 ('blinded', 31),
                 ('his', 1680),
                 ('eyes', 1208),
                 ('ruined', 227),
                 ('pits', 28),
                 ('so', 4254),
                 ('wizard', 91),
                 ('passes', 106),
                 ('hands', 632),
                 ('across', 970),
                 ("other's", 76),
                 ('hey', 401),
                 ('presto', 7),
                 ('restored', 90),
                 ('sort', 1469),
                 ('healing', 36),
                 ('apparently', 917),
                 ('works', 1279),
                 ('later', 2193),
                 ('mordred', 1),
                 ('animate', 14),
                 ('couple', 1698),
                 ('animal', 333),
                 ('skeletons', 18),
                 ('museum', 93),
                 ('fight', 1139),
                 ('guess', 1311),
                 ('one', 26516),
                 ('wins', 163),
                 ('side', 1269),
                 ('picture', 1472),
                 ('though', 4565),
                 ('much', 9760),
                 ('comic', 897),
                 ('book', 2395),
                 ("mordred's", 1),
                 ('human', 1587),
                 ('adventures', 201),
                 ('okay', 713),
                 ('plays', 2211),
                 ('role', 3182),
                 ('convincingly', 93),
                 ('seen', 6678),
                 ('lots', 798),
                 ('worse', 1468),
                 ("movie's", 423),
                 ('reputation', 191),
                 ('precedes', 12),
                 ('it', 24634),
                 ('anticipation', 72),
                 ('sat', 289),
                 ('watch', 6971),
                 ('letterbox', 4),
                 ('tcm', 51),
                 ('what', 3666),
                 ('major', 926),
                 ('disappointment', 404),
                 ('cast', 3816),
                 ('superb', 671),
                 ('values', 465),
                 ('first', 9057),
                 ('rate', 626),
                 ('without', 3267),
                 ('depth', 510),
                 ('plot', 6551),
                 ('thin', 360),
                 ('thing', 4499),
                 ('goes', 2442),
                 ('long', 3442),
                 ('for', 3268),
                 ('movie', 43568),
                 ('deals', 257),
                 ('alcoholism', 37),
                 ('family', 3098),
                 ('divisions', 5),
                 ('unfaithfulness', 6),
                 ('gambling', 73),
                 ('sexual', 714),
                 ('repression', 21),
                 ('curiously', 49),
                 ('flat', 577),
                 ('prosaic', 11),
                 ('lifeless', 65),
                 ('cliche', 84),
                 ('ridden', 88),
                 ('example', 1374),
                 ('portrayal', 507),
                 ('frank', 450),
                 ("hirsch's", 1),
                 ('unfaithfuness', 1),
                 ('rather', 2734),
                 ('heavy', 487),
                 ('handed', 182),
                 ('request', 33),
                 ('wife', 2057),
                 ('go', 5147),
                 ('upstairs', 46),
                 ('relax', 78),
                 ('bit', 3053),
                 ('followed', 373),
                 ('predictable', 853),
                 ('pleading', 10),
                 ('headache', 61),
                 ('leads', 746),
                 ('even', 12652),
                 ('predictably', 55),
                 ('evening', 235),
                 ('liaison', 17),
                 ('secretary', 135),
                 ('nancy', 209),
                 ('got', 3585),
                 ('blues', 99),
                 ('tonight', 94),
                 ("let's", 667),
                 ('drive', 445),
                 ('according', 296),
                 ('well', 10655),
                 ('worn', 88),
                 ('formula', 252),
                 ('we', 2060),
                 ('feel', 2943),
                 ('real', 4717),
                 ('cardboard', 131),
                 ('cutouts', 15),
                 ('acting', 6479),
                 ('marionette', 2),
                 ('play', 2226),
                 ('source', 206),
                 ('obvious', 1066),
                 ('friction', 15),
                 ('dave', 115),
                 ('hirsch', 15),
                 ('never', 6480),
                 ('really', 11736),
                 ('explored', 106),
                 ('explained', 284),
                 ("dave's", 2),
                 ('infatuation', 14),
                 ('again', 2290),
                 ('off', 1521),
                 ('gwen', 12),
                 ('inexplicable', 68),
                 ('light', 970),
                 ('fatuous', 6),
                 ('inability', 79),
                 ('defecate', 1),
                 ('get', 9270),
                 ('pot', 98),
                 ('subsequent', 121),
                 ('marriage', 416),
                 ('desperation', 104),
                 ('shirley', 97),
                 ('maclaine', 19),
                 ('ginny', 4),
                 ('is', 4085),
                 ('moment', 1104),
                 ('presented', 415),
                 ('viewer', 1195),
                 ('anyway', 1117),
                 ('obviously', 1163),
                 ('doomed', 104),
                 ('fail', 284),
                 ('clear', 786),
                 ('conventions', 72),
                 ('type', 1121),
                 ('soap', 285),
                 ('opera', 393),
                 ('could', 7747),
                 ('resolved', 60),
                 ('someone', 2241),
                 ('killed', 1111),
                 ('jealous', 121),
                 ('lover', 379),
                 ('started', 963),
                 ('running', 992),
                 ('around', 3615),
                 ('gun', 554),
                 ('bet', 242),
                 ('would', 12238),
                 ('phony', 83),
                 ('capital', 87),
                 ("'p'", 2),
                 ('having', 384),
                 ('said', 2196),
                 ("maclaine's", 3),
                 ('performance', 2897),
                 ('dean', 184),
                 ('martin', 342),
                 ('standouts', 15),
                 ('here', 2783),
                 ('interest', 1029),
                 ('purely', 169),
                 ('period', 766),
                 ('piece', 1533),
                 ('hollywood', 1796),
                 ('history', 1323),
                 ('1942', 40),
                 ('standards', 354),
                 ('making', 2951),
                 ('setup', 64),
                 ('her', 2579),
                 ('presents', 207),
                 ('dated', 266),
                 ('extreme', 349),
                 ('machinations', 24),
                 ('half', 2092),
                 ('pair', 238),
                 ('of', 2878),
                 ('husband', 949),
                 ('ex', 465),
                 ('back', 4959),
                 ('threat', 115),
                 ('another', 4320),
                 ('divorce', 101),
                 ('eventual', 59),
                 ('separation', 30),
                 ('means', 761),
                 ('jealousy', 57),
                 ('humiliation', 29),
                 ('schemes', 32),
                 ('done', 3095),
                 ('better', 5739),
                 ('classics', 229),
                 ('girl', 2722),
                 ('friday', 193),
                 ('philadelphia', 34),
                 ('story', 11896),
                 ('both', 595),
                 ('features', 640),
                 ('women', 1731),
                 ('strong', 1097),
                 ('indomitable', 4),
                 ('screen', 2479),
                 ('presence', 410),
                 ('played', 2588),
                 ('independent', 312),
                 ('proto', 10),
                 ('feminist', 83),
                 ('in', 8358),
                 ('estranged', 59),
                 ('divorced', 60),
                 ('witty', 273),
                 ('husbands', 76),
                 ('set', 2445),
                 ('marry', 225),
                 ('colorless', 11),
                 ('men', 1843),
                 ('exact', 189),
                 ('opposite', 266),
                 ('bamboozled', 6),
                 ('rejecting', 12),
                 ('soon', 1223),
                 ('to', 3141),
                 ('be', 1437),
                 ('re', 747),
                 ('igniting', 5),
                 ('passion', 295),
                 ('other', 1163),
                 ('switches', 38),
                 ('gender', 90),
                 ('norma', 32),
                 ('shearer', 25),
                 ('cary', 106),
                 ('grant', 246),
                 ('out', 2725),
                 ('time', 12684),
                 ('ward', 121),
                 ('boyfriend', 390),
                 ('george', 840),
                 ('sanders', 52),
                 ('hiring', 32),
                 ('robert', 939),
                 ('taylor', 285),
                 ('pose', 47),
                 ('gigolo', 21),
                 ('problem', 1447),
                 ('old', 4498),
                 ('playing', 1631),
                 ('suited', 111),
                 ('actress', 1209),
                 ('mid', 318),
                 ('late', 1210),
                 ('twenties', 29),
                 ('involved', 1076),
                 ('furniture', 55),
                 ('man', 5569),
                 ('love', 6392),
                 ('fiancée', 43),
                 ('seeing', 2098),
                 ('strange', 926),
                 ('come', 3185),
                 ('bathroom', 114),
                 ('happens', 1080),
                 ('knock', 139),
                 ('lights', 181),
                 ('cause', 479),
                 ('huge', 944),
                 ('scene', 5349),
                 ('and', 11449),
                 ('part', 4027),
                 ('trying', 2469),
                 ('channel', 431),
                 ('speech', 200),
                 ('inflections', 10),
                 ('overall', 1437),
                 ('essence', 138),
                 ('worst', 2727),
                 ('herself', 221),
                 ('used', 1879),
                 ('parts', 1191),
                 ('intellectual', 175),
                 ('sexiness', 8),
                 ('dramatic', 666),
                 ('consuelo', 3),
                 ('craydon', 1),
                 ('seems', 3619),
                 ('put', 2379),
                 ('throes', 10),
                 ('complete', 1032),
                 ('over', 1454),
                 ('emoting', 19),
                 ('gesturing', 3),
                 ('which', 1387),
                 ('still', 5619),
                 ('style', 1593),
                 ('appropriate', 221),
                 ('ten', 828),
                 ('years', 4503),
                 ('earlier', 663),
                 ('makes', 4201),
                 ('look', 4139),
                 ('extremely', 1069),
                 ('mannered', 47),
                 ('performer', 101),
                 ('wrenching', 62),
                 ('joke', 620),
                 ('situation', 666),
                 ('water', 533),
                 ('fairly', 587),
                 ('dry', 229),
                 ('sponge', 13),
                 ('fuels', 11),
                 ('fires', 44),
                 ('tell', 1714),
                 ('theory', 189),
                 ('gives', 1577),
                 ('irving', 40),
                 ('thalberg', 6),
                 ('maker', 157),
                 ('career', 1004),
                 ('chooser', 1),
                 ('most', 925),
                 ('roles', 1112),
                 ('passed', 246),
                 ('charlotte', 93),
                 ('vale', 3),
                 ('mrs', 259),
                 ('miniver', 3),
                 ('mega', 42),
                 ('hits', 271),
                 ('now', 2242),
                 ('voyager', 43),
                 ('mystery', 844),
                 ('accounts', 57),
                 ('state', 524),
                 ('burnt', 49),
                 ("she'd", 58),
                 ('lost', 1537),
                 ('altogether', 112),
                 ('secret', 601),
                 ('anyone', 2572),
                 ('experienced', 191),
                 ('essentially', 257),
                 ('focus', 505),
                 ("can't", 3531),
                 ('wait', 717),
                 ('retirement', 40),
                 ('end', 5643),
                 ('contract', 129),
                 ('near', 822),
                 ('leave', 1099),
                 ('such', 466),
                 ('case', 1527),
                 ('she', 2725),
                 ('tired', 380),
                 ('ill', 291),
                 ('ease', 110),
                 ('going', 4094),
                 ('autopilot', 9),
                 ('instead', 2191),
                 ('living', 1061),
                 ('after', 1880),
                 ('make', 8014),
                 ('more', 1298),
                 ('responsible', 276),
                 ('discovering', 75),
                 ('janet', 46),
                 ('leigh', 66),
                 ('star', 2051),
                 ('40s', 57),
                 ('60s', 136),
                 ('fan', 1901),
                 ('mormon', 54),
                 ('being', 402),
                 ('always', 3237),
                 ('felt', 1527),
                 ('humor', 1305),
                 ('exclusive', 27),
                 ('lds', 37),
                 ('community', 289),
                 ('made', 8356),
                 ('us', 3786),
                 ('seem', 2174),
                 ('bunch', 810),
                 ('obsessive', 56),
                 ('wackos', 4),
                 ('hoping', 406),
                 ('breath', 177),
                 ('fresh', 374),
                 ('air', 639),
                 ('halestorm', 8),
                 ('finally', 1536),
                 ('discuss', 106),
                 ('non', 898),
                 ('friends', 1772),
                 ('boy', 1476),
                 ('wrong', 1817),
                 ('figured', 187),
                 ('since', 2906),
                 ('b', 1257),
                 ('list', 582),
                 ('talent', 929),
                 ('clint', 108),
                 ('howard', 230),
                 ('gary', 261),
                 ('coleman', 33),
                 ('andrew', 140),
                 ('wilson', 188),
                 ('fred', 295),
                 ('willard', 25),
                 ('favorites', 187),
                 ('least', 3113),
                 ('little', 6426),
                 ('funny', 4276),
                 ('besides', 410),
                 ('church', 395),
                 ('basketball', 88),
                 ('ripe', 24),
                 ('potential', 612),
                 ('plenty', 631),
                 ('hilarious', 968),
                 ('gags', 262),
                 ('must', 3198),
                 ('say', 5391),
                 ('throughout', 1361),
                 ('entire', 1461),
                 ('seemed', 1363),
                 ('knew', 898),
                 ('doing', 190),
                 ('every', 3975),
                 ('fell', 346),
                 ('opportunity', 389),
                 ('genuinely', 251),
                 ('gag', 139),
                 ('went', 1462),
                 ('ignored', 120),
                 ('dialogue', 1541),
                 ('bland', 274),
                 ('development', 641),
                 ('ever', 5987),
                 ('single', 916),
                 ("wilson's", 12),
                 ('less', 1998),
                 ('dimensional', 255),
                 ('hard', 2665),
                 ('believe', 2504),
                 ('nine', 157),
                 ('writes', 94),
                 ('mind', 1972),
                 ('numbingly', 36),
                 ('stale', 87),
                 ('train', 409),
                 ('wreck', 127),
                 ('witnessed', 88),
                 ('words', 882),
                 ('rage', 109),
                 ('sitting', 450),
                 ('my', 2326),
                 ('extras', 227),
                 ('final', 1324),
                 ('game', 1262),
                 ('premiere', 78),
                 ('washington', 250),
                 ('city', 1156),
                 ('ut', 2),
                 ('kurt', 144),
                 ('hale', 38),
                 ('director', 4184),
                 ('avoided', 102),
                 ('contact', 148),
                 ('show', 6167),
                 ('he', 5278),
                 ('waited', 95),
                 ('door', 428),
                 ('seemingly', 348),
                 ('ready', 334),
                 ('feedback', 10),
                 ('bring', 867),
                 ('ripped', 138),
                 ('away', 2765),
                 ('hour', 1183),
                 ('life', 6557),
                 ('left', 2123),
                 ('nasty', 339),
                 ('painful', 415),
                 ('scar', 19),
                 ('forget', 717),
                 ('specific', 135),
                 ('problems', 887),
                 ('had', 412),
                 ('minor', 400),
                 ('subplot', 121),
                 ('janitor', 35),
                 ('chubby', 26),
                 ('piano', 122),
                 ('player', 295),
                 ('two', 6889),
                 ('came', 1671),
                 ('nowhere', 442),
                 ('impossible', 494),
                 ('care', 1381),
                 ('about', 1085),
                 ('constantly', 416),
                 ('wondering', 359),
                 ('supposed', 1515),
                 ('lame', 741),
                 ('uninteresting', 199),
                 ('popped', 48),
                 ('then', 2143),
                 ('promising', 207),
                 ('audience', 2145),
                 ('chance', 1065),
                 ('laughs', 658),
                 ('puff', 10),
                 ('smoke', 118),
                 ('ending', 2352),
                 ('start', 1697),
                 ('caring', 165),
                 ('pretty', 3661),
                 ('letdown', 43),
                 ('everyone', 2125),
                 ("who's", 707),
                 ('expecting', 588),
                 ('true', 2322),
                 ('jokes', 970),
                 ('mormons', 29),
                 ('loud', 436),
                 ('ringing', 28),
                 ('sensation', 38),
                 ('ears', 98),
                 ('please', 1047),
                 ('keep', 1597),
                 ('fantasy', 644),
                 ('spare', 107),
                 ('oh', 1425),
                 ('dear', 143),
                 ('oireland', 1),
                 ('religion', 234),
                 ('no', 2631),
                 ('doubt', 754),
                 ("we'll", 113),
                 ('depressing', 225),
                 ('nonsense', 287),
                 ('featuring', 277),
                 ('hunky', 39),
                 ('macho', 70),
                 ('freedom', 235),
                 ('fighters', 39),
                 ('ira', 53),
                 ('initial', 209),
                 ('reaction', 248),
                 ('credits', 671),
                 ('shock', 377),
                 ('starts', 1220),
                 ('day', 2698),
                 ('wedding', 304),
                 ('sean', 254),
                 ('cloney', 3),
                 ('sheila', 31),
                 ('kelly', 373),
                 ('1950s', 154),
                 ('slight', 135),
                 ("they're", 1249),
                 ('getting', 1624),
                 ('married', 585),
                 ('catholic', 152),
                 ('protestant', 14),
                 ('order', 946),
                 ('happen', 1041),
                 ('takes', 2192),
                 ('pledge', 10),
                 ('children', 1331),
                 ('brought', 737),
                 ('attend', 83),
                 ('school', 1632),
                 ('enough', 3450),
                 ('jumps', 160),
                 ('forward', 651),
                 ('daughters', 170),
                 ('decided', 705),
                 ("they'll", 121),
                 ('attending', 53),
                 ('local', 876),
                 ('disgust', 62),
                 ('priest', 219),
                 ('father', 1933),
                 ('stafford', 5),
                 ('from', 1255),
                 ('things', 3680),
                 ('escalate', 6),
                 ('let', 1665),
                 ('cards', 99),
                 ('table', 180),
                 ('despite', 1365),
                 ('irish', 195),
                 ('scottish', 93),
                 ('heritage', 29),
                 ('agnostic', 9),
                 ('considered', 484),
                 ('atheist', 21),
                 ('adult', 501),
                 ('fact', 3523),
                 ('comes', 2484),
                 ('marxist', 20),
                 ('cynical', 154),
                 ('weapon', 145),
                 ('manipulate', 37),
                 ('a', 7948),
                 ('divided', 47),
                 ('shows', 2305),
                 ('appointed', 24),
                 ('moral', 363),
                 ('guardians', 9),
                 ('take', 3505),
                 ('upon', 859),
                 ('think', 7297),
                 ('may', 3364),
                 ('temerity', 5),
                 ('karl', 77),
                 ('marx', 35),
                 ('saw', 3166),
                 ("he'd", 180),
                 ('call', 919),
                 ('masterpiece', 608),
                 ('perhaps', 1684),
                 ('drama', 1402),
                 ('thinking', 1177),
                 ('reply', 26),
                 ('reviewers', 262),
                 ('claimed', 100),
                 ('propaganda', 203),
                 ('claim', 222),
                 ('know', 6157),
                 ('details', 410),
                 ('happened', 1076),
                 ('county', 49),
                 ('wexford', 2),
                 ("there's", 3084),
                 ('denying', 38),
                 ('flock', 53),
                 ('sheep', 43),
                 ('portrayed', 601),
                 ('guys', 1287),
                 ('blameless', 3),
                 ('woman', 2646),
                 ('rural', 110),
                 ('village', 253),
                 ('ireland', 110),
                 ('catholics', 16),
                 ('changes', 386),
                 ('believes', 228),
                 ('consequences', 123),
                 ('taking', 953),
                 ('pledges', 3),
                 ('keeping', 276),
                 ('disappears', 72),
                 ('pick', 451),
                 ('pieces', 423),
                 ('shattered', 28),
                 ('lives', 1386),
                 ('picked', 330),
                 ('former', 509),
                 ('andy', 295),
                 ('bailey', 24),
                 ('shown', 994),
                 ('gallant', 4),
                 ('member', 322),
                 ('change', 959),
                 ("we're", 526),
                 ('talking', 945),
                 ("devil's", 60),
                 ('own', 434),
                 ('lot', 3966),
                 ('agree', 572),
                 ('if', 5896),
                 ('criticism', 173),
                 ('feels', 809),
                 ('tvm', 11),
                 ('cinematic', 412),
                 ('live', 1544),
                 ('essential', 162),
                 ('viewing', 750),
                 ('thinks', 437),
                 ('opium', 7),
                 ('masses', 81),
                 ('sure', 2686),
                 ('oldboy', 3),
                 ('days', 1256),
                 ('amazingly', 174),
                 ('shocking', 334),
                 ('high', 2143),
                 ('budget', 1830),
                 ('hyped', 72),
                 ('way', 8020),
                 ('spin', 152),
                 ('kick', 265),
                 ('comedy', 3220),
                 ('group', 1025),
                 ('decide', 482),
                 ('pour', 35),
                 ('hearts', 135),
                 ('tae', 3),
                 ('kwon', 6),
                 ('do', 1754),
                 ('regardless', 125),
                 ('expect', 1177),
                 ('guaranteed', 72),
                 ('moved', 322),
                 ('work', 4371),
                 ('pain', 379),
                 ('expectations', 401),
                 ('force', 506),
                 ('experience', 1057),
                 ('comedic', 314),
                 ('times', 3233),
                 ('moments', 1662),
                 ('rendered', 68),
                 ('beautifully', 436),
                 ('seriously', 1002),
                 ('hoodlum', 12),
                 ('turned', 925),
                 ('guy', 2944),
                 ('second', 1958),
                 ('meek', 42),
                 ('team', 804),
                 ('substitute', 46),
                 ('die', 788),
                 ('happy', 957),
                 ('rounded', 69),
                 ('importantly', 127),
                 ('hopes', 273),
                 ('dreams', 432),
                 ('while', 1769),
                 ('goals', 56),
                 ('simple', 1021),
                 ('aspects', 398),
                 ('merely', 359),
                 ('highlight', 202),
                 ('overcome', 153),
                 ('personal', 628),
                 ('inter', 46),
                 ('struggles', 155),
                 ('short', 1862),
                 ('feeling', 1142),
                 ('determined', 164),
                 ('satisfied', 106),
                 ('amazing', 1319),
                 ('you', 4760),
                 ('truly', 1743),
                 ('lived', 382),
                 ('tragic', 347),
                 ('events', 910),
                 ('cry', 395),
                 ('along', 1775),
                 ('passionate', 97),
                 ('himself', 638),
                 ('ensure', 57),
                 ('others', 1578),
                 ('survive', 260),
                 ('wretched', 79),
                 ('video', 1718),
                 ('down', 880),
                 ('manages', 582),
                 ('brighter', 21),
                 ('jonny', 21),
                 ('did', 1063),
                 ('unbearable', 116),
                 ('regret', 189),
                 ('knowing', 446),
                 ('sooner', 72),
                 ('visited', 75),
                 ('england', 289),
                 ('2', 2858),
                 ('able', 1258),
                 ("i'd", 1346),
                 ('met', 287),
                 ('him', 2520),
                 ('comforting', 21),
                 ('cloud', 33),
                 ('free', 695),
                 ('rest', 1802),
                 ('peace', 202),
                 ('deserve', 287),
                 ('given', 1848),
                 ('stardust', 50),
                 ('course', 2505),
                 ('magically', 46),
                 ('fairy', 204),
                 ('tale', 786),
                 ('princess', 194),
                 ('bride', 131),
                 ('definitely', 1580),
                 ('wonderful', 1655),
                 ('spectacle', 58),
                 ("2000's", 12),
                 ("1990's", 42),
                 ('exciting', 515),
                 ('equipped', 17),
                 ('imagery', 183),
                 ('unforgettable', 144),
                 ('michelle', 169),
                 ('pfeiffer', 53),
                 ("deniro's", 8),
                 ('especially', 2536),
                 ('challenge', 164),
                 ('smile', 289),
                 ('minutes', 2944),
                 ('perfectly', 637),
                 ('journey', 428),
                 ('destination', 49),
                 ('enthralls', 1),
                 ('finish', 411),
                 ('stars', 1684),
                 ('decimal', 4),
                 ('cinematographical', 2),
                 ('buffs', 98),
                 ('rank', 102),
                 ('anything', 2944),
                 ('profound', 138),
                 ('truth', 692),
                 ('intentions', 159),
                 ('series', 3389),
                 ('understand', 1644),
                 ('p', 326),
                 ('o', 300),
                 ('v', 266),
                 ('granted', 201),
                 ('specifics', 14),
                 ('renderings', 4),
                 ('writer', 1098),
                 ('cannot', 1097),
                 ('expected', 704),
                 ('biblically', 4),
                 ('accurate', 283),
                 ('justifiably', 17),
                 ('scares', 188),
                 ('viewers', 776),
                 ("i'm", 4736),
                 ('christian', 370),
                 ('due', 909),
                 ('decision', 239),
                 ('accept', 300),
                 ('jesus', 267),
                 ('savior', 28),
                 ('similar', 852),
                 ('circumstances', 218),
                 ('therein', 30),
                 ('remarkably', 105),
                 ('scare', 217),
                 ('actions', 311),
                 ('decisions', 104),
                 ('cheap', 890),
                 ('attempt', 1049),
                 ('believing', 136),
                 ('god', 1110),
                 ('attention', 906),
                 ("i'll", 973),
                 ('behind', 1278),
                 ...])



Now we have trained the tokenizer we can use it to vectorize the texts. In particular, we would like to transform the texts to sequences of word indexes.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Find in the keras documentation the appropriate Tokenizer method to transform a list of texts to a sequence. Apply it to both the training and test data to obtain matrices **X_train** and **X_test**.
 </td></tr>
</table>


```python
X_train=tokenizer.texts_to_sequences(X_train)    #####  Texts -> sequences fo integers
X_test=tokenizer.texts_to_sequences(X_test)
```

We can see now how a text has been transformed to a list of word indexes.


```python
X_train[0]
```




    [26,
     723,
     2,
     714,
     16,
     42,
     23,
     10,
     117,
     40,
     40,
     146,
     761,
     470,
     33,
     208,
     167,
     63,
     40,
     208,
     35,
     957,
     460,
     26,
     697,
     354,
     828,
     788,
     276,
     327,
     276,
     150,
     261,
     520,
     47,
     84,
     8,
     9,
     229,
     80]



This is enough to train a Sequential Network. However, for efficiency reasons it is recommended that all sequences in the data have the same number of elements. Since this is not the case for our data, should **pad** the sequences to ensure the same length. The padding procedure adds a special *null* symbol to short sequences, and clips out parts of long sequences, thus enforcing a common size.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Find in the keras documentation the appropriate text preprocessing method to pad a sequence. Then pad all sequences to have a maximum of 300 words, both in the training and test data.
 </td></tr>
</table>


```python
X_train = sequence.pad_sequences(X_train, maxlen=300)  ###padding/truncating
X_test = sequence.pad_sequences(X_test, maxlen=300)    ### 300 words
```

## Simple LSTM network with Embedding

To transform the word indices into something more amenable for a network we will use an <a href=https://keras.io/layers/embeddings/>**Embedding**</a> layer at the very beginning of the network. This layer will transform word indexes to a vector representation that is learned with the model together with the rest of network weights. After this transformation we will make use of an <a href=https://keras.io/layers/recurrent/#lstm>**LSTM**</a> layer to analyze the whole sequence, and then a final layer taking the decision of the network.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Build, compile and train a keras network with the following structure:
<ul>
 <li>Embedding layer producing a vector representation of 64 elements</li>
 <li>LSTM layer of 32 units</li>
 <li>Dropout of 0.9</li>
 <li>Dense layer of 1 unit with sigmoid activation</li>
</ul>
Note that the Embedding layer requires specifing as first argument the maximum number of words we chose to for the tokenizer. Also, the LSTM layer requires setting the **input_length** parameter as the number of elements in the input sequences. 
Use the binary crossentropy loss function for training, together with the adam optimizer. Train for 10 epochs. After training, measure the accuracy on the test set.
 </td></tr>
</table>


```python
vocab_size=len(tokenizer.word_index)
vocab_size
```




    88073




```python
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import GRU, LSTM, GlobalMaxPool1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense



model = Sequential()
model.add(Embedding(vocab_size+1, 64,input_length=300))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.
model.add(LSTM(32))
model.add(Dropout(0.50))

model.add(Dense(1))
model.add(Activation('sigmoid'))
```


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
  X_train, # Training data
  y_train, # Labels of training data
  batch_size=128, # Batch size for the optimizer algorithm
  epochs=10, # Number of epochs to run the optimizer algorithm
  verbose=2, # Level of verbosity of the log messages
  validation_data=(X_test,y_test)
)
```

    Train on 18750 samples, validate on 6250 samples
    Epoch 1/10
     - 53s - loss: 0.5806 - acc: 0.7078 - val_loss: 0.3823 - val_acc: 0.8440
    Epoch 2/10
     - 51s - loss: 0.3535 - acc: 0.8546 - val_loss: 0.3348 - val_acc: 0.8622
    Epoch 3/10
     - 50s - loss: 0.3174 - acc: 0.8726 - val_loss: 0.3397 - val_acc: 0.8579
    Epoch 4/10
     - 52s - loss: 0.3029 - acc: 0.8780 - val_loss: 0.3297 - val_acc: 0.8590
    Epoch 5/10
     - 52s - loss: 0.2938 - acc: 0.8822 - val_loss: 0.3472 - val_acc: 0.8581
    Epoch 6/10
     - 50s - loss: 0.2828 - acc: 0.8881 - val_loss: 0.3343 - val_acc: 0.8558
    Epoch 7/10
     - 50s - loss: 0.2677 - acc: 0.8922 - val_loss: 0.3379 - val_acc: 0.8565
    Epoch 8/10
     - 49s - loss: 0.2573 - acc: 0.8969 - val_loss: 0.3458 - val_acc: 0.8571
    Epoch 9/10
     - 50s - loss: 0.2516 - acc: 0.8979 - val_loss: 0.3671 - val_acc: 0.8539
    Epoch 10/10
     - 49s - loss: 0.2422 - acc: 0.9028 - val_loss: 0.3651 - val_acc: 0.8526
    




    <keras.callbacks.History at 0x25561514d30>



test_score=0.8590

## Stacked LSTMs

Much like other neural layers, LSTM layers can be stacked on top of each other to produce more complex models. Care must be taken, however, that the LSTM layers before the last one generate a whole sequence of outputs for the following LSTM to process.

<table>
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Repeat the training of the previous network, but using 2 LSTM layers. Make sure to configure the first LSTM layer in a way that it outputs a whole sequence for the next layer.
 </td></tr>
</table>


```python
model = Sequential()
model.add(Embedding(vocab_size+1, 64,input_length=300))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.50))

model.add(Dense(1))
model.add(Activation('sigmoid'))
```


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
  X_train, # Training data
  y_train, # Labels of training data
  batch_size=128, # Batch size for the optimizer algorithm
  epochs=20, # Number of epochs to run the optimizer algorithm
  verbose=2, # Level of verbosity of the log messages
  validation_data=(X_test,y_test)
)
```

    Train on 18750 samples, validate on 6250 samples
    Epoch 1/20
     - 277s - loss: 0.4929 - acc: 0.7537 - val_loss: 0.3903 - val_acc: 0.8274
    Epoch 2/20
     - 285s - loss: 0.3411 - acc: 0.8575 - val_loss: 0.3342 - val_acc: 0.8589
    Epoch 3/20
     - 286s - loss: 0.3296 - acc: 0.8665 - val_loss: 0.3376 - val_acc: 0.8547
    Epoch 4/20
     - 289s - loss: 0.3066 - acc: 0.8746 - val_loss: 0.3493 - val_acc: 0.8538
    Epoch 5/20
     - 291s - loss: 0.2907 - acc: 0.8812 - val_loss: 0.3413 - val_acc: 0.8576
    Epoch 6/20
     - 293s - loss: 0.2741 - acc: 0.8903 - val_loss: 0.3435 - val_acc: 0.8520
    Epoch 7/20
     - 295s - loss: 0.2617 - acc: 0.8941 - val_loss: 0.3672 - val_acc: 0.8530
    Epoch 8/20
     - 297s - loss: 0.2529 - acc: 0.8979 - val_loss: 0.3531 - val_acc: 0.8494
    Epoch 9/20
     - 297s - loss: 0.2490 - acc: 0.9017 - val_loss: 0.3639 - val_acc: 0.8469
    Epoch 10/20
     - 295s - loss: 0.2391 - acc: 0.9041 - val_loss: 0.3815 - val_acc: 0.8490
    Epoch 11/20
     - 298s - loss: 0.2263 - acc: 0.9097 - val_loss: 0.4204 - val_acc: 0.8456
    Epoch 12/20
     - 299s - loss: 0.2198 - acc: 0.9127 - val_loss: 0.3815 - val_acc: 0.8466
    Epoch 13/20
     - 300s - loss: 0.2193 - acc: 0.9135 - val_loss: 0.4046 - val_acc: 0.8435
    Epoch 14/20
     - 303s - loss: 0.1973 - acc: 0.9225 - val_loss: 0.4312 - val_acc: 0.8381
    Epoch 15/20
     - 305s - loss: 0.1834 - acc: 0.9313 - val_loss: 0.4541 - val_acc: 0.8389
    Epoch 16/20
     - 306s - loss: 0.1818 - acc: 0.9300 - val_loss: 0.4526 - val_acc: 0.8277
    Epoch 17/20
     - 305s - loss: 0.1617 - acc: 0.9406 - val_loss: 0.4780 - val_acc: 0.8226
    Epoch 18/20
     - 308s - loss: 0.1462 - acc: 0.9461 - val_loss: 0.5413 - val_acc: 0.8392
    Epoch 19/20
     - 311s - loss: 0.1584 - acc: 0.9401 - val_loss: 0.5257 - val_acc: 0.8339
    Epoch 20/20
     - 312s - loss: 0.1378 - acc: 0.9504 - val_loss: 0.5409 - val_acc: 0.8328
    




    <keras.callbacks.History at 0x255533c1550>



test_score=0.8589
