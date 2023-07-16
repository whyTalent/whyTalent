# NLP - 词嵌入Word Embedding

⾃然语⾔是⼀套⽤来表达含义的复杂系统，在这套系统中，词是表义的基本单元。词向量是⽤来表⽰词的向量，也可被认为是词的特征向量或表征，因此**把词映射为实数域向量的技术（或 文本特征提取 技术）也叫词嵌⼊（word embedding）**，近年来词嵌⼊已逐渐成为⾃然语⾔处理的基础知识。

在NLP领域，文本表示是第一步，也是很重要的一步，通俗来说就是**把人类的语言符号转化为机器能够进行计算的数字**，因为普通的文本语言机器是看不懂的，必须通过转化来表征对应文本。早期是**基于规则**的方法进行转化，而现代的方法是**基于统计机器学习**的方法。

**数据决定了机器学习的上限，而算法只是尽可能逼近这个上限**，在本文中数据指的就是文本表示，所以弄懂文本表示的发展历程，对于NLP学习者来说是必不可少的。

<div align="center"><img src="imgs/NLP-basic.png" style="zoom:80%;" /></div>

文本特征提取有两个非常重要的模型：

- 词集模型：单词构成的集合，集合自然每个元素都只有一个，也即词集中的每个单词都只有一个
- 词袋模型：在词集的基础上如果一个单词在文档中出现不止一次，统计其出现的次数（频数）

两者本质上的区别，词袋是在词集的基础上增加了频率的维度，词集只关注有和没有，词袋还要关注有几个。

​    

# 一 离散表示     

## 1 One-hot

One-Hot编码，又称为一位有效编码（简称读热向量编码），也是特征工程中最常用的方法，主要是采用N位状态寄存器来对**N** **个状态**进行编码，每个状态都有独立的寄存器位，并且在任意时候只有一位有效。简而言之，one hot编码是将类别变量转换为机器学习算法易于利用的一种形式的过程。

One-hot编码步骤如下：

* 1）构造文本分词后的字典，每个分词是一个比特值，比特值为0或者1；
* 2）每个分词的文本表示为该分词的比特位为1，其余位为0的矩阵表示，即将每个单词在词汇表对应的index维度置为1，其它元素保持不变，就能得到最终的one-hot向量；

<div align="center"><img src="imgs/nlp-one-hot.png" style="zoom:80%;" /></div>

例如：

```python
// a. 文本内容
I am a Chinese and I love China

// b. 以上句子可以构造一个词典（词汇表）
{"I": 1, "am": 2, "a": 3, "Chinese": 4, "and": 5, "love": 6, "China": 7}

// c. 每个词典索引对应着比特位，那么利用One-hot可得到了每个单词的embedding
I:  [1, 0, 0, 0, 0, 0, 0]
am: [0, 1, 0, 0, 0, 0, 0]
... ...
China: [0, 0, 0, 0, 0, 0, 1]
  
// d. 最后，就可以送入到机器学习或者深度学习模型中进行后续处理
```

使用示例：

```python
def test_pandas():
    """ 基于Pandas 的 One-hot编码
    """
    df = pd.DataFrame({
        "col1": ["Sun", "Sun", "Moon", "Earth", "Moon", "Venus"]
    })

    # One-hot编码
    df_new = pd.get_dummies(df, columns=["col1"], prefix="Planet", dtype=int)
    print("The transform data using get_dummies")
    print(df_new)


def test_sklearn():
    """ 基于 Sklearn 的 One-hot编码
    """
    targets = np.array(
           ["red", "green", "blue", "yellow", "pink", "white"]
    )
    labelEnc = preprocessing.LabelEncoder()
    # 打标签, 字符串转数字
    new_target = labelEnc.fit_transform(targets)
    
    onehotEnc = preprocessing.OneHotEncoder()
    # 使用 fit() 拟合原始特征
    onehotEnc.fit(new_target.reshape(-1, 1))
    # 使用 one-hot encoding 将原始特征转换为新特征
    targets_trans = onehotEnc.transform(new_target.reshape(-1, 1))

    print("The original data")
    print(targets)
    print(new_target)
    print("The transform data using OneHotEncoder")
    print(targets_trans.toarray())


def test_sklearn_newest():
    """ 基于最新版本的Sklearn 的 One-hot编码
    在较新版本的 sklearn 中，不需要将字符串转换为 int，因为 OneHotEncoder 会自动执行此操作
    """
    x = [[11, "Seattle"], [22, "London"], [33, "Lahore"], [44, "Berlin"], [55, "Abuja"]]
    y = preprocessing.OneHotEncoder().fit_transform(x).toarray()
    print("The transform data using the newest OneHotEncoder")
    print(y)
```

   

One-hot表示文本信息的**缺点**：

> 1）**容易造成维度灾难（curse of dimensionality）**：随着语料库的增加，数据特征的维度会越来越大，从而产生一个维度很高，又很稀疏的矩阵，会使得计算代价变大；
>
> 2）one-hot编码假设单词与单词之间是独立的，**无法体现单词与单词的关系远近程度**，例如，如果需要判断“Chinese” 和 “China” 向量之间的相似性，得出的向量内积值为 0（[0,0,0,1,0,0,0]\*[0,0,0,0,0,0,1]=0），与对“Chinese”和“and”之间的相似性没有任何区别，因为 “Chinese” 和 “and” 之间的one-hot向量内积也为0（[0,0,0,1,0,0,0]\*[0,0,0,0,1,0,0]=0），而明显 “China” 与 “Chinese” 的关系要近于 “China” 与 “and” 的，这在one-hot编码方式中无法体现；

​      

## 2 传统词袋模型 BOW

词袋模型（Bag-of-words model），忽略掉文本的语法和语序等要素，将其仅仅看作是若干个词汇的集合，文档中每个单词的出现都是独立的。BoW使用一组无序的单词(words)来表达一段文字或一个文档。

<div align="center"><img src="imgs/nlp-bow.png" style="zoom:80%;" /></div>

**文档的向量表示可以直接将各词的词向量表示加和**，例如：

```python
// a. 文本内容
John likes to watch movies. Mary likes too.
John also likes to watch football games.

// b. 基于上述两个文档中出现的单词, 构造一个词典（词汇表）
{"John": 1, "likes": 2,"to": 3, "watch": 4, "movies": 5,"also": 6, "football": 7, "games": 8,"Mary": 9, "too": 10}

// c. 上面的词典中包含10个单词, 每个单词有唯一的索引, 那么每个文本可以使用一个10维的向量来表示
[1, 2, 1, 1, 1, 0, 0, 0, 1, 1]
[1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
```

使用示例：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

import pandas as pd

def test_bow():
    """基于 Sklearn 的 BOW 编码
    """
    # 文本物料
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    # Convert a collection of text documents to a matrix of token counts
    vectorizer = CountVectorizer()
    # 将原始特征转换为新特征
    new_corpus = vectorizer.fit_transform(corpus).toarray()
    print(new_corpus)
   	
def test_bow_predict():
  	"""本地构建验证集计算F1得分
  	"""
    train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)
		
    vectorizer = CountVectorizer(max_features=3000)
    train_test = vectorizer.fit_transform(train_df['text'])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
    # 0.74
```

从上可知，BoW编码后向量特点：

> 1）**向量维度**根据词典中不重复词的个数确定；
>
> 2）向量中每个元素顺序与原来文本中单词出现的顺序没有关系，**与词典中的顺序一一对应**；
>
> 3）向量中每个数字是词典中每个单词在文本中出现的频率，即**词频**表示；

​     

传统词袋模型同样有一些**缺点**：

> 1）词向量化后，词与词之间是有大小关系的，不一定词出现的越多，权重越大；
>
> 2）词与词之间是没有顺序关系的，因此无法表达语义；

​     

## 3 TF-IDF

TF-IDF（term frequency–inverse document frequency）是一种用于信息检索与数据挖掘的常用加权技术

> TF即词频（Term Frequency），IDF即逆文本频率指数（Inverse Document Frequency）

TF-IDF同时也是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度

> **字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降，即一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章**。

TF-IDF加权的各种形式常被搜索引擎应用，作为文件与用户查询之间相关程度的度量或评级，除了TF-IDF以外，因特网上的搜索引擎还会使用**基于链接分析的评级方法**，以确定文件在搜寻结果中出现的顺序。

​    

在一份给定的文件里，TF指的是某一个给定的词语在该文件中出现的频率，这个数字是对词数（term count）的归一化，以防止它偏向长的文件（同一个词语在长文件里可能会比短文件有更高的词数，而不管该词语重要与否）。因此，对于在某一特定文件里的词语w，它的重要性可表示为：

<div align="center"><img src="imgs/nlp-tf.png" style="zoom:80%;" /></div>

IDF是一个词语普遍重要性的度量，对于某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到：

<div align="center"><img src="imgs/nlp-itf.png" style="zoom:80%;" /></div>

分母之所以加1，是为了避免分母为0。那么，**TF−IDF = TF \* IDF**，从这个公式可以看出，当w在文档中出现的次数增大时，而TF-IDF的值是减小的，即某一特定文件内的**高词语频率**，以及该词语在整个文件集合中的**低文件频率**，可以产生出高权重的TF-IDF。因此，**TF-IDF倾向于过滤掉常见的词语，保留重要的词语**。

示例：

```python
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

# 本地构建验证集计算F1得分
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

**缺点：**还是没有把词与词之间的关系顺序表达出来

​     

## 4 n-gram模型

N-gram与Count Vectors类似，不过加入了相邻单词组合成为新的单词，并进行计数。n-gram模型为了**保持词的顺序**，做了一个**滑窗**的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果，该模型考虑了词的顺序。

例如：

```
// a. 文本内容
John likes to watch movies. Mary likes too.
John also likes to watch football games.

// b. 基于上述两个文档中出现的单词, 构造一个词典（词汇表）
{"John likes”: 1, "likes to”: 2, "to watch”: 3, "watch movies”: 4, "Mary likes”: 5, "likes too”: 6, "John also”: 7, "also likes”: 8, “watch football”: 9, "football games": 10}

// c. 上面的词典中包含10个单词, 每个单词有唯一的索引, 那么每个文本可以使用一个10维的向量来表示
[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
[0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
```

**缺点：**随着n的大小增加，词表会成指数型膨胀，会越来越大。

​    

## 5 离散表示存在的问题

由于存在以下的问题，对于一般的NLP问题，是可以使用离散表示文本信息来解决问题的，但对于要求精度较高的场景就不适合了。

> 无法衡量词向量之间的关系；
>
> 词表的维度随着语料库的增长而膨胀；
>
> n-gram词序列随语料库增长呈指数型膨胀，更加快；
>
> 离散数据来表示文本会带来数据稀疏问题，导致丢失了信息，与生活中理解的信息是不一样的；

​     

# 二 分布式表示

科学家们为了提高模型的精度，又发明出了分布式的表示文本信息的方法，即**用一个词附近的其它词来表示该词，这是现代统计自然语言处理中最有创见的想法之一**。当初科学家发明这种方法是基于人的语言表达，认为一个词是由这个词的周边词汇一起来构成精确的语义信息。就好比，物以类聚人以群分，如果你想了解一个人，可以通过他周围的人进行了解，因为周围人都有一些共同点才能聚集起来。

​       

## 1 GloVe

GloVe（Globel Vectors）算法，其实就是SVD分解与Word2Vec的结合。在介绍GloVe的思想之前，我们先定义一个共现矩阵𝑋，该矩阵中的 𝑋(𝑖, 𝑗) 表示**第 𝑗 个单词出现在以第 𝑖 个单词为中心，长度为𝑛的窗口中的次数**，而后将长度为n的窗口遍历整个语料库，则得到了共现矩阵𝑋。

例如：

```python
// a. 文本内容
I like deep learning.	
I like NLP.	
I enjoy flying

// b. 基于以上三句话，设置滑窗为2，可以得到一个词典:
{"I like", "like deep", "deep learning", "like NLP", "I enjoy", "enjoy flying", "I like"}

// c. 可以得到一个如下图的共现矩阵(对称矩阵)，中间的每个格子表示的是行和列组成的词组在词典中共同出现的次数，也就体现了共现的特性
```

<div align="center"><img src="imgs/nlp-corpus.png" style="zoom:80%;" /></div>

​       

对于大型语料库，可以认为统计的 **词共现矩阵X可以很好的描述词与词之间的相关性**，但是这样大的一个共现矩阵在实际使用中将会面临复杂的 **维度灾难问题**，因此需要想办法词向量进行降维，比如奇异值分解SVD对词-文档共现矩阵进行降维就是这样一种思想。

而对于word2vec，每次训练词向量，都是针对于局部语料进行预测（根据局部上下文预测中心词，或根据中心词预测局部上下文），这就使得模型的训练过程中是很难考虑到整个语料库的全局信息的。因此，通常将其称为一种预测模型（predictive model），其目标是不断提高对其他词的预测能力，即减小预测损失，从而得到词向量。

那有没有一种方法，既能通过**训练的方式得到固定维度的词向量表征**，又能够使得到的**词向量能够充分考虑到语料库的全局特征**？因此提出了GloVe模型

简单来说，Glove相对于Word2Vec，需要提前统计**词共现矩阵**，并将其**整合到代价函数**之中，使得训练结果对于该统计是有一定的重建能力的。便将其称为一种统计模型（count-based model），其目标是**优化减小重建损失**（reconstruction loss），即**降维之后的向量能尽量表达原始向量的完整信息**。

​       

但**存在的问题：**

> 1）向量维数随着词典大小线性增长；
>
> 2）存储整个词典的空间消耗非常大；
>
> 3）一些模型如文本分类模型会面临稀疏性问题；
>
> 4）模型会欠稳定，每新增一份语料进来，稳定性就会变化；

​      

# 三 神经网络表示

## 1 NNLM

NNLM (Neural Network Language model)，神经网络语言模型是03年提出来的，通过训练得到中间产物--词向量矩阵，这就是得到的文本表示向量矩阵。

NNLM说的是定义一个前向窗口大小，其实和上面提到的窗口是一个意思。把这个窗口中最后一个词当做y，把之前的词当做输入x，通俗来说就是预测这个窗口中最后一个词出现概率的模型。

<div align="center"><img src="imgs/nlp-nnlm.png" style="zoom:80%;" /></div>

​      

## 2 Word2Vec

谷歌2013年提出的Word2Vec是目前最常用的词嵌入模型之一，除了作为词嵌入的方法之外，它的一些概念已经被证明可以有效地创建推荐引擎和理解时序数据。在商业的、非语言的任务中，像Airbnb、阿里巴巴、Spotify这样的公司都从NLP领域中提取灵感并用于产品中，从而为新型推荐引擎提供支持。

<div align="center"><img src="imgs/nlp-word2vec-case.png" style="zoom:80%;" /></div>

其主要思想是 “**一个词的意义，应该由其周围经常出现的词来表达**” ，Word2Vec实际是一种浅层的神经网络模型，它有两种网络结构，分别是 连续词袋 **CBOW**（Continues Bag of Words）和 **Skip-gram**。Word2Vec和上面的NNLM很类似，但比NNLM简单。

<div align="center"><img src="imgs/nlp-cbow-skip-gram.png" style="zoom:80%;" /></div>

​     

### 2.1 连续词袋模型 CBOW

**通过上下文来预测当前值**，即获得中间词两边的的上下文，然后用周围的词去预测中间的词，把中间词当做y，把窗口中的其它词当做x输入，x输入是经过one-hot编码过的，然后通过一个隐层进行求和操作，最后通过激活函数softmax，可以计算出每个单词的生成概率，接下来的任务就是训练神经网络的权重，使得语料库中所有单词的整体生成概率最大化，而求得的权重矩阵就是文本表示词向量的结果。总的来说，**相当于一句话中扣掉一个词，让你猜这个词是什么**。

<div align="center"><img src="imgs/nlp-cbow.png" style="zoom:80%;" /></div>

CBOW是一个三层神经网络（如下图）

> 1. 输入层：上下文单词的one-hot {假设单词向量空间dim为V，上下文单词个数为C}
> 2. 所有one-hot分别乘以共享的输入权重矩阵W. {VxN矩阵，N为自己设定的数，初始化权重矩阵W}
> 3. 所得的向量 {因为是one-hot所以为向量} 相加求平均作为隐层向量, size为1N
> 4. 乘以输出权重矩阵W' {NV}
> 5. 得到向量 {1V} 激活函数处理得到V-dim概率分布 {PS: 因为是one-hot，其中的每一维斗代表着一个单词}
> 6. 概率最大的index所指示的单词为预测出的中间词（target word）与true label的one-hot做比较，误差越小越好（根据误差更新权重矩阵）

<div align="center"><img src="imgs/nlp-cbow-arc.png" style="zoom:80%;" /></div>

所以，需要定义loss function（一般为交叉熵代价函数），采用梯度下降算法更新W和W'。训练完毕后，输入层的每个单词与矩阵W相乘得到的向量的就是想要的词向量（word embedding），这个矩阵（所有单词的word embedding）也叫做look up table（矩阵W自身），也就是说，任何一个单词的one-hot乘以这个矩阵都将得到自己的词向量，即有了look up table就可以免去训练过程直接查表得到单词的词向量了。

​     

### 2.2 跳字模型 Skip-gram

**用中心词来预测上下文**，即通过当前词来预测窗口中上下文词出现的概率模型，把当前词当做x，把窗口中其它词当做y，依然是通过一个隐层接一个Softmax激活函数来预测其它词的概率。总的来说，**相当于给你一个词，让你猜前面和后面可能出现什么词**。

<div align="center"><img src="imgs/nlp-skip-gram.png" style="zoom:80%;" /></div>

**Skip-gram**的网络结构共包含三层：输入层，隐藏层和输出层，处理步骤：

> 1. 输入层接收shape为 [1,𝑉] 的one-hot向量𝑥，其中 𝑉 代表词表中单词的数量，这个one-hot向量就是上边提到的中心词；
> 2. 隐藏层包含一个shape为 [𝑉,𝑁] 的参数矩阵𝑊1，其中这个 𝑁 代表词向量的维度，𝑊1 就是word embedding 矩阵，即要学习的词向量。将输入的one-hot向量 𝑥 与𝑊1 相乘，便可得到一个shape为[1,𝑁] 的向量，即该输入单词对应的词向量𝑒；
> 3. 输出层包含一个shape为 [𝑁,𝑉] 的参数矩阵𝑊2，将隐藏层输出的𝑒与 𝑊2 相乘，便可以得到shape为[1,𝑉]的向量𝑟，内部的数值分别代表每个候选词的打分，使用softmax函数，对这些打分进行归一化，即得到中心词的预测各个单词的概率；

<div align="center"><img src="imgs/nlp-skip-gram-arc.png" style="zoom:80%;" /></div>

​     

### 2.3 优化

Word2Vec模型是一个超级大的神经网络（权重矩阵规模非常大）。举个例子，我们拥有10000个单词的词汇表，如果想嵌入300维的词向量，那么 输入-隐层权重矩阵 和 隐层-输出层 的权重矩阵都会有 10000 x 300 = 300万个权重，在如此庞大的神经网络中进行梯度下降是相当慢的。更糟糕的是，需要大量的训练数据来调整这些权重并且避免过拟合。百万数量级的权重矩阵和亿万数量级的训练样本意味着训练这个模型将会是个灾难（太凶残了）。

下面主要介绍两种方法优化训练过程：

1）**层次Softmax**

作为一种计算高效的近似方法，Hierarchical Softmax被广泛使用，主要为了避免要计算所有词的softmax概率，该方法不用为了获得概率分布而评估神经网络中的W个输出结点，而只需要评估大约log2(W)个结点。层次Softmax使用一种二叉树结构来表示词典里的所有词，V个词都是二叉树的叶子结点，而这棵树一共有V−1个非叶子结点。

> **哈夫曼树(Huffman Tree)**：给定N个权值作为N个[叶子结点](https://baike.baidu.com/item/叶子结点/3620239)，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。

<div align="center"><img src="imgs/nlp-softmax.png" style="zoom:80%;" /></div>

​      

2）**负例采样**（Negative Sampling）

是用来提高训练速度并且改善所得到词向量的质量的一种方法，且不同于原本每个训练样本更新所有的权重，负采样每次让一个训练样本仅仅更新一小部分的权重，这样就会降低梯度下降过程中的计算量。

<div align="center"><img src="imgs/nlp-negative-sample.png" style="zoom:80%;" /></div>

以 “**Pineapples are spiked and yellow**” 为例进行讲解，如上图所示，其中中心词是spiked和上下文词是正样本Pineapples are and yellow，这里这个正样本代表该词是中心词的上下文。

以正样本单词Pineapples为例，**之前的做法**是在使用softmax学习时，需要最大化Pineapples的推理概率，同时最小化其他词表中词的推理概率。之所以计算缓慢，是因为需要对词表中的所有词都计算一遍。然而我们还可以使用另一种方法，就是随机从词表中选择几个代表词，通过最小化这几个代表词的概率，去**近似**最小化整体的预测概率。

例如，先指定一个中心词（spiked）和一个目标词正样本（Pineapples），再随机在词表中采样几个目标词负样本（如”dog，house”等）。

有了这些正负样本，skip-gram模型就变成了一个二分类任务。对于目标词正样本，需要最大化它的预测概率；对于目标词负样本，需要最小化它的预测概率。通过这种方式，就可以完成计算加速，这个做法就是**负采样**。

负采样整体训练流程：

> 1）获取中心词spiked的正负样本（正负样本是目标词），这里一般会设定个固定的窗口，比如中心词前后3个词算是中心词的上下文（即正样本）；
>
> 2）获取对应词的词向量，其中中心词从黄色的向量矩阵中获取词向量，目标词从灰色的向量矩阵中获取词向量；
>
> 3）将中心词和目标词的词向量进行点积并经过sigmoid函数，我们知道sigmoid是可以用于2分类的函数，通过这种方式来预测中心词和目标词是否具有上下文关系；
>
> 4）将预测的结果和标签使用交叉熵计算损失值，并计算梯度进行反向迭代，优化参数；

经过这个训练的方式，就可以训练出我们想要的词向量，但**上图**中包含两个词向量矩阵（黄色的和灰色的），一般是将中心词对应的词向量矩阵（黄色的）作为正式训练出的词向量。

​       

**Word2Vec存在的问题**

> 1）对每个local context window单独训练，没有利用包 含在global co-currence矩阵中的统计信息；
>
> 2）对多义词无法很好的表示和处理，因为使用了唯一的词向量；
>
> 3）Word2vec 是一种静态的方式，虽然通用性强，但是无法针对特定任务做动态优化；

​        

### 2.4 多样的2vec向量化模型

word2vec模型的问题在于词语的多义性，比如duck这个单词常见的含义有水禽或者下蹲，但对于 word2vec 模型来说，它倾向于将所有概念做归一化平滑处理，得到一个最终的表现形式。

比如：doc2vec / tweet2vec / batter-pitcher-2vec / illustration-2vec / lda2vec / sentence2vec / wiki2vec / topicvec / entity2vec / str2vec / node2vec / item2vec / author2vec / playlist-to-vec / sense2vec / med2vec / game2vec / paper2vec

​     

# 四 图解 Word2Vec

下面通过实例简单讲解Word2Vec的一般过程

## 1 词嵌入 Word Embeddings

假设下面是一个单词“king”的词嵌入Embeddings（在维基百科上训练的GloVe向量）

> [ 0.50451 , 0.68607 , -0.59517 , -0.022801, 0.60046 , -0.13498 , -0.08813 , 0.47377 , -0.61798 , -0.31012 , -0.076666, 1.493 , -0.034189, -0.98173 , 0.68229 , 0.81722 , -0.51874 , -0.31503 , -0.55809 , 0.66421 , 0.1961 , -0.13495 , -0.11476 , -0.30344 , 0.41177 , -2.223 , -1.0756 , -1.0783 , -0.34354 , 0.33505 , 1.9927 , -0.04234 , -0.64319 , 0.71125 , 0.49159 , 0.16754 , 0.34344 , -0.25663 , -0.8523 , 0.1661 , 0.40102 , 1.1685 , -1.0137 , -0.21585 , -0.15155 , 0.78321 , -0.91241 , -1.6106 , -0.64426 , -0.51042 ]

这是一个包含50个数字的列表，通过观察数值看不出什么，但是可以给它稍微可视化，根据它们的值对单元格进行**颜色编码**（如果接近2则为红色，接近0则为白色，接近-2则为蓝色），以便比较其它词向量。

<div align="center"><img src="imgs/nlp-word2vec-color-1.png" style="zoom:80%;" /></div>

如果忽略数字并仅查看颜色以指示单元格的值，那么将“king”与其它单词进行比较：

<div align="center"><img src="imgs/nlp-word2vec-color-2.png" style="zoom:80%;" /></div>

可以看出 “Man” 和 “Woman” 彼此之间的相似度比它们中的任何一个与 “King” 的相似度如何？上面向量图示很好的展现了这些单词的信息/含义/关联。

另一个示例列表（通过垂直扫描列来查找具有相似颜色的列）：

<div align="center"><img src="imgs/nlp-word2vec-color-other.png" style="zoom:80%;" /></div>

有几个要点需要指出：

> 1. 所有这些不同的单词都有一条直的红色列，它们在这个维度上是相似的（虽然不知道每个维度是什么）；
> 2. 可以看到 “woman” 和 “girl” 在很多地方是相似的，“man” 和 “boy” 也是一样；
> 3. “boy” 和 “girl” 也有彼此相似的地方，但这些地方却与“woman”或“man”不同，这些是否可以总结出一个模糊的“youth”概念？或许是；
> 4. 除了最后一个单词，所有单词都是代表人，通过添加了一个对象 “water” 来显示类别之间的差异，可以看到蓝色列一直向下并在 “water” 的词嵌入之前停下了；
> 5. “king”和“queen”彼此之间相似，但它们与其它单词都不同，这些是否可以总结出一个模糊的 “royalty” （王权）概念？

​     

## 2 类比 Analogies

> *"Words can carry any burden we wish. All that's required is agreement and a tradition upon which to build."* 

展现嵌入奇妙属性的著名例子是类比，可以添加、减去词嵌入并得到有趣的结果。一个著名例子是公式：**“king”-“man”+“woman”**：

<div align="center"><img src="imgs/nlp-word2vec-analogies-case.png" style="zoom:80%;" /></div>

上图显示了与结果向量最相似的单词列表，每个单词都具有余弦相似性，而后像之前一样对这个类比可视化：

<div align="center"><img src="imgs/nlp-word2vec-analogies-case-color.png" style="zoom:80%;" /></div>

由图可知，由 “king-man + woman” 生成的向量并不完全等同于“queen”，但 “queen” 是语料集合中包含的400,000个字嵌入中最接近它的单词。现在已经了解了训练好的词嵌入，接下来了解一下训练过程，但在开始使用word2vec之前，先来看一下词嵌入的父概念 -- **神经语言模型**。

​     

## 3 语言模型 Language Modeling

自然语言处理最典型例子：智能手机输入法中的**下一单词预测功能**，这是个被数十亿人每天使用上百次的功能。

<div align="center"><img src="imgs/nlp-word2vec-lm-case.png" style="zoom:80%;" /></div>

下一单词预测是一个可以通过语言模型实现的任务，模型会通过单词列表（比如说两个词）去尝试预测可能紧随其后的单词，如上图截屏中，可以认为模型接收到两个绿色单词（thou shalt），并推荐了一组单词（“not” 就是其中最有可能被选用的一个）：

<div align="center"><img src="imgs/nlp-word2vec-lm-case-pre.png" style="zoom:80%;" /></div>

可以把这个模型想象以下黑盒:

<div align="center"><img src="imgs/nlp-word2vec-lm-case-model.png" style="zoom:80%;" /></div>

但事实上，模型不会只输出一个单词，而是计算出模型词库中所有可能输出单词的概率分数（可能有几千到几百万个单词），而后选出分数最高的词推荐给用户。其次，NLP模型的输出其实是模型所知单词的概率评分，通常以百分比形式表示其输出概率，但在模型输出向量组中，实际上40%表示为0.4。

<div align="center"><img src="imgs/nlp-word2vec-lm-case-model-all.png" style="zoom:80%;" /></div>

同时，早期自然语言模型（如：Bengio 2003），经过训练后会按如下三步骤进行预测：

<div align="center"><img src="imgs/nlp-word2vec-lm-early-steps.png" style="zoom:80%;" /></div>

第一步就是Embedding，模型在经过训练之后会生成一个**映射单词表所有单词的矩阵**，因此在进行预测的时候，模型算法就是在这个映射矩阵中查询输入的单词，然后计算出预测值：

<div align="center"><img src="imgs/nlp-word2vec-lm-early-embedding.png" style="zoom:80%;" /></div>

现在让我们把重点放到模型训练上，来学习一下**如何构建这个映射矩阵**。

​      

## 4 语言模型训练 LM Training

> *A process cannot be understood by stopping it. Understanding must move with the flow of the process, must join it and flow with it.*

相较于大多数其他机器学习模型，语言模型有一个很大有优势，那就是有丰富的文本内容来训练语言模型，包括书籍、文章、维基百科、及各种类型的文本内容，而相比之下，许多其他机器学习的模型开发需要手工设计数据或者专门采集数据。

通常情况下，如果找出出现在每个单词附近的词，那就能获得样本词间的映射关系。大致流程如下：

> 1. 先是获取大量文本数据（例如：维基百科内容）；
> 2. 然后建立一个可以沿文本滑动的窗（例如：一个窗里包含三个单词）；
> 3. 利用这样的滑动窗就能为训练模型生成大量样本数据；

<div align="center"><img src="imgs/nlp-word2vec-lm-text-data.png" style="zoom:80%;" /></div>

当这个窗口沿着文本滑动时，就可生成一套用于模型训练的数据集，为了明确这个过程，来看下滑动窗是如何处理这个短语的：

> 在一开始的时候，窗口锁定在句子的前三个单词上，并把前两个单词单做**特征**，第三个单词单做**标签**label，从此生成了数据集中的第一个样本，用在后续的语言模型训练中。
>
> <div align="center"><img src="imgs/nlp-word2vec-lm-window-three-words.png" style="zoom:80%;" /></div>
>
> 接着将窗口滑动到下一个位置，并生产第二个样本:
>
> <div align="center"><img src="imgs/nlp-word2vec-lm-window-three-words-2.png" style="zoom:80%;" /></div>
>
> 重复以上操作，就能得到一个**较大的数据集**，从数据集中可看到在不同的单词组后面会出现的单词：
>
> <div align="center"><img src="imgs/nlp-word2vec-lm-window-three-words-other.png" style="zoom:80%;" /></div>
>
> 在实际应用中，模型往往在滑动窗口时同时被训练，但将**生成数据集**和**训练模型**分为两个阶段会显得更清晰易懂一些。当然除了使用神经网络建模之外，还常用N-gams的技术进行模型训练。

​      

## 5 顾及两头 Look both ways

根据前面的信息进行填空：

<div align="center"><img src="imgs/nlp-word2vec-lm-fill-blank.png" style="zoom:80%;" /></div>

如果在空白处提供了5个可选单词（如果事先提及到‘bus’），可以肯定大多数人都会把bus填入空白中，但是如果需要填充空白后的一个单词，那答案会有变吗？

<div align="center"><img src="imgs/nlp-word2vec-lm-fill-blank-2.png" style="zoom:80%;" /></div>

这下空白处改填的内容完全变了，这时’red’这个词最有可能适合这个位置。从这个例子中，可以知道一个单词的前后词语都带信息价值，同时也证实了在解答问题前，需要考虑两个方向的单词（目标单词的左侧单词与右侧单词）。

**那该如何调整训练方式以满足这个要求呢** ？

​     

## 6 跳字模型 Skipgram

> Intelligence takes chance with limited data in an arena where mistakes are not only possible but also necessary.

训练过程不仅要考虑目标单词的前两个单词，还要考虑其后两个单词

<div align="center"><img src="imgs/nlp-word2vec-lm-cbow-case.png" style="zoom:80%;" /></div>

如果这么做，实际上构建并训练的模型（CBOW，连续词袋模型）就如下所示：

<div align="center"><img src="imgs/nlp-word2vec-lm-cbow-case-model.png" style="zoom:80%;" /></div>

然后还有另一种架构，不是根据前后文（前后单词）来猜测目标单词，而是**推测当前单词可能的前后单词**，设想一下滑动窗在训练数据时如下图所示（绿框中的词语是输入词，粉框则是可能的输出结果）：

<div align="center"><img src="imgs/nlp-word2vec-lm-skipgram-pre.png" style="zoom:80%;" /></div>

这里粉框颜色深度呈现不同，是因为滑动窗给训练集产生了4个独立的样本：

<div align="center"><img src="imgs/nlp-word2vec-lm-skipgram-samples.png" style="zoom:80%;" /></div>

上述方式称为 **Skipgram架构**，而模型滑动窗训练数据集过程如下：

> 这样就为数据集提供了4个样本数据：
>
> <div align="center"><img src="imgs/nlp-word2vec-lm-skipgram-sam-step1.png" style="zoom:80%;" /></div>
>
> 然后移动滑动窗到下一个位置，并生成了新的4个样本数据：
>
> <div align="center"><img src="imgs/nlp-word2vec-lm-skipgram-sam-step2.png" style="zoom:80%;" /></div>
>
> 重复移动几组位置之后，就能得到一批样本集：
>
> <div align="center"><img src="imgs/nlp-word2vec-lm-skipgram-sam-other.png" style="zoom:80%;" /></div>

​       

## 7 重新审视训练过程

现在已经从现有的文本中获得了Skipgram模型的训练数据集，那接下来看看如何使用它来训练一个能**预测相邻词汇**的自然语言模型。

<div align="center"><img src="imgs/nlp-word2vec-lm-skipgram-map.png" style="zoom:80%;" /></div>

即从数据集中的第一个样本开始，将特征输入到**未训练的模型**，让它预测一个可能的相邻单词

<div align="center"><img src="imgs/nlp-word2vec-lm-model-predict.png" style="zoom:80%;" /></div>

模型经过三个步骤执行，后计算输出得到一个预测向量（对应单词表中每个单词的概率），但由于模型未经训练，所以该阶段预测结果肯定是错误的，但基于训练集数据中的结果标签，实际上训练前我们已知任务最后会输出哪个词（PS：“目标向量” 中目标单词概率为1，其他所有单词概率为0）。

<div align="center"><img src="imgs/nlp-word2vec-lm-target&prefict.png" style="zoom:80%;" /></div>

从任务结果预测向量和实际样本标签数据，可知 **模型偏差向量=实际样本向量 - 模型预测向量** ：

<div align="center"><img src="imgs/nlp-word2vec-lm-vector-off.png" style="zoom:80%;" /></div>

而后使用误差向量Error用于更新任务模型了，即在下一轮预测中，如果用 "not" 作为输入，那么更有可能得到 "thou" 作为输出了，而这其实就是训练的第一步。

<div align="center"><img src="imgs/nlp-word2vec-lm-vector-off-update.png" style="zoom:80%;" /></div>

最后继续对数据集内下一份样本进行同样的操作，直到遍历所有的样本为止，这一过程称之为 **一轮（epoch）**。如果再多做几轮（epoch），就可得到训练过的模型，最后将模型输出的Embedding矩阵用于其它训练任务。虽然以上确实有助于理解整个流程，但这依然不是word2vec真正训练的方法。

​       

## 8 负例采样 Negative Sampling

回想一下这个神经语言模型计算预测值的三个步骤： 

<div align="center"><img src="imgs/nlp-word2vec-lm-pre-model.png" style="zoom:80%;" /></div>

但从计算的角度来看，第三步非常昂贵，尤其是需要在数据集中为每个训练样本都做一遍（很容易就多达数千万次），因此需要寻找提高性能的方法。

常见的一种方法是将目标分为两个步骤：

> 1. 生成**高质量的词embedding**（不要担心下一个单词预测）；
> 2. 使用这些**高质量的embedding来训练语言模型**（进行下一个单词预测）；

如果使用高性能模型生成高质量embedding，那么可以稍微调整一下预测相邻单词任务：

<div align="center"><img src="imgs/nlp-word2vec-lm-pre-next.png" style="zoom:80%;" /></div>

即将其切换到一个提取输入与输出单词的模型，并输出一个表明它们是否是邻居的分数（0表示“不是邻居”，1表示“邻居”）。

<div align="center"><img src="imgs/nlp-word2vec-lm-extract-model.png" style="zoom:80%;" /></div>

这个转换则是将任务模型从 **神经网络 **改为 **逻辑回归模型**，使得任务更简单且计算速度更快，但需要更新数据集的结构，新增一个值为0或1的标签列，以满足任务模型执行需要（PS：所有添加的单词都是邻居，初始化为1）。

<div align="center"><img src="imgs/nlp-word2vec-lm-data-format.png" style="zoom:80%;" /></div>

虽然模型计算速度有提升，几分钟内能处理数百万个例子，但仍需要解决一个**漏洞**：即如果所有例子都是邻居（target=1），那模型可能会被训练得永远返回1，准确性百分百，导致模型未学到任何内容，仅生成了垃圾embeddings结果（如下图）。

<div align="center"><img src="imgs/nlp-word2vec-lm-problem.png" style="zoom:80%;" /></div>

所以为了解决此问题，需要在模型数据集中引入**负样本** （不是邻居的单词样本），即对数据集中的每个样本词，都添加了负面样例（target=0）。

<div align="center"><img src="imgs/nlp-word2vec-lm-negative-sample.png" style="zoom:80%;" /></div>

而后，参考**噪声对比评估**的思想，随机从词汇表中 **抽取单词** 作为embeddings结果输出，

> [噪声对比评估](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)：将实际信号（相邻单词的正例）与噪声（随机选择的不是邻居的单词）进行对比

<div align="center"><img src="imgs/nlp-word2vec-lm-random-sample.png" style="zoom:80%;" /></div>

​     

## 9 Word2vec训练流程

现在已经了解了word2vec中的两个核心思想：**负例采样** 和 **skipgram**，接下来看一下word2vec模型的实际训练过程：

<div align="center"><img src="imgs/nlp-word2vec-lm-two-model.png" style="zoom:80%;" /></div>

1）在训练前，需要预先处理正在训练模型的文本，并在这一步中确定一下词典的大小（称之为vocab_size，比如说10,000），以及哪些词被它包含在内。

2）在训练开始时，创建两个包含**词汇表中每个单词的embedding**的矩阵（Embedding和Context矩阵），大小为 vocab_size * embedding_size。

<div align="center"><img src="imgs/nlp-word2vec-lm-two-matrixs.png" style="zoom:80%;" /></div>

其次，用**随机值**初始化这些矩阵，而后开始训练，并且在每一训练步骤中，使用一个正例（positive example）和相关的负例样本（negative examples）进行训练（如下第一组训练数据）。

<div align="center"><img src="imgs/nlp-word2vec-lm-dataset.png" style="zoom:80%;" /></div>

3）确定随机样本词的embeddings

* **输入单词**： "not" ，从Embedding矩阵中查找；
* **输出/上下文单词**："thou"（实际邻居词），"aaron" 和 "taco"（负面例子），从Context矩阵中查找；

PS：词汇表中每个单词的Embedding均包含在两个矩阵中

<div align="center"><img src="imgs/nlp-word2vec-lm-lookup.png" style="zoom:80%;" /></div>

4）然后计算输入样本 embedding 与每个上下文样本embedding的点积，得到结果表示的是  **输入和上下文embedding的相似性 **值。

<div align="center"><img src="imgs/nlp-word2vec-lm-embedding-dot.png" style="zoom:80%;" /></div>

5）而后使用逻辑函数 **sigmoid** 将上述概率值进行归一化处理，得到的结果作为**模型输出**，可以看到无论是sigmoid操作之前还是之后，taco得分最高，aaron最低。

<div align="center"><img src="imgs/nlp-word2vec-lm-embedding-dot-softmax.png" style="zoom:80%;" /></div>

6）而后，计算模型 **预测误差 = target - sigmoid_scrores**

<div align="center"><img src="imgs/nlp-word2vec-lm-embedding-error.png" style="zoom:80%;" /></div>

7）最后，模型利用这个误差分数来调整更新not、thou、aaron和taco这4个样本词embeddings，使模型在下一次计算时，得到结果会更接近目标分数。

<div align="center"><img src="imgs/nlp-word2vec-lm-embedding-error-update.png" style="zoom:80%;" /></div>

8）至此第一个样本训练步骤结束，而后使用更新后的embeddings（not，thou，aaron和taco）继续下一个样本集的训练（样本训练步骤一致）。

<div align="center"><img src="imgs/nlp-word2vec-lm-dataset-update.png" style="zoom:80%;" /></div>

9）经过多次循环遍历完整个数据集后，embeddings不断更新，最后停止训练过程，丢弃Context矩阵，并使用 Embeddings 矩阵作为模型下一个任务预训练的embeddings。

​        

## 10 窗口大小和负样本数量

word2vec训练过程中的两个关键超参数是 **滑动窗口大小** 和 **负样本数量**

<div align="center"><img src="imgs/nlp-word2vec-lm-training-win-size.png" style="zoom:80%;" /></div>

但不同任务适合不同的窗口大小，通常情况下会存在以下情况：

> 较小的窗口大小（2-15）：两个样本embeddings高相似性分数表示这些词可以互换（PS：反义词在只查看它们周围单词时经常可以互换，例如好的和坏的经常出现在类似的语境中）
>
> 较大的窗口大小（15-50，甚至更多）：样本embeddings相似性更能反映单词的相关性；

其次，负样本数量也是训练训练过程的另一个重要参数，原论文认为5-20个负样本是比较理想的数量，且还指出，当拥有足够大的数据集时，2-5个似乎就已经足够了（Gensim默认为5个负样本）。

<div align="center"><img src="imgs/nlp-word2vec-lm-training-negative-size.png" style="zoom:80%;" /></div>

​         

​    

# 附录



1. https://github.com/NLP-LOVE/ML-NLP
2. [自然语言处理 03：N-gram 语言模型](https://yey.world/2020/03/09/COMP90042-03/)
3. [不懂n-gram，怎么学好语言模型？](https://www.cnblogs.com/hithink/p/12107622.html)
4. [NLP教程 | 斯坦福CS224n · 课程带学与全套笔记解读](https://www.showmeai.tech/tutorials/36)
5. [Word2Vec: 一种词向量的训练方法](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/word_representation/word2vec.html)
6. [NLP+2vec︱认识多种多样的2vec向量化模型](https://cloud.tencent.com/developer/article/1020400)
7. [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
8. [图解Word2vec，读这一篇就够了](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669277&idx=2&sn=bc8f0590f9e340c1f1359982726c5a30&chksm=bd4c648e8a3bed9817f30c5a512e79fe0cc6fbc58544f97c857c30b120e76508fef37cae49bc&scene=0&xtrack=1#rd)
9. [词向量(one-hot/SVD/NNLM/Word2Vec/GloVe)](https://www.cnblogs.com/sandwichnlp/p/11596848.html)

