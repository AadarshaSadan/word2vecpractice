
# coding: utf-8

# In[1]:


from __future__ import absolute_import,division,print_function
#for word encoding
import codecs
#regex
import glob
#concurrency
import multiprocessing
#dealin with operating system,like read file
import os
#import pprint
#regular expression
import re
#naturaal lanuage toolkit
import nltk
#word 2 vec
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
import pandas as pd
#visualization 
import seaborn as sns


# In[2]:


#process our data
#clan
nltk.download('punkt')#pretrained tokenizer
nltk.download('stopwords')#word like and,the,as ,an


# In[3]:


#get the book names mathing text file name
book_filenames=sorted(glob.glob("data/*.txt"))
if book_filenames:
    print("found books")


# Combine books into one string

# In[4]:


corpus_raw=u""
for book_filename in book_filenames:
    print("Readinfile'{0}'...".format(book_filename))
    with codecs.open(book_filename,"r",  "utf-8") as book_file:
        corpus_raw+=book_file.read()
    print("corpus is now {0} char long".format(len(corpus_raw)))
    print()


# # split corpus into sentences

# In[5]:


tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')


# In[6]:


#into sentences
raw_sentences=tokenizer.tokenize(corpus_raw)


# In[7]:


#convert sentence into list of words
def sentence_to_wordlist(raw):
    clean=re.sub("[^a-zA-Z]"," ", raw)
    words=clean.split()
    return words


# In[8]:


#senteces where each words is tokinize
sentences=[]
for raw_sentence in raw_sentences:
    if len(raw_sentence)>0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[9]:


print(raw_sentences[6])
print(sentence_to_wordlist(raw_sentences[6]))


# In[10]:


token_count=sum([len(sentence)for sentence in sentences])
print("book contains {0:,}".format(token_count))


# # Train Word2Vec

# In[11]:


#once we have vector our 3 main task taht vector help is
#distance,similarity,ranking
#more dimension more computationally expensive to train
#but also more accurate
#more generalize

#dimensionality of the resultinf word vector
num_features=300

#vector is a type of tensor

#minimum word count threshold
min_word_count=3

#number of threads to run in paralle;
#more worker faster to train
num_workers=multiprocessing.cpu_count()

#context window length
context_size=7

downsampling=1e-3

#seed for the RNG,to make  the result reproductive
#random number generator
#deterministic,good for debugging
seed=1





# In[12]:


thrones2vec=w2v.Word2Vec(
    sentences,
    #sg=1,
    #seed=seed,
    size=num_features,
    window=context_size,
    min_count=min_word_count,
    workers=num_workers,
    
    #sample=downsampling
)


# In[13]:


#thrones2vec.build_vocab(sentences)


# In[14]:


print("wordvec vocabulary length:",len(thrones2vec.wv.vocab))


# # start traning

# In[15]:


thrones2vec.train(sentences,total_examples=len(sentences),epochs=10)


# In[18]:


if not os.path.exists("trained_sad"):
    os.makedirs("trained_sad")


# In[19]:


thrones2vec.save(os.path.join("trained_sad","thrones2vec.w2v"))


# # load trained module

# In[20]:


thrones2vec=w2v.Word2Vec.load(os.path.join("trained_sad","thrones2vec.w2v"))


# # compress the word vectors into 2d space and plot them

# In[21]:


tsne=sklearn.manifold.TSNE(n_components=2,random_state=0)


# In[24]:


all_word_vectors_matrix=thrones2vec.wv.syn0


# In[25]:


all_word_vectors_matrix_2d=tsne.fit_transform(all_word_vectors_matrix)


# In[26]:


points=pd.DataFrame{ 
    [
        (word,coords[0],coords[1])
        for word,coords in[
            (word,all_word_vectors_matrix_2d[thrones2vec.vocab[word].index])
            for word in thrones2vec.vocab
        ]
    ],
    colums=["word","x","y"]
}


# In[ ]:


points.head(10)

