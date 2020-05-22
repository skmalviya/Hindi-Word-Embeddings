## Help Links :

### Convert Genism model .bin --> .txt

```
(pydial_py37)$ python3
>>> from gensim.models import Word2Vec
>>> model = Word2Vec.load("word2vec.model")
>>> model.wv.save_word2vec_format('test_w2v.txt', binary=False)
```

1.Rerun on complete data (21 May 2020): hindi-word-embeddings : https://github.com/manikbhandari/hindi-word-embeddings

```
conda activate pydial_py37
python3 manikbhandari_hindi_w2v.py
Content Used : 100000 sentences from 'monolingual.hi' + final_testing_diag.txt + final_training_diag.txt + final_valid_diag.txt
No of lines : 108998
Total No of words : 2235549
Total No of Unique words : 211169
Training started...
Training finished.
Time taken in seconds =  16.82583498954773
Vector-Dimension = 300
```

1.hindi-word-embeddings : https://github.com/manikbhandari/hindi-word-embeddings

```
conda activate pydial_py37
python3 manikbhandari_hindi_w2v.py
Content Used : 100000 sentences from 'monolingual.hi' + final_testing_diag.txt + final_training_diag.txt + final_valid_diag.txt
No of lines  :  106817
Total No of words : 2209329
Total No of Unique words : 210937
Time taken in seconds =  15.336698293685913
Vector-Dimension = 300
```


2.Word2Vec and FastText Word Embedding with Gensim : https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c
  * sudo -E pip3 install -U gensim (for python3.6)
  * python3 genism_w2v.py
