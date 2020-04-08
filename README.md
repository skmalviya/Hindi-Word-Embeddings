**Task 1**
* I have added gensim model (see directory 'gensim').
* training prgram : manikbhandari_hindi_w2v.py
```
model = Word2Vec(sen, size=300,window=5,min_count=5, negative=20, workers=16)
model.save("word2vec.model")
```
* Using 'word2vec.model', you can generate word-embeddings for any word in our corpus.
* Use this model, apply in NBT program @ https://github.com/skmalviya/DST-NBT-Hindi 

# Hindi-Word-Embeddings
Repo for generating various Word-Embeddings for a Hindi Corpus

# Tutorial
1. Training ELMO from Scratch on Custom Data-set for Generating Embeddings (Tensorflow) : https://appliedmachinelearning.blog/2019/11/30/training-elmo-from-scratch-on-custom-data-set-for-generating-embeddings-tensorflow/
2. Deep Dive The ELMo Implementation : https://huntzhan.github.io/posts/nlp/elmo/
3.How to build custom NER model with Context based Word Embeddings in Vernacular languages: https://medium.com/saarthi-ai/how-to-make-your-own-ner-model-with-contexual-word-embeddings-5086276e04a0

**Task 0**
* Follow the above tutorial.
* The program 'training_swb.py' generates training, testing data from swb.zip wich contains transcript of switchboard corpus in order to evaluate a WE model.
* Write program to build W2V (CBOW, SG), GLOVE, FASTTEST, BLIM-TF, FLAIR and compare them on swb.zip corpus before training our own corpus.
