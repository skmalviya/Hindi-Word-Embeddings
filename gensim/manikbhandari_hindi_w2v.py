import os
import time
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim import corpora
from gensim.models import Word2Vec

def train_w2v():
	head = []
	with open('monolingual.hi') as f:
		head = [next(f) for x in range(100000)]
	    #for line in f:
	    #	head.append(line)
	    #	l += 1
	## Writing 100000 sentences to monolingual.small.hi file
	fw = open('monolingual.small.hi','w')
	fw.write(''.join(head))    
	
	with open('final_testing_diag.txt') as f:
		for line in f:
			head.append(line)

	with open('final_training_diag.txt') as f:
		for line in f:
			head.append(line)

	with open('final_valid_diag.txt') as f:
		for line in f:
			head.append(line)
	print (f"No of lines : {len(head)}")

	sen = []
	for line in head:
	    words = line.split()
	    sen.append(words)

	allwords = []
	for l in sen:
	    allwords += l    

	print (f"Total No of words : {len(allwords)}")

	print (f"Total No of Unique words : {len(set(allwords))}")

	print ("Training started...")
	start = time.time()
	path = get_tmpfile("word2vec.model")
	model = Word2Vec(sen, size=300,window=5,min_count=5, negative=20, workers=16)
	model.save("word2vec.model")
	print ("Training finished.")
	end = time.time()
	print("Time taken in seconds = ", end - start)
	print (len(model.wv[sen[0][0]]))

if __name__ == "__main__":
	if not os.path.isfile('word2vec.model'):
		train_w2v()
		print ("Model is trained.")
	else:
		print ("Using the already trained Model.")
	model = Word2Vec.load("word2vec.model")
	## Model can be trained again.
	#print (model.wv["नमस्ते"])
	#model.train([["नमस्ते", "दुनिया"]], total_examples=1, epochs=1)
	#print (model.wv["दुनिया"], len(model.wv["दुनिया"]))

