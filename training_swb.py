import os
import pandas as pd
from collections import Counter


########################### Prepare Training Data ###############################
# Preparing training data
data_train = pd.read_csv("swb/swb-train.csv")
print("Shape of training data = ", data_train.shape)

# Adding white space separated full stop to each sentence in data. There are 1.24K sentences in train.csv here.
data_train['transcript'] = data_train['transcript'] + " ."

# As the training requires multiple files with one text sentence per line, 
# we will create 20K training files by writing 6 sentences per file. 
# After running the below python snippet, we get 20K files in train directory.
if not os.path.exists("swb/train"):
    os.makedirs("swb/train")
 
for i in range(0,data_train.shape[0],6):
    text = "\n".join(data_train['transcript'][i:i+6].tolist())
    fp = open("swb/train/"+str(i)+".txt","w")
    fp.write(text)
    fp.close()

########################### Prepare Validation Data ###############################
# Validation data is also prepared in the similar manner as training data.
data_dev = pd.read_csv("swb/swb-dev.csv")
data_dev['transcript'] = data_dev['transcript'] + " ."
if not os.path.exists("swb/dev"):
    os.makedirs("swb/dev")
 
for i in range(0,data_dev.shape[0],6):
    text = "\n".join(data_dev['transcript'][i:i+6].tolist())
    fp = open("swb/dev/"+str(i)+".txt","w")
    fp.write(text)
    fp.close()   

########################### Preparing Vocabulary File ###############################
# The vocabulary file is a a text file with one token per line. 
# It must also include the special tokens <S>, </S> and <UNK> (case sensitive) in the file. 
# The vocabulary file should be sorted in descending order by token count in your training data. 
# The first three lines should be the special tokens (<S>, </S> and <UNK>), 
# then the most common token in the training data, ending with the least common token.    

texts = " ".join(data_train['transcript'].tolist())
words = texts.split(" ")
print("Number of tokens in Training data = ",len(words))
dictionary = Counter(words)
print("Size of Vocab",len(dictionary))
sorted_vocab = ["&lt;S&gt;","&lt;/S&gt;","&lt;UNK&gt;"]
sorted_vocab.extend([pair[0] for pair in dictionary.most_common()])
 
text = "\n".join(sorted_vocab)
fp = open("swb/vocab.txt","w")
fp.write(text)
fp.close()     