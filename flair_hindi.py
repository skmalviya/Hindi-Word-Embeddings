from flair.embeddings import WordEmbeddings, CharacterEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence

# Hindi embeddings trained with Wikipedia
# https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/hi-wiki-fasttext-300d-1M.vectors.npy'
# https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/hi-wiki-fasttext-300d-1M
hindi_wikipedia_embeddings = WordEmbeddings('hi')

# Hindi embeddings trained with Web crawls
# https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/hi-crawl-fasttext-300d-1M.vectors.npy
# https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/hi-crawl-fasttext-300d-1M
hindi_webcrawl_embeddings = WordEmbeddings('hi-crawl')

# https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-stefan-it/lm-hi-opus-large-forward-v0.1.pt
hindi_flair_forward = FlairEmbeddings('hi-forward')
# https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-stefan-it/lm-hi-opus-large-backward-v0.1.pt
hindi_flair_backword = FlairEmbeddings('hi-backward')


sentence = Sentence('घास हरी है । ')


# just embed a sentence using the StackedEmbedding as you would with any single embedding.
hindi_wikipedia_embeddings.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding, token.embedding.shape)
