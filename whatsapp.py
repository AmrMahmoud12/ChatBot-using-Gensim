import pandas as pd
df=pd.read_csv('z.csv')
train_data = pd.DataFrame(columns = ['id','text','response','name'])

prev_msg = ''
for index, row in df.iterrows():
    if prev_msg != '':
        tmp = pd.DataFrame({'text': [prev_msg], 'response': [row['message']], 'id': [row['id']], 'name': [row['name']]})
        train_data = train_data.append(tmp[['id','text','response','name']], ignore_index=True)
    prev_msg = row['message']
train_data.head()

import gensim

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import multiprocessing
import os

class MyTexts(object):
    def __iter__(self):
        for i in range(len(train_data)):
            yield TaggedDocument(words=simple_preprocess(train_data['text'][i]), tags=[train_data['id'][i]])

assert gensim.models.doc2vec.FAST_VERSION > -1
cores = multiprocessing.cpu_count()
texts = MyTexts()
doc2vec_model = Doc2Vec(vector_size=300, workers=cores, min_count=1, window=3, negative=5)
doc2vec_model.build_vocab(texts)
doc2vec_model.train(texts, total_examples=doc2vec_model.corpus_count, epochs=20)
if not os.path.exists('models'):
    os.makedirs('models')

doc2vec_model.save('models/doc2vec.model')
doc2vec_model.save_word2vec_format('models/trained.word2vec')


from colorama import Fore
from IPython.display import clear_output
from IPython.display import display
from ipywidgets import Output


def chatbot():
    quit = False
    responses = []

    while quit == False:
        text = str(input('Message: '))
        if text == 'quit()':
            quit = True
        else:
            tokens = text.split()
            ##infer vector for text the model may not have seen
            new_vector = doc2vec_model.infer_vector(tokens)
            ##find the most similar [i] tags
            index = doc2vec_model.docvecs.most_similar([new_vector], topn=10)
            response = Fore.RED + 'Chatbot: ' + train_data.iloc[int(index[0][0])].response
            responses.append(response)
            out = Output()
            display(out)
            with out:
                clear_output()
                print(response)

def chating(m):
        doc2vec_model = Doc2Vec.load('models/doc2vec.model')

        tokens = m.split()
        new_vector=doc2vec_model.infer_vector(tokens)
        index = doc2vec_model.docvecs.most_similar([new_vector], topn=10)
        response =  train_data.iloc[int(index[0][0])].response#output
        return response

