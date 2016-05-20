import pynlpir
#from gensim import corpora, models
#import gensim
import os
import shutil

class LDA(object):
  """docstring for LDA"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    self.docs = self._part_document()

  def _part_document(self):
    pynlpir.open()
    docs = []
    for dirname, dirnames,filenames in os.walk('./data'):
      i = 0
      for filename in filenames:
        i += 1
        if i>4944: break
        path = os.path.join(dirname, filename)
        text = ''
        with open(path) as f:
          print(path)
          text = f.readline()
          words = pynlpir.segment(text,pos_tagging=False)
          clean_words = [w for w in words if w not in self.stop_list]
          docs.append(clean_words)
          print(len(docs))
    return docs

  def _get_stop_list(self,path='new_list.txt'):
    stop_list = []
    with open(path,encoding="utf-8") as f:
      lines = f.readlines()
      for l in lines:
        l = l.strip()
        stop_list.append(l)
    return stop_list

  def train(self,num_topics):
    dictionary = corpora.Dictionary(self.docs)
    corpora = [dictionary.doc2bow(text) for text in self.docs]
    # model = gensim.models.ldamodel.LdaModel(self.corpora, num_topics=num_topics, id2word=dictionary, alpha='auto', minimum_probability=0)
    gensim.models.wrappers.LdaMallet('mallet-2.0.8RC3/bin/mallet', corpus=corpora, num_topics=num_topics, id2word=dictionary, workers=4, prefix=None, optimize_interval=10, iterations=2000)
    perplex = model.bound(corpora)
    return perplex


def make_stop_list():
  with open('stop_list.txt') as f:
    lines = f.readlines()
    for l in lines:
      l = l.strip()
      arr = l.split(',')
      print(arr)
      for a in arr:
        if a not in stop_list:
          stop_list.append(a)
  with open('new_list.txt','w') as f:
    for s in stop_list:
      f.write(s+'\n')

def _part_document():
  for dirname, dirnames,filenames in os.walk('./data'):
    i = 0
    for filename in filenames:
      i += 1
      if i>4944: break
      path = os.path.join(dirname, filename)
      text = ''
      with open(path) as f:
        print(path)
        text = f.readline()
      write_path = os.path.join('new_data', str(i)+'.txt')
      with open(write_path,'w',encoding='utf-8') as f:
        f.write(text)

def move_file():
  for dirname, dirnames,filenames in os.walk('./data'):
    i = 0
    for filename in filenames:
      i += 1
      if i>4944: break
      if filename.endswith('_1.txt'):
        path = os.path.join(dirname, filename)
        shutil.move(path,"./new_data/"+filename)
