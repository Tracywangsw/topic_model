import pynlpir
from gensim import corpora, models
import gensim
import os
import io
import csv

class LDA(object):
  """docstring for LDA"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    self.docs = self._part_document()

  def _part_document(self):
    pynlpir.open()
    docs = []
    for dirname, dirnames,filenames in os.walk('dependence/data'):
      for filename in filenames:
        path = os.path.join(dirname, filename)
        text = ''
        with io.open(path, 'r',encoding='utf-8') as f:
          text = f.readline()
          words = pynlpir.segment(text,pos_tagging=False)
          clean_words = [w for w in words if w not in self.stop_list and len(w)>1]
          docs.append(clean_words)
    return docs

  def _get_stop_list(self,path='dependence/new_list.txt'):
    stop_list = []
    with io.open(path,'r',encoding='utf-8') as f:
      lines = f.readlines()
      for l in lines:
        l = l.strip()
        stop_list.append(l)
    return stop_list

  def train(self,num_topics):
    dictionary = corpora.Dictionary(self.docs)
    corporas = [dictionary.doc2bow(text) for text in self.docs]
    model = gensim.models.ldamodel.LdaModel(corporas, num_topics=num_topics, id2word=dictionary, minimum_probability=0, iterations=2000)
    # model = gensim.models.wrappers.LdaMallet('mallet-2.0.8RC3/bin/mallet', corpus=corporas, num_topics=num_topics, id2word=dictionary, workers=4, prefix=None, optimize_interval=10, iterations=2000)
    perplex = model.bound(corporas)
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

def part_document():
  for dirname, dirnames,filenames in os.walk('./data'):
    for filename in filenames:
      path = os.path.join(dirname, filename)
      text = ''
      with open(path) as f:
        print(path)
        text = f.readline()
      write_path = os.path.join('new_data', str(i)+'.txt')
      with open(write_path,'w',encoding='utf-8') as f:
        f.write(text)

def optimize_parameters(filename):
  num_topic_list = [10*x for x in range(1,20)]
  m = LDA()
  perplex_list = []
  for n in num_topic_list:
    perplex = m.train(n)
    perplex_list.append(perplex)
    print([n,perplex])
  with open('dependence/perplex/'+filename,'w',newline='') as f:
    a = csv.writer(f)
    for s in zip(num_topic_list,perplex_list):
      a.writerow(s)
  return max(perplex_list)

class CompanySimilairy(object):
  """docstring for CompanySimilairy"""
  def __init__(self):
    self.company_list = self._get_company_list()

  # def _get_company_list():

    
  def topic_similarity(self,topic_a,topic_b):
    kl = self.kl_divergence(topic_a,topic_b)
    return math.exp(-1*kl)

  def kl_divergence(self,p,q):
    return np.sum([stats.entropy(p,q),stats.entropy(q,p)])

  # def save_document_similarity_matrix():
