import pynlpir
from gensim import corpora, models
import gensim
import numpy as np
import scipy.stats as stats
import os
import io
import csv
import json
import math

class LDA(object):
  """docstring for LDA"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    self.docs, self.dictionary, self.corporas = self._part_document()

  def _part_document(self):
    pynlpir.open()
    docs = {}
    for dirname, dirnames,filenames in os.walk('dependence/data'):
      for filename in filenames:
        path = os.path.join(dirname, filename)
        text = ''
        with io.open(path, 'r',encoding='utf-8') as f:
          text = f.readline()
          words = pynlpir.segment(text,pos_tagging=False)
          clean_words = [w for w in words if w not in self.stop_list and len(w)>1]
          company = filename[:6]
          docs[company] = clean_words
    dictionary = corpora.Dictionary(docs.values())
    corporas = [dictionary.doc2bow(text) for text in docs.values()]
    return docs, dictionary, corporas

  def _get_stop_list(self,path='dependence/new_list.txt'):
    stop_list = []
    with io.open(path,'r',encoding='utf-8') as f:
      lines = f.readlines()
      for l in lines:
        l = l.strip()
        stop_list.append(l)
    return stop_list

  def train(self,num_topics,alpha):
    model = gensim.models.ldamodel.LdaModel(self.corporas, alpha=alpha, num_topics=num_topics, id2word=self.dictionary, minimum_probability=0, iterations=5000)
    # model = gensim.models.ldamulticore.LdaMulticore(self.corporas, alpha=alpha, num_topics=num_topics, id2word=self.dictionary, workers=3, iterations=3000)
    # model = gensim.models.wrappers.LdaMallet('dependence/mallet-2.0.8RC3/bin/mallet', corpus=self.corporas, num_topics=num_topics, id2word=self.dictionary, workers=4, prefix=None, optimize_interval=10, iterations=2000)
    # perplex = model.bound(self.corporas)
    # print(model.print_topics(num_topics))
    return model

  def get_company_corporas(self,company):
    company_corporas = self.dictionary.doc2bow(self.docs[company])
    return company_corporas

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
  for dirname, dirnames,filenames in os.walk('dependence/data'):
    for filename in filenames:
      path = os.path.join(dirname, filename)
      text = ''
      with open(path) as f:
        print(path)
        text = f.readline()
      write_path = os.path.join('dependence/new_data', filename[:6]+'.txt')
      with open(write_path,'w',encoding='utf-8') as f:
        f.write(text)

def optimize_parameters(filename):
  num_topic_list = [5*x for x in range(1,20)]
  alpha_list = [0.01*x for x in range(1,20)]

  m = LDA()
  import sys
  best_p = -sys.maxsize
  best_parameters = []
  record_list = []
  for n in num_topic_list:
    for a in alpha_list:
      perplex = m.train(n,a)
      if perplex>best_p: 
        best_p = perplex
        best_parameters = [n,a,perplex]
      record_list.append([n,a,perplex])
      print([n,a,perplex])
  with open('dependence/perplex/'+filename,'w',newline='') as f:
    a = csv.writer(f)
    a.writerows(record_list)
  return best_parameters

class CompanySimilairy(object):
  """docstring for CompanySimilairy"""
  def __init__(self,num_topics,alpha):
    self.company_list = self._get_company_list()
    self.lda = LDA()
    self.model = self.lda.train(num_topics,alpha)

  def _get_company_list(self):
    company_list = []
    for dirname, dirnames,filenames in os.walk('dependence/data'):
      for filename in filenames:
        company_list.append(filename[:6])
    return company_list

  def get_company_topics(self,company):
    company_corporas = self.lda.get_company_corporas(company)
    return self.model.get_document_topics(company_corporas)

  def _topic_similarity(self,company_a,company_b):
    topic_a = self.get_company_topics(company_a)
    topic_b = self.get_company_topics(company_b)
    kl = self.kl_divergence(topic_a,topic_b)
    return math.exp(-1*kl)

  def kl_divergence(self,p,q):
    return np.sum([stats.entropy(p,q),stats.entropy(q,p)])

  def save_document_similarity_matrix(self,path='dependence/similarity/'):
    similarity_matrix = {}
    for c in self.company_list:
      similarity_matrix[c] = {}
      for d in self.company_list:
        if c != d:
          if d in similarity_matrix: similarity_matrix[c][d] = similarity_matrix[d][c]
          else: similarity_matrix[c][d] = str(self._topic_similarity(c,d))
          print((c,d))
      with open(path+c+'.json','w') as f: json.dump(similarity_matrix[c],f)
    return similarity_matrix

def test():
  c = CompanySimilairy(15,0.04)
  print(c.get_company_topics('430041'))
  c.save_document_similarity_matrix()
  return c