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



class LoadData(object):
  """docstring for LoadData"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    self.docs, self.dictionary, self.corporas = self._part_document()
    self.doc_list = self.docs.keys()

  def _part_document(self):
    pynlpir.open()
    docs = {}
    for dirname, dirnames,filenames in os.walk('dependence/new_data'):
      for filename in filenames:
        path = os.path.join(dirname, filename)
        text = ''
        with io.open(path, 'r',encoding='utf-8') as f:
          text = f.readline()
          words = pynlpir.segment(text,pos_tagging=False)
          clean_words = [w for w in words if w not in self.stop_list and len(w)>1]
          index = filename[:6]
          docs[index] = clean_words
    dictionary = corpora.Dictionary(docs.values())
    corporas = {index: dictionary.doc2bow(docs[index]) for index in docs}
    return docs, dictionary, corporas

  def _get_stop_list(self,path='dependence/new_list.txt'):
    stop_list = []
    with io.open(path,'r',encoding='utf-8') as f:
      lines = f.readlines()
      for l in lines:
        l = l.strip()
        stop_list.append(l)
    return stop_list

def recode_document():
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

class LDA(object):
  """docstring for LDA"""
  def __init__(self,dictionary,corporas,num_topics,alpha):
    self.dictionary = dictionary
    self.corporas = corporas
    self.lda = self.train_model(num_topics,alpha)

  def train_model(self,num_topics,alpha):
    corporas = self.corporas.values()
    model = gensim.models.ldamodel.LdaModel(corporas, alpha=alpha, num_topics=num_topics, id2word=self.dictionary, minimum_probability=0, iterations=5000)
    # model = gensim.models.ldamulticore.LdaMulticore(self.corporas, alpha=alpha, num_topics=num_topics, id2word=self.dictionary, workers=3, iterations=3000)
    return model

  def get_doc_topics(self,index):
    document_corporas = self.corporas[index]
    return self.lda.get_document_topics(document_corporas)

class LSI(object):
  """docstring for LSI"""
  def __init__(self,dictionary,corporas,num_topics):
    self.dictionary = dictionary
    self.corporas = corporas
    self.tfidf = self.__tfidf_model()
    self.lsi = self.train_model(num_topics)

  def __tfidf_model(self):
    tfidf = gensim.models.tfidfmodel.TfidfModel(self.corporas.values())
    return tfidf

  def train_model(self,num_topics):
    corporas = self.corporas.values()
    corporas_tfidf = self.tfidf[corporas]
    model = gensim.models.lsimodel.LsiModel(corporas_tfidf,id2word=self.dictionary,num_topics=num_topics)
    return model

  def get_doc_topics(self,index):
    doc_tfidf = self.tfidf[self.corporas[index]]
    return self.lsi[doc_tfidf]


def optimize_lda_parameters(self,dictionary,corporas,filename):
  num_topic_list = [5*x for x in range(1,20)]
  alpha_list = [0.01*x for x in range(1,20)]
  import sys
  best_p = -sys.maxsize
  best_parameters = []
  record_list = []
  for n in num_topic_list:
    for a in alpha_list:
      model = LDA(dictionary,corporas,n,a)
      perplex = model.bound(corporas.values())
      if perplex>best_p: 
        best_p = perplex
        best_parameters = [n,a,perplex]
      record_list.append([n,a,perplex])
      print([n,a,perplex])
  with open('dependence/perplex/'+filename,'w',newline='') as f:
    a = csv.writer(f)
    a.writerows(record_list)
  return best_parameters


def cosin_simiarity(topic_a,topic_b):
  vector_a = np.array([x[1] for x in topic_a])
  vector_b = np.array([x[1] for x in topic_b])
  norm_a = np.linalg.norm(vector_a)
  norm_b = np.linalg.norm(vector_b)
  inner_ab = np.dot(vector_a,vector_b)
  return inner_ab/(norm_a*norm_b)

def kl_delivergence(topic_a,topic_b):
  kl = np.sum([stats.entropy(topic_a,topic_b),stats.entropy(topic_b,topic_a)])
  return math.exp(-1*kl)

class DocSimilairy(object):
  """docstring for CompanySimilairy"""
  def __init__(self,model):
    self.model = model

  def save_document_similarity_matrix(self,document_list,similarity=cosin_simiarity,path='dependence/similarity/'):
    similarity_matrix = {}
    for c in document_list:
      similarity_matrix[c] = {}
      for d in document_list:
        if c != d:
          if d in similarity_matrix: similarity_matrix[c][d] = similarity_matrix[d][c]
          else:
            topic_c = self.model.get_doc_topics(c)
            topic_d = self.model.get_doc_topics(d)
            similarity_matrix[c][d] = str(similarity(topic_c,topic_d))
          print((c,d))
      with open(path+c+'.json','w') as f: json.dump(similarity_matrix[c],f)
    return similarity_matrix

def run_lsi():
  data = LoadData()
  # lda = LDA(data.dictionary,data.corporas,15,0.05)
  lsi = LSI(data.dictionary,data.corporas,15)
  sim = DocSimilairy(lsi)
  sim.save_document_similarity_matrix(data.doc_list,path='dependence/doc_similarity/lsi_similarity/')
  return sim

def run_lda():
  data = LoadData()
  lda = LDA(data.dictionary,data.corporas,15,0.05)
  sim = DocSimilairy(lda)
  sim.save_document_similarity_matrix(data.doc_list,similarity=kl_delivergence,path='dependence/doc_similarity/lda_similarity/')
  return sim
