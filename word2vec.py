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
import sys
import pdb
import chardet
from chardet.universaldetector import UniversalDetector

class LoadData(object):
  """docstring for LoadData"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    self.docs, self.dictionary, self.corporas = self._part_document()
    self.doc_list = self.docs.keys()

  def _part_document(self):
    pynlpir.open()
    docs = {}
    for dirname, dirnames,filenames in os.walk('dependence/new_docs'):
      for filename in filenames:
        path = os.path.join(dirname, filename)
        f = open(path)
        text = f.readlines()
        f.close()
        index = filename[:6]
        docs[index] = []
        for line in text:
          for w in line.split(' '):
            docs[index].append(w.strip())
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

  def document2sentences(self,document):
    pynlpir.open()
    words = pynlpir.segment(document,pos_tagging=False)
    sign = ['。', ';', '.', ';']
    pause_position = []
    for i in range(len(words)):
      if words[i] in sign: pause_position.append(i)
    setences = []
    if len(pause_position) == 0:
      clean_d = [s.strip() for s in words if s not in self.stop_list]
      setences.append(' '.join(clean_d)+'\n')
    else:
      for i in range(len(pause_position)):
        setence = []
        if i == 0: setence = words[:pause_position[i]]
        elif i == len(pause_position)-1 and i != 0: break
        else: setence = words[pause_position[i]:pause_position[i+1]]
        clean_s = [s.strip() for s in setence if s not in self.stop_list]
        setences.append(' '.join(clean_s)+'\n')
    return setences

  def write_sentences_per_doc(self):
    for dirname, dirnames,filenames in os.walk('dependence/new_data'):
      for filename in filenames:
        read_path = os.path.join(dirname, filename)
        rf = open(read_path, 'r',encoding='utf-8')
        document = rf.readline()
        rf.close()
        if document == '': print(filename)
        setences = self.document2sentences(document)
        if not setences: print(filename)
        write_path = os.path.join('dependence/new_docs', filename)
        wf = open(write_path, 'w')
        wf.writelines(setences)
        wf.close()

def line_ss():
  lines = []
  for dirname, dirnames,filenames in os.walk('dependence/new_docs'):
    for filename in filenames:
      read_path = os.path.join(dirname, filename)
      rf = open(read_path, 'r',encoding='utf-8')
      # pdb.set_trace()
      document = rf.readlines()
      lines.extend(document)
      rf.close()
  wf = open('dependence/input_sentences.txt', 'w')
  wf.writelines(lines)
  wf.close()

class Word2Vec(object):
  """docstring for Word2Vec"""
  def __init__(self,docs,dictionary,corporas,size=300):
    self.size = size
    self.docs = docs
    self.dictionary = dictionary
    self.corporas = corporas
    self.tfidf = self.__tfidf_model()
    self.model = self.__w2v_model(size)

  def __tfidf_model(self):
    tfidf = gensim.models.tfidfmodel.TfidfModel(self.corporas.values())
    return tfidf

  def __w2v_model(self,size):
    sentences = gensim.models.word2vec.LineSentence('dependence/corporas/wiki_cn.txt')
    model = gensim.models.word2vec.Word2Vec(sentences,size=size,window=5,min_count=5,workers=4)
    model.save('word2vec')
    return model

  def get_doc_tfidf(self,index):
    doc_tfidf = self.tfidf[self.corporas[index]]
    return doc_tfidf

  def get_word_vector(self,wordid):
    word = self.dictionary[wordid]
    word_vector = self.model[word]
    return self.model[word]

  def get_doc_vector(self,index):
    doc_tfidf = self.get_doc_tfidf(index)
    doc_vector = np.array([0]*self.size,dtype=float)
    for w in doc_tfidf:
      wordid,tfidf = w[:]
      if self.dictionary[wordid] in self.model.vocab.keys():
        word_vector = self.get_word_vector(wordid)
        doc_vector += tfidf*np.array(word_vector)/len(doc_tfidf)
      else: print(self.dictionary[wordid])
    return doc_vector

def cosin_simiarity(vector_a,vector_b):
  norm_a = np.linalg.norm(vector_a)
  norm_b = np.linalg.norm(vector_b)
  inner_ab = np.dot(vector_a,vector_b)
  return inner_ab/(norm_a*norm_b)

def save_document_similarity_matrix(model,document_list,similarity=cosin_simiarity,path='dependence/similarity/'):
  similarity_matrix = {}
  for c in document_list:
    similarity_matrix[c] = {}
    for d in document_list:
      if c != d:
        if d in similarity_matrix: similarity_matrix[c][d] = similarity_matrix[d][c]
        else:
          vector_c = model.get_doc_vector(c)
          vector_d = model.get_doc_vector(d)
          similarity_matrix[c][d] = str(similarity(vector_c,vector_d))
        print((c,d))
    with open(path+c+'.json','w') as f: json.dump(similarity_matrix[c],f)
  return similarity_matrix

def part_sentence(stop_list):
  pynlpir.open()
  for dirname, dirnames,filenames in os.walk('dependence/ch_corporas/wiki/lost'):
      for filename in filenames:
        lines = []
        read_path = os.path.join(dirname, filename)
        rf = open(read_path,'rb')
        print(filename)
        for line in rf:
          # detector.feed(byte)
          encoding = chardet.detect(line)['encoding']
          if encoding == None: encoding = 'utf-8'
          new_line = line.decode(encoding,'ignore')
          words = pynlpir.segment(new_line,pos_tagging=False)
          clean_words = [w.strip() for w in words if w not in stop_list]
          str_line = ' '.join(clean_words)
          if str_line: lines.append(str_line+'\n')
        rf.close()
        write_path = os.path.join('dependence/ch_corporas/wiki_clean', filename)
        wf = open(write_path, 'w')
        wf.writelines(lines)
        wf.close()



def test():
  l = LoadData()
  w2v = Word2Vec(l.docs,l.dictionary,l.corporas,size=50)
  save_document_similarity_matrix(w2v,l.doc_list,path='dependence/word2vec_similarity/')
  # part_sentence(l.stop_list)
  # l.write_sentences_per_doc()
