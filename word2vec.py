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

class LoadData(object):
  """docstring for LoadData"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    # self.docs, self.dictionary, self.corporas = self._part_document()
    # self.doc_list = self.docs.keys()

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

  def document2sentences(self,document):
    pynlpir.open()
    # f = open('dependence/new_data/'+filename,'r')
    # text = f.readline()
    words = pynlpir.segment(document,pos_tagging=False)
    sign = ['。', ';', '.', ';']
    pause_position = []
    if not words: print('word split error')
    for i in range(len(words)):
      w = words[i]
      if w in sign: pause_position.append(i)
    setences = []
    if len(pause_position) == 0:
      clean_d = [s for s in words if s not in self.stop_list]
      setences.append(' '.join(clean_d)+'\n')
    else:
      for i in range(len(pause_position)-1):
        setence = []
        if i == 0: setence = words[:pause_position[i]]
        else: setence = words[pause_position[i]:pause_position[i+1]]
        clean_s = [s for s in setence if s not in self.stop_list]
        setences.append(' '.join(clean_s)+'\n')
    return setences

  def write_sentences_per_doc(self):
    for dirname, dirnames,filenames in os.walk('dependence/new_data'):
      for filename in filenames:
        read_path = os.path.join(dirname, filename)
        rf = open(read_path, 'r',encoding='utf-8')
        document = rf.readline()
        rf.close()

        setences = self.document2sentences(document)
        if len(setences) == 0: print(filename)
        write_path = os.path.join('dependence/new_docs', filename)
        wf = open(write_path, 'w')
        wf.writelines(setences)
        wf.close()

def test():
  l = LoadData()
  l.write_sentences_per_doc()


