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
import logging

class LoadData(object):
  """docstring for LoadData"""
  def __init__(self):
    self.stop_list = self._get_stop_list()
    self.docs, self.dictionary, self.corporas = self._part_document()
    self.doc_list = self.docs.keys()

  def _part_document(self):
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
        text = rf.readlines()
        rf.close()
        document = ''.join(text).replace('\n','')
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
  def __init__(self,docs,dictionary,corporas,word2vec_path,size=300):
    self.size = size
    self.docs = docs
    self.dictionary = dictionary
    self.corporas = corporas
    self.tfidf = self.__tfidf_model()
    self.model = self.load_word2vec_model(word2vec_path)
    self.doc_vectors = self.get_doc_vector_map()
    self.key_docs_map = self.get_key_docs_map()

  def __tfidf_model(self):
    tfidf = gensim.models.tfidfmodel.TfidfModel(self.corporas.values())
    return tfidf

  def load_word2vec_model(self,path):
    model = gensim.models.word2vec.Word2Vec.load(path)
    return model

  def get_doc_tfidf(self,index):
    doc_tfidf = self.tfidf[self.corporas[index]]
    return doc_tfidf

  def get_doc_keys(self,index,top=10,val=0.1):
    doc_tfidf = self.get_doc_tfidf(index)
    doc_tfidf.sort(key=lambda x: x[1])
    keys = [k for k in doc_tfidf if k[0]>val]
    return keys

  def get_key_docs_map(self):
    key_docs_map = {}
    doc_keys_map = {d: self.get_doc_keys(d) for d in self.docs}
    for doc,keys in doc_keys_map.items():
      for k in keys:
        if k[0] not in key_docs_map: key_docs_map[k[0]] = [(doc,k[1])]
        else: key_docs_map[k[0]].append((doc,k[1]))
    for key,doc in key_docs_map.items():
      doc.sort(key=lambda x: x[1])
      key_docs_map[key] = set([d[0] for d in doc])
    return key_docs_map

  def get_word_vector(self,wordid):
    word = self.dictionary[wordid]
    word_vector = self.model[word]
    return self.model[word]

  def get_doc_vector(self,index):
    doc_tfidf = self.get_doc_tfidf(index)
    doc_vector = np.array([0]*self.size,dtype=float)
    word_count = 0
    for w in doc_tfidf:
      wordid,tfidf = w[:]
      if self.dictionary[wordid] in self.model.vocab.keys():
        word_count += 1
        word_vector = self.get_word_vector(wordid)
        doc_vector += tfidf*np.array(word_vector)/word_count
      else: print(self.dictionary[wordid])
    return doc_vector

  def get_doc_vector_map(self):
    vector_map = {}
    for d in self.docs:
      vector_map[d] = self.get_doc_vector(d)
    return vector_map

class Search(object):
  """docstring for Query"""
  def __init__(self,model_path='dependence/word2vec/word2vec_size_300'):
    data = LoadData()
    self.stop_list = set(data.stop_list)
    self.w2v = Word2Vec(data.docs,data.dictionary,data.corporas,model_path,size=300)
    self.vocab = self.w2v.model.vocab
    self.word2id = {word:id for id,word in self.w2v.dictionary.iteritems()}

  def query2words(self,query):
    words = []
    segs = query.split(' ')
    for s in segs:
      s = s.strip() ## need regularization
      if s in self.vocab: words.append(s) ## in word2vec vocab
      else:
        pynlpir.open()
        # words.extend(pynlpir.get_key_words(query,max_words=3))
        word_segs = pynlpir.segment(query,pos_tagging=False)
        for word in word_segs:
          if word not in self.stop_list: words.append(word)
        print(words)
    return words

  def query2vector(self,words):
    query_vector = np.array([0]*self.w2v.size,dtype=float)
    # words = self.query2words(query)
    ## if all words in dictionary, we can use tfidf
    # dictionary = self.w2v.dictionary
    # query_vector = np.array([0]*self.w2v.size,dtype=float)
    # query_corpora = dictionary.doc2bow(words)
    # query_tfidf = self.w2v.tfidf[query_corpora]
    # for w in query_tfidf:
    #   wordid,tfidf = w[:]
    #   word_vector = self.w2v.get_word_vector(wordid)
    #   query_vector += tfidf*np.array(word_vector)
    n = len(words)
    if n>0:
      for word in words:
        query_vector += (1/n)*self.w2v.model[word]
    return query_vector

  def query_key_match(self,words):
    key_docs = self.w2v.key_docs_map
    doc_set = set()
    for word in words:
      if word in self.word2id:
        wordid  = self.word2id[word]
        if wordid in key_docs:
          doc_set.update(key_docs[wordid])
      else: print(str(word)+' is not in dictionary!')
    return doc_set

  def query_vector_relative_docs(self,query,top=100,val=0.3):
    words = self.query2words(query)
    key_match_docs = self.query_key_match(words)
    query_vector = self.query2vector(words)
    docs = []; return_list = []
    for doc,vec in self.w2v.doc_vectors.items():
      sim = cosin_simiarity(query_vector,vec)
      if sim>val: docs.append([sim,doc])
    if docs:
      docs.sort(reverse=True)
      if key_match_docs:
        return_list = [d[1] for d in docs if d[1] in key_match_docs]
      else: return_list = [d[1] for d in docs] # if return_list is null, return query_key_match
    return return_list

  def word_doc_sim(self,word,doc):
    sim = 0
    if word in self.vocab:
      word_vector = self.w2v.model[word]
      sim = cosin_simiarity(self.w2v.doc_vectors[doc],word_vector)
    return sim

  def query_words_relative_docs(self,query,top=100,val=0.3):
    words = self.query2words(query)
    docs = []; return_list = []
    if words:
      for doc in self.w2v.doc_vectors:
        avg_sim = np.average([self.word_doc_sim(w,doc) for w in words])
        if avg_sim>val: docs.append([avg_sim,doc])
      if docs:
        docs.sort(reverse=True)
        return_list = [d[1] for d in docs]
    return return_list

  def word_for_docs(self,word,val):
    doc_sim = {}
    if word in self.vocab:
      query_vector = self.w2v.model[word]
      for doc,vec in self.w2v.doc_vectors.items():
        sim = cosin_simiarity(vec,query_vector)
        if sim>val: doc_sim[doc] = sim
      if not doc_sim: print('no documents is relative to the word with correlation of '+str(val))
    else: print(word + 'not in vocab')
    return doc_sim

search = Search()
def doc_search(query,val=0.3):
  return search.query_vector_relative_docs(query,val)


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

def train_w2v_model(size,save_path='dependence/word2vec/word2vec_2'):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  sentences = gensim.models.word2vec.LineSentence('dependence/corporas/wiki_cn.txt')
  model = gensim.models.word2vec.Word2Vec(size=size,window=5,min_count=5,workers=4)
  model.build_vocab(sentences)
  model.train(sentences)
  model.save(save_path)
  return model

def test():
  l = LoadData()
  w2v = Word2Vec(l.docs,l.dictionary,l.corporas,size=300)
  save_document_similarity_matrix(w2v,l.doc_list,path='dependence/word2vec_similarity/')


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