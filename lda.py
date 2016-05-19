import pynlpir

class LDA(object):
  """docstring for LDA"""
  def __init__(self):
    self.stop_list = self._get_stop_list()

  def part_document(self):
    pynlpir.open()
    text = ''
    with open('memect_companyInfo/430003-北京时代-北京时代科技股份有限公司.txt') as f:
      text = f.readline()
      print(text)
    words = pynlpir.segment(text,pos_tagging=False)
    clean_words = [w for w in words if w not in self.stop_list]
    return clean_words

  def _get_stop_list(self,path='new_list.txt'):
    stop_list = []
    with open(path) as f:
      lines = f.readlines()
      for l in lines:
        l = l.strip()
        stop_list.append(l)
    return stop_list

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

