import urllib.request

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en'
urllib.request.urlretrieve(url, './data/train_en.txt')

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi'
urllib.request.urlretrieve(url, './data/train_vi.txt')

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en'
urllib.request.urlretrieve(url, './data/test_en.txt')

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi'
urllib.request.urlretrieve(url, './data/test_vi.txt')

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.en'
urllib.request.urlretrieve(url, './data/vocab_en.txt')

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.vi'
urllib.request.urlretrieve(url, './data/vocab_vi.txt')

url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/dict.en-vi'
urllib.request.urlretrieve(url, './data/dict_en_vi.txt')
