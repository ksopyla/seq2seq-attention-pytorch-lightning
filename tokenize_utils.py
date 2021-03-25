
from collections import Counter, OrderedDict

from itertools import zip_longest, tee


special_tokens=["<pad>", "<unk>", "</s>", "<s>", "<copy>", "<mask>"]
special_tokens = {token: 100 for token in special_tokens}


## ngram iterators and tokenizers, working on list or generators

from collections import Counter, OrderedDict
import itertools



## ngram iterators and tokenizers, working on list or generators

from itertools import zip_longest
def ngram_tokenizer_iter(iterable, n, fillvalue=''):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    zip_tuples = zip_longest(*args, fillvalue=fillvalue)
    for tup in zip_tuples:
        yield "".join(tup)

def ngram_tokenizer(ngrams):
    '''generates ngram as separate groups of chars [abcd]->ab, cd
    '''
    def func(text):
        return (text[pos:pos + ngrams] for pos in range(0, len(text), ngrams))
    return func


def ngram_vocab_gen(ngrams):
    '''generates all ngrams [abcd]->ab, bc, cd, d
    '''
    def func(text):
        return (text[i:i+ngrams] for i in range(len(text)+1-ngrams))
    return func



def ngram_vocab_gen_iter(iterator, ngrams):
    '''generates all ngrams with iterators
    eg. bigrams [abcd]->ab, bc, cd, d 
    trigrams: s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,4), ...
    '''
    iter_tuple = itertools.tee(iterator, ngrams)

    nested_iter_next=0
    list_of_iters = []
    for i,one_iter in enumerate(iter_tuple):
       
        for _ in range(nested_iter_next):
            next(one_iter,"")
       
        list_of_iters.append(one_iter)
        nested_iter_next+=1
    
    for tup in zip(*list_of_iters):
        yield "".join(tup)


def uni_bi_grams_vocab_gen(text):
    ''' generates all unigram and bigrams 'abc' -> 'a','b','ab','b','c','bc','c'
    '''
    first = text[0]
    yield first

    last=''
    for pos in range(1, len(text), 1):
        #yield first
        last = text[pos]
        yield last
        yield first + last
        first = last
        
def uni_bi_grams_vocab_gen_seq(text):
    """generates list of ngrams and single characters"""
    for x in list(text):
        yield x
    for x in ngram_vocab_gen(2)(text):
        yield x

def bigrams_tokenize(text):
    '''tokenize to bigrams 'abcde' -> 'ab','cd', 'e'
    '''
    #(text[pos:pos + ngrams] for pos in range(0, len(text), ngrams)) 
    for pos in range(0, len(text), 2):
        yield text[pos:pos+2]





    
    