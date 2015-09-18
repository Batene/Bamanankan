# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

## functions for training individual taggers

from crf import CRFTagger
from create_reader import create_reader
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger, AffixTagger, tnt
from regextagger_tonal import Regexp as RegexpTonal
from regextagger_non_tonal import Regexp
from regextagger_non_tonal_SA import Regexp as RegexpSA
from regextagger_tonal_SA import Regexp as RegexpTonalSA
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag.hmm import LidstoneProbDist
from create_reader import dictionary, dictionary_backoff


def indivDefault(bambara):
    default = DefaultTagger('n')
    print(default.evaluate(bambara.test_sents))
    return default

def indivCRF(bambara, tone, tag):
    crf = CRFTagger(training_opt={'max_iterations':100,'max_linesearch' : 10,'c1': 0.0001,'c2': 1.0})#best training_opt für CRF
# c1 and c2 according to suggestion on http://nbviewer.ipython.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
    print("Training CRF Tagger...")
    crf.train(bambara.train_sents, "Models/model.indivCRF"+tone+tag+".tagger")
    print("CRF accuracy: ",crf.evaluate(bambara.test_sents))
    return crf

def indivTnT(bambara, backoff):
    tnttagger = tnt.TnT(unk=backoff, Trained= True, N=100)
    tnttagger.train(bambara.train_sents)
    print("TnT accuracy: ",tnttagger.evaluate(bambara.test_sents))
    return tnttagger

def indivHMM(bambara):
    tag_set= set()
    symbols=set()
    for i in bambara.train_sents:
        for j in i:
            tag_set.add(j[1])
            symbols.add(j[0])
    trainer = HiddenMarkovModelTrainer(list(tag_set), list(symbols))
    hmm = trainer.train_supervised(bambara.train_sents, estimator=lambda fd, bins:LidstoneProbDist(fd, 0.1, bins))
    print("HMM accuracy:",hmm.evaluate(bambara.test_sents))
    return hmm

def indivAffix(bambara, affix_length, backoff):
    affix=AffixTagger(bambara.train_sents, min_stem_length=0, affix_length=affix_length, backoff = backoff)
    print("Affix accuracy: ",affix.evaluate(bambara.test_sents))
    return affix

def indivUnigram(bambara,backoff):
    unigram= UnigramTagger(bambara.train_sents, backoff=backoff)
    print("Unigram accuracy: ",unigram.evaluate(bambara.test_sents))
    return unigram

def indivBigram(bambara, backoff):
    bigram= BigramTagger(bambara.train_sents, backoff=backoff)
    print("Bigram accuracy: ",bigram.evaluate(bambara.test_sents))
    return bigram

def indivTrigram(bambara,backoff):
    trigram=TrigramTagger(bambara.train_sents, backoff=backoff)
    print("Trigram accuracy: ",trigram.evaluate(bambara.test_sents))
    return trigram

def indivRegexp(bambara, option_tag, option_tones, backoff):
    if option_tones == "tonal" and option_tag == "Affixes":
        regex=RegexpTonalSA(backoff=backoff)
    if option_tones == "tonal" and option_tag == "POS":
        regex=RegexpTonal(backoff=backoff)
    if option_tones == "nontonal" and option_tag == "Affixes":
        regex=RegexpSA(backoff=backoff)
    if option_tones == "nontonal" and option_tag == "POS":
        regex=Regexp(backoff=backoff)
    print("Regexp accuracy: ",regex.evaluate(bambara.test_sents))
    return regex

def indivDic(bambara, tone):#backoff  = DefaultTagger('n')
    dic = dictionary(tone)
    print("Dictionary accuracy: ",dic.evaluate(bambara.test_sents))
    return dic

def indivDic_backoff(bambara,tone,backoff):
    dic = dictionary_backoff(tone, backoff)
    print("Dictionary with backoff accuracy: ",dic.evaluate(bambara.test_sents))
    return dic



