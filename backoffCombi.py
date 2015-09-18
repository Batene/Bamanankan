# Copyright (C) 2015 Kathrin Donandt
# For license information, see LICENSE.txt

#This program combines taggers with backoff: if the first is unable to
#classify a word, it passes this word to the next tagger in the backoff-list

from nltk.tag import UnigramTagger, TrigramTagger, DefaultTagger, AffixTagger, tnt, BigramTagger, untag
from nltk.tag.hmm import HiddenMarkovModelTrainer, LidstoneProbDist
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import ToolboxCorpusReader
from crf import CRFTagger
import getpass
from os import listdir
from regextagger_tonal import Regexp as RegexpTonal
from regextagger_non_tonal import Regexp
from regextagger_non_tonal_SA import Regexp as RegexpSA
from regextagger_tonal_SA import Regexp as RegexpTonalSA
from bambara_tagging_htmlreaderALL import BambaraTagging
from toolboxreaderRun import get_alt_pos
import os
from collections import Counter
from create_reader import create_reader, dictionary
from indivTaggers import *

defaultTagger = DefaultTagger("n")


def backoff_tagger(num_tagger, bambara, option_tones="tonal", option_tag="POS",backoff=defaultTagger):
    """ backoff_tagger of the NLTK cookbook [adapted] """
    taggers = []
    for i in num_tagger:
        if i == 0:
            taggers = taggers + [UnigramTagger]
        if i == 1:
            taggers = taggers + [BigramTagger]
        if i == 2:
            taggers = taggers + [TrigramTagger]
        if i == 3:
            taggers = taggers + [QuadgramTagger]
        if i == 4:
            taggers+=["crf"]
        if i == 5:
            taggers+=["regexp"]
        if i == 6:
            taggers+=["dic"]
        if i == 8:
            taggers+=["affix"]
        if i == 9:
            taggers+=["tnt"]
        if i == 10:
            taggers+=["hmm"]
    #CRF and HMM both do not accept backoff and therefore can only be the last tagger in a backoff chain
    # -> DefaultTagger has to be substituted
    if "hmm" in taggers:
        tag_set= set()
        symbols=set()
        for i in bambara.train_sents:
            for j in i:
                tag_set.add(j[1])
                symbols.add(j[0])
        trainer = HiddenMarkovModelTrainer(list(tag_set), list(symbols))
        hmm = trainer.train_supervised(bambara.train_sents, estimator=lambda fd, bins:LidstoneProbDist(fd, 0.1, bins))
        backoff = hmm
        taggers.remove("hmm")
    if "crf" in taggers:
        backoff = indivCRF(bambara, tone, tag) 
        backoff.train(bambara.train_sents,"model.crfbackoff"+option_tag+option_tones+".tagger")
        backoff.set_model_file("model.crfbackoff"+option_tag+option_tones+".tagger")
        taggers.remove("crf")                                              
    for cls in taggers:
        if cls != "tnt" and cls!="affix" and cls!="regexp" and cls!="dic":
            backoff1 = backoff
            backoff = cls(bambara.train_sents, backoff=backoff1)
            #print(backoff._taggers)
        else:
            if cls == "dic":
                backoff=dictionary_backoff(option_tones, backoff=backoff)
            if cls == "regexp":
                if option_tones == "tonal" and option_tag == "Affixes":
                    backoff=RegexpTonalSA(backoff=backoff)
                if option_tones == "tonal" and option_tag == "POS":
                    backoff=RegexpTonal(backoff=backoff)
                if option_tones == "nontonal" and option_tag == "Affixes":
                    backoff=RegexpSA(backoff=backoff)
                if option_tones == "nontonal" and option_tag == "POS":
                    backoff=Regexp(backoff=backoff)
            if cls == "affix":
                backoff = AffixTagger(bambara.train_sents, min_stem_length=0, affix_length=-4, backoff = backoff)
            if cls == "tnt":
                backoff = tnt.TnT(unk=backoff, Trained= True, N=100)
                backoff.train(bambara.train_sents)
    return backoff
        

def calculate_tagXprec_rec(tags_orig, tags_tagger, tag_i, counter_tagger, counter_orig):
    '''calculates precision and recall of the tagger for a specific tag X'''
    tagI_occ_tagger = counter_tagger[tag_i]#occurence of tag_i in tagger´s tagged sentences
    tagI_occ_original = counter_orig[tag_i]
    zipped = list(zip(tags_orig, tags_tagger))
    zippedCounter = Counter(zipped)
    tagI_occ_orig_and_tagger = zippedCounter[(tag_i, tag_i)]
    if tagI_occ_tagger == 0:
        precision = 0
    else:
        precision =(tagI_occ_orig_and_tagger/tagI_occ_tagger)
    if tagI_occ_original == 0:
        recall = 0
    else:
        recall = (tagI_occ_orig_and_tagger/tagI_occ_original)
    return precision, recall
    

def overall_precision(precisions, recalls, counter_tagger, counter_orig, tagset): # it´s the same as accuracy if each word has only and exactly one tag
    '''calculates weighted average of precision and recall'''
    prec_freq = 0
    for i in tagset:
        prec_freq+=(counter_tagger[i]*precisions[i])
    final_prec = prec_freq/(sum(counter_tagger.values()))
    recall_freq = 0
    for i in tagset:
        recall_freq+=(counter_orig[i]*recalls[i])
    final_recall=recall_freq/(sum(counter_orig.values()))
    f = 2*(final_prec*final_recall)/(final_prec+final_recall)
    return final_prec, final_recall, f
    


def backoff_evaluation(tone, tag):
    '''Calculates various backoff-combinations'''
    
    bambara = create_reader(tone, tag)
    print("Calculating Backoff-Combinations...\n")

    unidef = backoff_tagger([2,1,0], bambara,option_tones=tone, option_tag=tag)
    print("unidef acc: ", unidef.evaluate(bambara.test_sents))
    print("hallo")
    
    unidef = backoff_tagger([0], bambara,option_tones=tone, option_tag=tag)
    print("unidef acc: ", unidef.evaluate(bambara.test_sents))

    uniregdef = backoff_tagger([5,0],bambara, option_tones=tone, option_tag=tag)
    print("uniregdef acc: ", uniregdef.evaluate(bambara.test_sents))

    unidicdef = backoff_tagger([6,0], bambara,option_tones=tone, option_tag=tag)
    print("unidicdef acc: ", unidicdef.evaluate(bambara.test_sents))

    unibidef = backoff_tagger([1,0], bambara,option_tones=tone, option_tag=tag)
    print("unibidef acc: ", unibidef.evaluate(bambara.test_sents))

    unibidicdef = backoff_tagger([6,1,0], bambara,option_tones=tone, option_tag=tag)
    print("unibidicdef acc: ", unibidicdef.evaluate(bambara.test_sents))

    uniaffixdef = backoff_tagger([8,0],bambara,option_tones=tone, option_tag=tag)
    print("uniaffixdef acc: ", uniaffixdef.evaluate(bambara.test_sents))

    uniaffixregdef = backoff_tagger([5,8,0],bambara,option_tones=tone, option_tag=tag)
    print("uniaffixregdef acc: ", uniaffixregdef.evaluate(bambara.test_sents))

    uniaffixdicdef = backoff_tagger([6,8,0],bambara,option_tones=tone, option_tag=tag)
    print("uniaffixdicdef acc: ", uniaffixdicdef.evaluate(bambara.test_sents))

    uniaffixdicdef = backoff_tagger([5,6,8,0],bambara,option_tones=tone, option_tag=tag)
    print("uniaffixdicregdef acc: ", uniaffixdicdef.evaluate(bambara.test_sents))


    bidef = backoff_tagger([1],bambara, option_tones=tone, option_tag=tag)
    print("bief acc: ", bidef.evaluate(bambara.test_sents))

    biregdef = backoff_tagger([5,1],bambara,option_tones=tone, option_tag=tag)
    print("biregdef acc: ", biregdef.evaluate(bambara.test_sents))

    bidicdef = backoff_tagger([6,1], bambara,option_tones=tone, option_tag=tag)
    print("bidicdef acc: ", bidicdef.evaluate(bambara.test_sents))

    biunidef = backoff_tagger([0,1], bambara,option_tones=tone, option_tag=tag)
    print("biunidef acc: ", biunidef.evaluate(bambara.test_sents))

    biunidicdef = backoff_tagger([6,0,1], bambara,option_tones=tone, option_tag=tag)
    print("biunidicdef acc: ", biunidicdef.evaluate(bambara.test_sents))

    biaffixdef = backoff_tagger([8,1],bambara,option_tones=tone, option_tag=tag)
    print("biaffixdef acc: ", biaffixdef.evaluate(bambara.test_sents))

    biaffixregdef = backoff_tagger([5,8,1],bambara,option_tones=tone, option_tag=tag)
    print("biaffixregdef acc: ", biaffixregdef.evaluate(bambara.test_sents))

    biaffixdicdef = backoff_tagger([6,8,1],bambara,option_tones=tone, option_tag=tag)
    print("biaffixdicdef acc: ", biaffixdicdef.evaluate(bambara.test_sents))

    biaffixdicdef = backoff_tagger([5,6,8,1],bambara,option_tones=tone, option_tag=tag)
    print("biaffixdicregdef acc: ", biaffixdicdef.evaluate(bambara.test_sents))

    tridef = backoff_tagger([2], bambara,option_tones=tone, option_tag=tag)
    print("tridef acc: ", tridef.evaluate(bambara.test_sents))

    triregdef = backoff_tagger([5,2],bambara,option_tones=tone, option_tag=tag)
    print("triregdef acc: ", triregdef.evaluate(bambara.test_sents))

    tridic = backoff_tagger([6,2],bambara,option_tones=tone, option_tag=tag)
    print("tridic acc: ", tridic.evaluate(bambara.test_sents))

    tribiuni = backoff_tagger([0,1,2],bambara,option_tones=tone, option_tag=tag)
    print("tribiuni acc: ", tribiuni.evaluate(bambara.test_sents))

    tribiunidic = backoff_tagger([6,0,1,2],bambara,option_tones=tone, option_tag=tag)
    print("tribiunidic acc: ", tribiunidic.evaluate(bambara.test_sents))

    triaffix = backoff_tagger([8,2],bambara,option_tones=tone, option_tag=tag)
    print("triaffix acc: ", triaffix.evaluate(bambara.test_sents))

    triaffixreg = backoff_tagger([5,8,2],bambara,option_tones=tone, option_tag=tag)
    print("triaffixreg acc: ", triaffixreg.evaluate(bambara.test_sents))

    triaffixdic = backoff_tagger([6,8,2],bambara,option_tones=tone, option_tag=tag)
    print("triaffixdic acc: ", triaffixdic.evaluate(bambara.test_sents))

    triaffixdicreg = backoff_tagger([5,6,8,2],bambara,option_tones=tone, option_tag=tag)
    print("triaffixdicregdef acc: ", triaffixdicreg.evaluate(bambara.test_sents))


    tntdef = backoff_tagger([9],bambara,option_tones=tone, option_tag=tag)
    print("tntdef acc: ", tntdef.evaluate(bambara.test_sents))

    tntreg = backoff_tagger([5,9],bambara,option_tones=tone, option_tag=tag)
    print("tntreg acc: ", tntreg.evaluate(bambara.test_sents))

    tntdic = backoff_tagger([6,9],bambara,option_tones=tone, option_tag=tag)
    print("tntdic acc: ", tntdic.evaluate(bambara.test_sents))

    tntbiuni = backoff_tagger([0,1,9],bambara,option_tones=tone, option_tag=tag)
    print("tntbiuni acc: ", tntbiuni.evaluate(bambara.test_sents))

    tntaffix = backoff_tagger([8,9],bambara,option_tones=tone, option_tag=tag)
    print("tntaffix acc: ", tntaffix.evaluate(bambara.test_sents))

    tntaffixdic = backoff_tagger([6,8,9],bambara,option_tones=tone, option_tag=tag)
    print("tntaffixdic acc: ", tntaffixdic.evaluate(bambara.test_sents))

    tntaffixdic = backoff_tagger([5,6,8,9],bambara,option_tones=tone, option_tag=tag)
    print("tntaffixdicregdef acc: ", tntaffixdic.evaluate(bambara.test_sents))

    tntcrf = backoff_tagger([4,9],bambara,option_tones=tone, option_tag=tag)
    print("tntcrf acc"+tone+" "+tag+": ", tntcrf.evaluate(bambara.test_sents))

    tnthmm = backoff_tagger([10,9], bambara, option_tones=tone, option_tag=tag)
    print("tnthmm acc: ", tnthmm.evaluate(bambara.test_sents))

    print("Calculating precision of individual taggers...\n")
    indivDefault(bambara)
    indivUnigram(bambara,backoff=None)
    indivBigram(bambara,backoff=None)
    indivTrigram(bambara,backoff=None)
    indivAffix(bambara, affix_length=-4, backoff=None)
    indivTnT(bambara, backoff=None)
    indivRegexp(bambara, tag, tone, backoff=None)
    indivDic(bambara, tone)
    indivHMM(bambara)
    indivCRF(bambara)


#run e.g.
#backoff_evaluation("tonal","POS")
