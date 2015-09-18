# Copyright (C) 2015 Kathrin Donandt
# For license information, see LICENSE.txt

#Calculates percentage of words tagged i which in reality are j; saves result to
#"Verwechslungen [taggername][tone[[tag][dis]".xls"
#Looks for the words which are responsible for these errors;
#for each error, a file [orgtag]_as_[taggertag].txt is created

#see instructions to run at the end

import codecs
from bambara_tagging_htmlreaderALL import BambaraTagging
from nltk.tag import untag
import pickle
from collections import Counter
from create_reader import create_reader as import_create_reader
from indivTaggers import *

tone = input("nontonal/tonal? -> ")
while tone!= "nontonal" and tone !="tonal":
    print("wrong input")
    tone = input("nontonal/tonal? -> ")
   
tag = input("POS/Affixes? -> ")
while tag!= "POS" and tag !="Affixes":
    print("wrong input")
    tag = input("POS/Affixes? -> ")

dis = input("Disambiguated? J/N -> ")
while dis != "J" and dis != "N":
    print("wrong input")
    dis = input("Disambiguated? -> ")

def disambiguated_i(sent):
    dis_sent = []
    for w in sent:
        if len(w[1].split("/"))>1:
            return False
    return True

def analyze(option_tones, option_tag, tagger, dis, taggername):
    '''print to file: how mamy times a word was tagged with tag i but in reality is tag j (Verwechslungen),
    the percentage of that error regarding all words (% error(sum_words)) and regarding the only the errors
    (%error(sum_error)).
    First build the confusion matrix to get the tagpairs (tag i, tag j) and their corresponding error
    (Verwechslungen).
    '''
    bambara = import_create_reader(option_tones, option_tag)
    if dis == "J":
        disamb_train_sents = [i for i in bambara.train_sents if disambiguated_i(i) == True]
        disamb_test_sents = [i for i in bambara.test_sents if disambiguated_i(i) == True]#==devset
        bambara.train_sents = disamb_train_sents
        bambara.test_sents = disamb_test_sents
    print("Calculating switches...")
    bambara.calculate_contingenz_with_sets(tagger)
    tagpairs_not_null = bambara.matrix()
    sum_errors = sum([i[2] for i in tagpairs_not_null])
    sum_words = len(sum(bambara.test_sents, []))
    tagpairs_not_null_big = [i for i in tagpairs_not_null if i[2]>=10]
    tagpairs_sorted = sorted(tagpairs_not_null_big, key=lambda tup: tup[2])
    tagpairs_sorted.reverse()
    output = ["errors:"+"\t"+str(sum_errors)+"\nwords:"+"\t"+str(sum_words)+
              "\norg_tag"+"\t"+"tag_tag"+"\t"+"error"+"\t"+"%error(sum_words)"+"\t"+"%error(sum_error)\n"]
    for i in tagpairs_sorted:
        output.append(i[0]+"\t"+i[1]+"\t"+str(i[2]).replace(".",",")+"\t"+str(round(100*i[2]/sum_words,2)).replace(".",",")+"\t"
                      +str(round(100*i[2]/sum_errors,2)).replace(".",",")+"\n")
    file = codecs.open("Results\\Verwechslungen "+taggername+tone+tag+dis+".xls", mode="wb", encoding="utf-8")
    file.writelines(output)
    file.close()
    return bambara, tagpairs_sorted

def calc_switched_words(bambara, tagger, tagpairs):
    '''iterates over the switched tag-pairs to find all the words which are responsible
    for these switches'''
    untag_testsents = [untag(i) for i in bambara.test_sents]
    tagger_tagged_sents = tagger.tag_sents(untag_testsents)
    compareTags = list(zip(sum(tagger_tagged_sents,[]), sum(bambara.test_sents,[])))
    word_tag_list = sum(bambara.reader.tagged_sents, [])
    switch_list = [(i[0],i[1]) for i in tagpairs]
    for i in switch_list:
        calc_one_switched_word(i[0], i[1], compareTags, word_tag_list)


def calc_one_switched_word(o_tag, t_tag, compareTags, wordtaglist):
    '''looks for words, which where tagged with t_tag by the tagger
    but have o_tag in the original corpus'''
    print("Calculating words responsible for switches...")
    switch = []
    for i in compareTags:
        if i[0][1] == t_tag and i[1][1] == o_tag:
            switch.append(i)
    switchset = set(switch)
    sc = Counter(switch)
    in_corp = dict()
    c = Counter(wordtaglist)
    for i in switchset:
        in_corp[i[0][0]+"_as_"+t_tag]=c[i[0]]
        in_corp[i[0][0]+"_as_"+o_tag]=c[i[1]]
    ### write in table
    t_tag = t_tag.replace("/", "_")
    o_tag = o_tag.replace("/","_")
    lines = [[o_tag, "->",t_tag, "\t", "#switch in testsents", "\t",  "# Vorkommen als "+t_tag, "\t", "# Vorkommen als "+o_tag, "\n"]]
    for i in switchset:
        lines.append([i[0][0],5*"\t", str(sc[i]), 5*"\t", str(c[i[0]]), 5*"\t", str(c[i[1]]), "\n"])
    o_tag = o_tag.replace("|", "_")
    t_tag = t_tag.replace("|", "_")
    file = codecs.open("Results\\"+o_tag+"_as_"+t_tag+".txt", mode="wb", encoding="utf-8")
    lines = sum(lines,[])
    file.writelines(lines)
    file.close()


### 1) load trained tagger, if you saved it (with pickle) e.g.:
##file = open("crfTonalAffixesDisambiguated.pickle", "rb")
##crf = pickle.load(file)
##crf.set_model_file("Results/model.indivCRFTonalAffixesDis.tagger")

##then run
##(bambara, tagpairs)= analyze(tone, tag, crf, dis, "crf")
##calc_switched_words(bambara, crf, tagpairs)



#### 2) train a tagger, e.g. (disambiguated):
##bambara = import_create_reader(tone, tag)

#### disambiguate if you wish to
##disamb_train_sents = [i for i in bambara.train_sents if disambiguated_i(i) == True]
##disamb_test_sents = [i for i in bambara.test_sents if disambiguated_i(i) == True]#==devset
##disamb_testset =  [i for i in bambara.testset if disambiguated_i(i) == True]

##bambara.train_sents = disamb_train_sents
##bambara.test_sents = disamb_test_sents
##bambara.testset=disamb_testset
##unigram = indivUnigram(bambara, None)

#### finally run
##(bambara, tagpairs)= analyze(tone, tag, unigram, "J", "unigram")
##calc_switched_words(bambara, unigram, tagpairs)





