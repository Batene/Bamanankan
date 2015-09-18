# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

# Calculates Compare(A,B), additive complementarity and disagreement
# (for further information, see Brill, Eric; Wu, Jun (1998):
# Classifier combination for improved lexical disambiguation.
# In: Proceedings COLING-ACL’98, S. 191–195.

from nltk.tag.hmm import LidstoneProbDist, HiddenMarkovModelTrainer
from nltk.tag import tnt, UnigramTagger, untag
from crf import CRFTagger
from nltk.metrics import accuracy
from create_reader import create_reader
from indivTaggers import *
import pickle
import codecs
from os import listdir
import getpass
from backoffCombi import backoff_tagger


def compareTagger(tag1, tag2,orgTaggedSents, tag1Sents, tag2Sents):
    '''Compares the incorrect tags of tag1 (Tagger) with the tags of tag2
    and returns the "complementary rate" of these taggers (Brill & Wu 1998);
    further information:
    http://www.evalita.it/sites/evalita.fbk.eu/files/proceedings2009/PoSTagging/POS_ILC.pdf'''
    if tag1 == tag2:
        return "  0  "
    else:
        orgTaggedSents = sum(orgTaggedSents,[])
        tag1Sents = sum(tag1Sents,[])
        tag2Sents = sum(tag2Sents,[])
        righttags = []
        commonErr = 0
        tag1OnlyErr = 0
        for i in range(len(orgTaggedSents)):
            if (tag1Sents[i]) == (orgTaggedSents[i]):
                righttags.append(i)
        righttags = righttags[::-1]
        for i in righttags: #pop the tagged words which were correct in tag1Sents from all lists
            orgTaggedSents.pop(i)
            tag1Sents.pop(i)
            tag2Sents.pop(i)
        tag1OnlyErr = len(tag1Sents)#because in tag1Sents, there are now only the incorrectly tagged words
        commonErr = tag1OnlyErr #at the beginning, assume, tagger2 has also incorrectly tagged the words that tagger1 has
        for i in range(len(orgTaggedSents)):
            if tag2Sents[i] == orgTaggedSents[i]:
                commonErr-=1 #if tagger2 has correctly tagged a word, that tagger1 as incorrectly tagged
        result = round(((1-(commonErr/tag1OnlyErr))*100),2)
        resultfinal = str(result)
        if len(resultfinal)==3:
            resultfinal = " "+" "+resultfinal
        if len(resultfinal)==4:
            resultfinal = " "+resultfinal
        return resultfinal  
                      

def errorrates(crftags, tnttags, hmmtags, utags,crf_err, tnt_err):
    orgtags = [i[1] for i in (sum(bambara.test_sents,[]))]
    crf_disagree = disagreement(crftags, utags, tnttags, hmmtags, orgtags)
    tnt_disagree = disagreement(tnttags, hmmtags, crftags, utags, orgtags)
    hmm_disagree = disagreement(hmmtags, crftags, utags, tnttags, orgtags)
    unigram_disagree = disagreement(utags, tnttags, hmmtags, crftags, orgtags)
    crf_disagree_wu = disagreement_without_unigram(crftags, tnttags, hmmtags, orgtags)
    tnt_disagree_wu = disagreement_without_unigram(tnttags, hmmtags, crftags, orgtags)
    hmm_disagree_wu = disagreement_without_unigram(hmmtags, crftags, tnttags, orgtags)
    errorrates_disagree = [crf_disagree, tnt_disagree, hmm_disagree,unigram_disagree]
    errorrates_disagree_wu = [crf_disagree_wu, tnt_disagree_wu, hmm_disagree_wu]
    add_compl = additive_complementarity(crftags, tnttags, hmmtags, utags, orgtags, crf_err)
    add_compl1 = additive_complementarity3(crftags, tnttags,utags, orgtags, crf_err)
    add_compl2 = additive_complementarity3(crftags, hmmtags, utags, orgtags, crf_err)
    add_compl3 = additive_complementarity3(tnttags,hmmtags,utags,orgtags,tnt_err)
    return (errorrates_disagree, errorrates_disagree_wu, add_compl, add_compl1, add_compl2, add_compl3)
    
def disagreement(tags1, tags2, tags3, tags4, otags):
    disagreement_counter = 0
    disagreement_and_wrong_counter = 0
    for i in range(len(tags1)):
        if (tags1[i] != tags2[i]) or (tags1[i] != tags3[i])  or (tags1[i] != tags4[i]):
            disagreement_counter+=1
            if tags1[i] != otags[i]:
                disagreement_and_wrong_counter+=1
    return (round((disagreement_and_wrong_counter/disagreement_counter), 4))


def disagreement_without_unigram(tags1, tags2, tags3, otags):
    disagreement_counter = 0
    disagreement_and_wrong_counter = 0
    for i in range(len(tags1)):
        if (tags1[i] != tags2[i]) or (tags1[i] != tags3[i]):
            disagreement_counter+=1#??????????
            if tags1[i] != otags[i]:
                disagreement_and_wrong_counter+=1
    return (round((disagreement_and_wrong_counter/disagreement_counter), 4))


def additive_complementarity(tags1, tags2, tags3, tags4, otags, crferror):
    oneadded = 0
    twoadded = 0
    threeadded=0
    for i in range(len(tags1)):
        if (tags1[i] != otags[i]) and (tags2[i] != otags[i]):
            oneadded+=1
            if tags3[i] != otags[i]:
                twoadded+=1
                if tags4[i] != otags[i]:
                    threeadded+=1
    add_compl1 = round(oneadded/(len(otags)),4)
    add_compl2 = round((twoadded/len(otags)),4)
    add_compl3 = round((threeadded/len(otags)),4)
    return ([crferror,add_compl1, add_compl2, add_compl3])

## three Taggers
def additive_complementarity3(tags1, tags2, tags3, otags, single_error ):
    oneadded = 0
    twoadded = 0
    for i in range(len(tags1)):
        if (tags1[i] != otags[i]) and (tags2[i] != otags[i]):
            oneadded+=1
            if tags3[i] != otags[i]:
                twoadded+=1
    add_compl1 = round(oneadded/(len(otags)),4)
    add_compl2 = round((twoadded/len(otags)),4)
    return ([single_error,add_compl1, add_compl2])


tone = input("nontonal/tonal? -> ")
while tone!= "nontonal" and tone !="tonal":
    print("wrong input")
    tone = input("nontonal/tonal? -> ")
   
tag = input("POS/Affixes? -> ")
while tag!= "POS" and tag !="Affixes":
    print("wrong input")
    tag = input("POS/Affixes? -> ")

backoff = input("DefaultTagger as backoff? J/N-> ")
if backoff =="J":
    backoff = DefaultTagger('n')
else:
    backoff = None



last = input("Unigram or Bigram+Affix+Dictionary+Regexp+Default? UNIGRAM/BIGRAM-> ")
while last!= "UNIGRAM" and last !="BIGRAM":
    print("wrong input")
    last = input("Unigram or Bigram+Affix+Dictionary+Regexp+Default? UNIGRAM/BIGRAM-> ")



print("\nCalculating Brill & Wu Complementarity\n\nAfter the programme finished, please look for \
'Results\CompareBrillWu_"+last+tone+tag+".txt'  and 'Results\Disagreement_BrillWuHtml_"+last+tone+tag+".txt' in your current working directory.\n")



bambara = create_reader(tone, tag)          


if last == "BIGRAM":
    lasttagger = backoff_tagger([5,6,8,1],bambara,option_tones=tone, option_tag=tag)
    lasttaggertagged = lasttagger.tag_sents([untag(i) for i in bambara.test_sents])
    lasttagger_acc = lasttagger.evaluate(bambara.test_sents)
    lasttagger_err = round((1-(lasttagger_acc)),4)
if last == "UNIGRAM":
    lasttagger = indivUnigram(bambara, backoff)
    lasttaggertagged = lasttagger.tag_sents([untag(i) for i in bambara.test_sents])
    lasttagger_acc = lasttagger.evaluate(bambara.test_sents)
    lasttagger_err = round((1-(lasttagger_acc)),4)

hmm = indivHMM(bambara)
hmmtagged = hmm.tag_sents([untag(i) for i in bambara.test_sents])
hmm_acc = hmm.evaluate(bambara.test_sents)
hmm_err = round((1 - (hmm_acc)),4)

crf = indivCRF(bambara, tone, tag)
crftagged = crf.tag_sents([untag(i) for i in bambara.test_sents]) 
crf_acc = crf.evaluate(bambara.test_sents)
crf_err = round((1-(crf_acc)),4)

tnt = indivTnT(bambara, backoff)
tnttagged = tnt.tag_sents([untag(i) for i in bambara.test_sents])
tnt_acc = tnt.evaluate(bambara.test_sents)
tnt_err = round((1-(tnt_acc)),4)



taggers = [crf,tnt,hmm,lasttagger]
taggedSents = [crftagged,tnttagged,hmmtagged,lasttaggertagged]

taggernames = [" CRF \t"," TNT \t"," HMM \t"," "+last[:3]+" \t"]
comparedHeader = [["     \t CRF \t TNT \t HMM \t "+last[:3]+" \n"]]
compared = []
counter_orgtagger = 0
for i in taggers:
    comparedline=[taggernames[counter_orgtagger]]
    counter_cmptagger = 0 # counter for the tagger i to be compared with all the other taggers
    for j in taggers:
        cmp = compareTagger(i,j,bambara.test_sents, taggedSents[counter_orgtagger],taggedSents[counter_cmptagger])
        comparedline.append(cmp+"\t")
        counter_cmptagger+=1 #increment to get next tagger to compare with tagger i
    comparedline.append("\n")
    compared.append(comparedline)
    counter_orgtagger+=1


lasttaggertags = [i[1] for i in (sum(lasttaggertagged,[]))]
tnttags =[i[1] for i in (sum(tnttagged,[]))]
hmmtags = [i[1] for i in (sum(hmmtagged,[]))]
crftags = [i[1] for i in (sum(crftagged,[]))]

errors_alone = [crf_err, tnt_err, hmm_err, lasttagger_err]   
errors = errorrates(crftags, tnttags, hmmtags, lasttaggertags, crf_err, tnt_err)

disagree = errors[0]
disagree_wu = errors[1]
complement = errors[2]
complement1 = errors[3]
complement2 = errors[4]
complement3 = errors[5]


comparedHeader.append(sum(compared,[]))
file = codecs.open("Results\\CompareAB_BrillWu_"+last+tone+tag+".txt","w","utf-8")
file.writelines(sum(comparedHeader,[]))
file.close()

f = open("Results\\Disagreement_BrillWuHtml_"+last+tone+tag+".txt", "w")
header1 = ["disagreement error rate\n"]
first_line_dis = [24*" "," CRF   "," TNT   ", " HMM   "," "+last[:3]+"   ","\n"]
rates0 = [str(i)+"  " for i in errors_alone]
rates1 = [str(i)+"  " for i in disagree]
rates2 = [str(i)+"  " for i in disagree_wu]
er1 = ["  Overall Error Rate    "]+rates0+["\n"]
er2 = ["Error Rate Disagreement "]+rates1+["\n"]
er3 = ["Err Rate Disagree W"+last[0]+"    "]+rates2+["\n"]


rates3 = [str(i)+"  " for i in complement]
header2 = ["Complementarity additive?\n"]
first_line_add_compl = [25*" "," CRF   ","+TNT   ", "+HMM   ","+"+last[:3]+"   ","\n"]
additive_compl = ["% of times all are wrong "]+rates3+["\n\n"]

rates4 = [str(i)+"  " for i in complement1]
header3 = ["Complementarity additive?\n"]
first_line_add_compl1 = [25*" "," CRF   ","+TNT  ", "+"+last[:3]+"   ","\n"]
additive_compl1 = ["% of times all are wrong "]+rates4+["\n\n"]

rates5 = [str(i)+"  " for i in complement2]
header4 = ["Complementarity additive?\n"]
first_line_add_compl2 = [25*" "," CRF   ","+HMM  ", "+"+last[:3]+"   ","\n"]
additive_compl2 = ["% of times all are wrong "]+rates5+["\n\n"]

rates6 = [str(i)+"  " for i in complement3]
header5 = ["Complementarity additive?\n"]
first_line_add_compl3 = [25*" "," TNT   ","+HMM  ", "+"+last[:3]+"   ","\n"]
additive_compl3 = ["% of times all are wrong "]+rates6+["\n"]

finallines = header1+first_line_dis+er1+er2+er3+["\n\n\n"]+header2+first_line_add_compl+additive_compl+first_line_add_compl1+additive_compl1+first_line_add_compl2+additive_compl2+first_line_add_compl3+additive_compl3


f.writelines(finallines)
f.close()

