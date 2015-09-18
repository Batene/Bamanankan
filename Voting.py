# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

# Implementation of voting strategies Majority, TotPrecision,TagPrecision, PrecisionRecall
# (van Halteren et al. (2001): Improving Accuracy in Word Class Tagging through the Combination
# of Machine Learning Systems. In: Computational Linguistics 27 (2), S. 199–230.)

from nltk.metrics import accuracy
from create_reader import create_reader_9_1
from indivTaggers import *
from CrossValidation import CrossValidation
from collections import defaultdict
from nltk.tag import untag
from collections import OrderedDict
from collections import defaultdict
import pickle
import os
from os import listdir
import getpass


class Majority(object):
    def __init__(self, taggerlist, test_sents, crf=None, tnt=None, hmm=None, unigram=None, regex=None, bigram=None):
        self.taggerlist = taggerlist
        self.crf = crf
        self.tnt = tnt
        self.hmm=hmm
        self.unigram=unigram
        self.bigram=bigram
        self.regex=regex
        self.test_sents=test_sents

    def tag_sents(self, sents):
        '''tag_sents method for majority tagging'''
        tagged_sents_suggestions = []
        for i in self.taggerlist:
            if i == "unigram":
                uni_tagged_sents = self.unigram.tag_sents(sents)
                tagged_sents_suggestions.append(uni_tagged_sents)
            if i == "tnt":
                tnt_tagged_sents = self.tnt.tag_sents(sents)
                tagged_sents_suggestions.append(tnt_tagged_sents)
            if i == "hmm":
                hmm_tagged_sents = self.hmm.tag_sents(sents)
                tagged_sents_suggestions.append(hmm_tagged_sents)
            if i == "crf":
                crf_tagged_sents = self.crf.tag_sents(sents)
                tagged_sents_suggestions.append(crf_tagged_sents)
            if i == "regex":
                regex_tagged_sents = self.regex.tag_sents(sents)
                tagged_sents_suggestions.append(regex_tagged_sents)
            if i == "bigram":
                bigram_tagged_sents=self.bigram.tag_sents(sents)
                tagged_sents_suggestions.append(bigram_tagged_sents)

        tagged_sents = []
        for i in range(len(sents)):
            sent_suggestion=[]
            for j in range(len(self.taggerlist)):
                sent_suggestion.append(tagged_sents_suggestions[j][i])                       
            sent = sents[i]
            tagsuggestions = [[i[1] for i in j] for j in sent_suggestion] # tags of each Tagger for one sentence (sents[i])
            tagsuggestions_words = list(zip(*tagsuggestions))#list of tuples for each word in sent; tuple consist of a tag per tagger for the word
            final_tags = []
            for liste in tagsuggestions_words:
                groups= defaultdict(list)#defaultdict necessary, otherwise, need to check if a tag already exists in the dict, before adding 1 to its vote
                for obj in liste:
                    groups[obj].append(1)
                votes = [sum(i) for i in groups.values()]
                tags_votes = list(zip(groups.keys(), votes))
                right_tag = max(tags_votes, key=lambda x: x[1])
                final_tags.append(right_tag[0])
            tagged_sents.append(list(zip(sent,final_tags)))
        return tagged_sents        

    def evaluate(self, orgsents): 
        orgsents_words = sum(orgsents,[])
        sents_to_tag = [untag(i) for i in orgsents]
        self.tagger_taggedsents = self.tag_sents(sents_to_tag)
        tagger_tagged_words = sum(self.tagger_taggedsents,[])
        return accuracy(orgsents_words,tagger_tagged_words)

    def final(self):
        print("Accuracy combined of {0}: {1}\n".format(self.taggerlist, self.evaluate(self.test_sents)))


class TotPrecision(object):
    def __init__(self, taggerlist, test_sents, crf=None, tnt=None, hmm=None, unigram=None,regex=None,bigram=None,st=None,
                 crf_totprecision=None, tnt_totprecision=None, hmm_totprecision=None, unigram_totprecision=None, regex_totprecision=None, bigram_totprecision=None,
                 st_totprecision=None):
        self.taggerlist = taggerlist
        self.st = st
        self.crf = crf
        self.tnt = tnt
        self.hmm = hmm
        self.unigram = unigram
        self.regex=regex
        self.bigram=bigram
        self.st_totprecision=st_totprecision
        self.unigram_totprecision = unigram_totprecision
        self.tnt_totprecision = tnt_totprecision
        self.hmm_totprecision = hmm_totprecision
        self.crf_totprecision = crf_totprecision
        self.regex_totprecision=regex_totprecision
        self.bigram_totprecision=bigram_totprecision
        self.test_sents = test_sents



    def tag_sentsST(self,sents_to_tag):
        tagged_sents_suggestions = []
        for i in self.taggerlist:
            if i == "st":
                    sttag0 = self.st.tag_sents(sents_to_tag)
                    sttag = [(i[0][1],i[1][1], self.st_totprecision) for i in sttag0 if len(i)>1]####
                    st_tags_precs = sttag
                    tagged_sents_suggestions.append(st_tags_precs)
                    print("hello")
                    print(tagged_sents_suggestions[0][0])
                    print(len(tagged_sents_suggestions))

            if i == "crf":
                    crftag = self.crf.tag_sents(sents_to_tag)
                    crf_tags_precs = [[(i[0],i[1], self.crf_totprecision) for i in j] for j in crftag]
                    tagged_sents_suggestions.append(sum(crf_tags_precs,[]))
                    print(tagged_sents_suggestions[0][0])
                    print(len(tagged_sents_suggestions))
            if i == "unigram":
                    unitag = self.unigram.tag_sents(sents_to_tag)
                    uni_tags_precs = [[(i[0],i[1], self.unigram_totprecision) for i in j] for j in unitag]
                    tagged_sents_suggestions.append(sum(uni_tags_precs,[]))
                    print(tagged_sents_suggestions[0][0])
                    print(len(tagged_sents_suggestions))

        tagged_sents = []
        for i in range(len(tagged_sents_suggestions)):
            sent_suggestion=[]
            for j in range(len(self.taggerlist)):
                sent_suggestion.append(tagged_sents_suggestions[j])                       
            sent = sum(sents_to_tag,[])
            tagsuggestions = [[(i[1],i[2]) for i in j]for j in sent_suggestion] # tags + totprecision of each tagger for one sentence (sents[i])
            tagsuggestions_words = list(zip(*tagsuggestions))#list of tuples for each word in sent; tuple consist of a tag + totprecision per tagger for the word
            final_tags = []
            for liste in tagsuggestions_words:
                groups= defaultdict(list)#defaultdict necessary, otherwise, need to check if a tag already exists in the dict, before adding 1 to its vote
                for obj in liste:
                    groups[obj[0]].append(obj[1])
                votes = [sum(i) for i in groups.values()]
                tags_votes = list(zip(groups.keys(), votes))
                right_tag = max(tags_votes, key=lambda x: x[1])
                final_tags.append(right_tag[0])
            tagged_sents.append(list(zip(sent,final_tags)))
        return tagged_sents 
        
        
    def tag_sents(self, sents_to_tag):
        '''each tagger votes for its tag with its totprecision'''
        tagged_sents_suggestions = []
        for i in self.taggerlist:
            if i == "crf":
                crftag = self.crf.tag_sents(sents_to_tag)
                crf_tags_precs = [[(i[0],i[1], self.crf_totprecision) for i in j] for j in crftag]
                tagged_sents_suggestions.append(crf_tags_precs)
            if i == "tnt":
                tnttag = self.tnt.tag_sents(sents_to_tag)
                tnt_tags_precs = [[(i[0],i[1], self.tnt_totprecision) for i in j] for j in tnttag]
                tagged_sents_suggestions.append(tnt_tags_precs)
            if i == "hmm":
                hmmtag = self.hmm.tag_sents(sents_to_tag)
                hmm_tags_precs = [[(i[0],i[1], self.hmm_totprecision) for i in j] for j in hmmtag]
                tagged_sents_suggestions.append(hmm_tags_precs)
            if i == "unigram":
                utag = self.unigram.tag_sents(sents_to_tag)
                unigram_tags_precs = [[(i[0], i[1], self.unigram_totprecision) for i in j] for j in utag]
                tagged_sents_suggestions.append(unigram_tags_precs)
            if i == "regex":
                regextag = self.regex.tag_sents(sents_to_tag)
                regex_tags_precs = [[(i[0], i[1], self.regex_totprecision) for i in j] for j in regextag]
                tagged_sents_suggestions.append(regex_tags_precs)
            if i == "bigram":
                bigramtag = self.bigram.tag_sents(sents_to_tag)
                bigram_tags_precs = [[(i[0], i[1], self.bigram_totprecision) for i in j] for j in bigramtag]
                tagged_sents_suggestions.append(bigram_tags_precs)

        tagged_sents = []
        for i in range(len(sents_to_tag)):
            sent_suggestion=[]
            for j in range(len(self.taggerlist)):
                sent_suggestion.append(tagged_sents_suggestions[j][i])                       
            sent = sents_to_tag[i]
            tagsuggestions = [[(i[1],i[2]) for i in j] for j in sent_suggestion] # tags + totprecision of each tagger for one sentence (sents[i])
            tagsuggestions_words = list(zip(*tagsuggestions))#list of tuples for each word in sent; tuple consist of a tag + totprecision per tagger for the word
            final_tags = []
            for liste in tagsuggestions_words:
                groups= defaultdict(list)#defaultdict necessary, otherwise, need to check if a tag already exists in the dict, before adding 1 to its vote
                for obj in liste:
                    groups[obj[0]].append(obj[1])
                votes = [sum(i) for i in groups.values()]
                tags_votes = list(zip(groups.keys(), votes))
                right_tag = max(tags_votes, key=lambda x: x[1])
                final_tags.append(right_tag[0])
            tagged_sents.append(list(zip(sent,final_tags)))
        return tagged_sents 
        
    def evaluate(self, orgsents):
        orgsents_words = sum(orgsents,[])
        sents_to_tag = [untag(i) for i in orgsents]
        self.tagger_taggedsents = self.tag_sents(sents_to_tag)
        tagger_tagged_words = sum(self.tagger_taggedsents,[])
        return accuracy(orgsents_words, tagger_tagged_words)

    def evaluateST(self, orgsents):
        orgsents_words = sum(orgsents,[])
        sents_to_tag = [untag(i) for i in orgsents]
        self.tagger_taggedsents = self.tag_sentsST(sents_to_tag)
        tagger_tagged_words = self.tagger_taggedsents[0]
        print(len(tagger_tagged_words[0]))
        print(len(orgsents_words))
        return accuracy(orgsents_words, tagger_tagged_words)

    def final(self):
        print("Accuracy combined of {0}: {1}\n".format(self.taggerlist, self.evaluate(self.test_sents)))

    def finalST(self):
        print("Accuracy combined of {0}: {1}\n".format(self.taggerlist, self.evaluateST(self.test_sents)))


class TagPrecision(object):
    def __init__(self, taggerlist, test_sents, crf=None, tnt=None, hmm=None,unigram=None,regex=None,bigram=None,
                 crf_tagprecision=None, tnt_tagprecision=None,hmm_tagprecision=None,unigram_tagprecision=None,regex_tagprecision=None,bigram_tagprecision=None ):
        self.crf = crf
        self.tnt=tnt
        self.hmm=hmm
        self.unigram = unigram
        self.regex=regex
        self.bigram=bigram
        self.regex_tagprecision=regex_tagprecision
        self.unigram_tagprecision = unigram_tagprecision
        self.tnt_tagprecision =  tnt_tagprecision
        self.crf_tagprecision = crf_tagprecision
        self.hmm_tagprecision = hmm_tagprecision
        self.bigram_tagprecision=bigram_tagprecision
        self.test_sents=test_sents
        self.taggerlist=taggerlist


    def add_prec(self, taggerprec, taggedsent):
        '''adds to each tag the tagger´s precision of this tag;
        returns list (a sentence) of tuples: (word, tag, tagprecision)'''
        lookup = dict(taggerprec)
        prec_added = [] #returns list of triples (word, tag, precision)
        for i in taggedsent:
            if i[1] not in lookup.keys():
                prec_added.append((i[0], i[1], 0.0))
            else:
                prec_added.append((i[0], i[1],lookup[i[1]]))
        return prec_added

    def tag_sents(self, sents_to_tag):
        '''tag_sents function for TagPrecision;
        chooses the tag according to tagprecision values of the taggers'''
        tagged_sents_suggestions = []
        for i in self.taggerlist:
            if i == "crf":
                crf_tested = self.crf.tag_sents(sents_to_tag)
                crf_prec_added = [self.add_prec(self.crf_tagprecision,i) for i in crf_tested]
                tagged_sents_suggestions.append(crf_prec_added)
            if i == "tnt":
                tnt_tested = self.tnt.tag_sents(sents_to_tag)
                tnt_prec_added = [self.add_prec(self.tnt_tagprecision,i) for i in tnt_tested]
                tagged_sents_suggestions.append(tnt_prec_added)
            if i == "hmm":
                hmm_tested = self.hmm.tag_sents(sents_to_tag)
                hmm_tested_tags = [i[1] for i in sum(hmm_tested,[])]
                hmm_prec_added = [self.add_prec(self.hmm_tagprecision,i) for i in hmm_tested]
                tagged_sents_suggestions.append(hmm_prec_added)
            if i == "unigram":
                unigram_tested = self.unigram.tag_sents(sents_to_tag)
                unigram_prec_added = [self.add_prec(self.unigram_tagprecision, i) for i in unigram_tested]
                tagged_sents_suggestions.append(unigram_prec_added)
            if i == "regex":
                regex_tested=self.regex.tag_sents(sents_to_tag)
                regex_prec_added = [self.add_prec(self.regex_tagprecision, i) for i in regex_tested]#self.add_prec(self.regex_tagprecision, regex_tested_tags)
                tagged_sents_suggestions.append(regex_prec_added)
            if i == "bigram":
                bigram_tested = self.bigram.tag_sents(sents_to_tag)
                bigram_prec_added = [self.add_prec(self.bigram_tagprecision, i) for i in bigram_tested]
                tagged_sents_suggestions.append(bigram_prec_added)

        tagged_sents = []
        for i in range(len(sents_to_tag)):
            sent_suggestion=[]
            for j in range(len(self.taggerlist)):
                sent_suggestion.append(tagged_sents_suggestions[j][i])                       
            sent = sents_to_tag[i]
            tagsuggestions = [[(i[1],i[2]) for i in j] for j in sent_suggestion] # tags + tagprecision of each tagger for one sentence (sents[i])
            tagsuggestions_words = list(zip(*tagsuggestions))#list of tuples for each word in sent; tuple consist of a tag + tagprecision per tagger for the word
            final_tags = []
            for liste in tagsuggestions_words:
                groups= OrderedDict(defaultdict(list))#defaultdict necessary, otherwise, need to check if a tag already exists in the dict, before adding 1 to its vote
                for obj in liste:
                    if obj[0] not in groups:
                        groups[obj[0]] = [obj[1]]
                    else:
                        groups[obj[0]].append(obj[1])
                votes = [sum(i) for i in groups.values()]
                tags_votes = list(zip(groups.keys(), votes))
                right_tag = max(tags_votes, key=lambda x: x[1])
                final_tags.append(right_tag[0])
            tagged_sents.append(list(zip(sent,final_tags)))
        return tagged_sents

    def evaluate(self, orgsents):
        orgwords = sum(orgsents,[])
        tagger_taggedsents = self.tag_sents([untag(i) for i in orgsents])
        tagger_taggedwords = sum(tagger_taggedsents,[])
        return accuracy(orgwords, tagger_taggedwords)

    def final(self):
        print("Accuracy combined of {0}: {1}\n".format(self.taggerlist, self.evaluate(self.test_sents)))


class PrecisionRecall(object): 
    def __init__(self, taggerlist, test_sents,
                 crf=None, tnt=None, hmm=None, unigram=None, regex=None,bigram=None,
                 crf_tagprecision=None, tnt_tagprecision=None, hmm_tagprecision=None, unigram_tagprecision=None, regex_tagprecision=None,bigram_tagprecision=None,
                 crf_tagrecall=None, tnt_tagrecall=None, hmm_tagrecall=None, unigram_tagrecall=None,regex_tagrecall=None,bigram_tagrecall=None):
        self.taggerlist=taggerlist
        self.test_sents = test_sents
        self.tagger_taggedsents = []
        self.crf = crf
        self.tnt = tnt
        self.hmm = hmm
        self.unigram = unigram
        self.bigram=bigram
        self.regex=regex
        self.regex_tagprecision=regex_tagprecision
        self.unigram_tagprecision = unigram_tagprecision
        self.bigram_tagprecision=bigram_tagprecision
        self.tnt_tagprecision =  tnt_tagprecision
        self.crf_tagprecision = crf_tagprecision
        self.hmm_tagprecision = hmm_tagprecision
        self.crf_tagrecall = crf_tagrecall
        self.tnt_tagrecall = tnt_tagrecall
        self.hmm_tagrecall = hmm_tagrecall
        self.unigram_tagrecall = unigram_tagrecall
        self.regex_tagrecall=regex_tagrecall
        self.bigram_tagrecall=bigram_tagrecall

    
    def tag_sents(self,sents_to_tag):
        
        tagged_sents_suggestions = []
        tagrecall = []
        tagprecision=[]
        for i in self.taggerlist:
            if i == "crf":
                crf_tested0 = self.crf.tag_sents(sents_to_tag)
                crf_tested = [[(i[0], str(i[1]).replace("None", "None")) for i in j] for j in crf_tested0]
                tagged_sents_suggestions.append(crf_tested)
                tagrecall.append(self.crf_tagrecall)
                tagprecision.append(self.crf_tagprecision)
            if i == "tnt":
                tnt_tested0 = self.tnt.tag_sents(sents_to_tag)
                tnt_tested = [[(i[0], (i[1]).replace("Unk", "None")) for i in j] for j in tnt_tested0]
                tagged_sents_suggestions.append(tnt_tested)
                tagrecall.append(self.tnt_tagrecall)
                tagprecision.append(self.tnt_tagprecision)

            if i == "hmm":
                hmm_tested0 = self.hmm.tag_sents(sents_to_tag)
                hmm_tested = [[(i[0], (i[1]).replace("None", "None")) for i in j] for j in hmm_tested0]
                tagged_sents_suggestions.append(hmm_tested)
                tagrecall.append(self.hmm_tagrecall)
                tagprecision.append(self.hmm_tagprecision)

            if i == "unigram":
                unigram_tested0 = self.unigram.tag_sents(sents_to_tag)
                unigram_tested = [[(i[0], (i[1]).replace("None", "None")) for i in j] for j in unigram_tested0]
                tagged_sents_suggestions.append(unigram_tested)
                tagrecall.append(self.unigram_tagrecall)
                tagprecision.append(self.unigram_tagprecision)

            if i == "regex":
                regex_tested0=self.regex.tag_sents(sents_to_tag)
                regex_tested = [[(i[0], (i[1]).replace("None", "None")) for i in j] for j in regex_tested0]
                tagged_sents_suggestions.append(regex_tested)
                tagrecall.append(self.regex_tagrecall)
                tagprecision.append(self.regex_tagprecision)

            if i == "bigram":
                bigram_tested0 = self.bigram.tag_sents(sents_to_tag)
                bigram_tested = [[(i[0], (i[1]).replace("None", "None")) for i in j] for j in bigram_tested0]
                tagged_sents_suggestions.append(bigram_tested)
                tagrecall.append(self.bigram_tagrecall)
                tagprecision.append(self.bigram_tagprecision)

                

        tagged_sents = []
        for i in range(len(sents_to_tag)):
            sent_suggestion=[]
            for j in range(len(self.taggerlist)):
                sent_suggestion.append(tagged_sents_suggestions[j][i])                       
            sent = sents_to_tag[i]
            tagsuggestions = [[i[1] for i in j] for j in sent_suggestion] # tags of each tagger for one sentence (sents[i])
            tagsuggestions_words = list(zip(*tagsuggestions))#list of tuples for each word in sent; tuple consist of a tag per tagger for the word
            final_tags = []
            sent = sents_to_tag[i]
            for ts in tagsuggestions_words:
                different_tags = set(ts)
                weights = dict()
                for tag in different_tags:
                    summe = 0
                    counter = 0
                    for i in ts: # i = tagsuggestion of a tagger
                        if i == tag:
                            summe+=tagprecision[counter][tag]#if tagger suggested tag, then add tagprecision of tagger
                        else:
                            if tag not in tagrecall[counter]:
                                print(tagsuggestions_words)
                            else:
                                summe+=tagrecall[counter][tag]#add tagrecall of tagger
                        counter+=1
                    weights[tag]= summe        
                right_tag = (max(weights, key=weights.get))
                final_tags.append(right_tag)
            tagged_sents.append(list(zip(sent, final_tags)))
        return tagged_sents  

    def evaluate(self, orgsents):
        orgwords = sum(orgsents,[])                               
        self.tagger_taggedsents = self.tag_sents([untag (i) for i in orgsents])
        tagged_words = sum(self.tagger_taggedsents,[])
        return accuracy(orgwords, tagged_words)
    
    def final(self):
        print("Accuracy combined of {0}: {1}\n".format(self.taggerlist, self.evaluate(self.test_sents)))




#### Begin Voting #####

tone = input("nontonal/tonal? -> ")
while tone!= "nontonal" and tone !="tonal":
    print("wrong input")
    tone = input("nontonal/tonal? -> ")
   
tag = input("POS/Affixes? -> ")
while tag!= "POS" and tag !="Affixes":
    print("wrong input")
    tag = input("POS/Affixes? -> ")       

#if choosing bigram, uncomment corresponding region below!!
last = input("Unigram or Bigram+Affix+Dict+Regexp+Default as 4th Tagger? U/B -> ") 
while last != "U" and last != "B":
    print("wrong input")
    last = input("U/B -> ") 

bambara = create_reader_9_1(tone,tag)
    
crf = indivCRF(bambara, tone, tag)
tnt = indivTnT(bambara, backoff=DefaultTagger('n'))
hmm = indivHMM(bambara)
unigram = indivUnigram(bambara, backoff=DefaultTagger('n'))
regex = indivRegexp(bambara, tag, tone, backoff=DefaultTagger('n'))
dic = dictionary_backoff(tone, regex)
affix=indivAffix(bambara, affix_length=-4, backoff = dic) 
bigram = indivBigram(bambara, backoff=affix)


print("\n\nCalculating CrossValidation...")
cV = CrossValidation(9, bambara.train_sents, tone, tag)
cV.trainALL(last)
cV.trainRegexp(DefaultTagger('n'))
crf_avg_acc = cV.crf_avg_acc
tnt_avg_acc = cV.tnt_avg_acc
hmm_avg_acc = cV.hmm_avg_acc
uni_avg_acc = cV.lasttagger_avg_acc
bigram_avg_acc = cV.lasttagger_avg_acc
regex_avg_acc = cV.regex_avg_acc

crf_tagprecision = cV.crf_tagprecision
tnt_tagprecision = cV.tnt_tagprecision
hmm_tagprecision = cV.hmm_tagprecision
unigram_tagprecision = cV.lasttagger_tagprecision
bigram_tagprecision = cV.lasttagger_tagprecision
regex_tagprecision = cV.regex_tagprecision

crf_tagrecall = cV.crf_tagrecall
tnt_tagrecall = cV.tnt_tagrecall
hmm_tagrecall= cV.hmm_tagrecall
unigram_tagrecall= cV.lasttagger_tagrecall
bigram_tagrecall = cV.lasttagger_tagrecall
regex_tagrecall = cV.regex_tagrecall


print("\n\nCalculating Majority...\n")

m = Majority(["crf", "tnt", "hmm", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram)
m.final()
m = Majority(["crf", "tnt", "hmm"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm)
m.final()
m = Majority(["crf", "tnt", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram)
m.final()
m = Majority(["crf",  "hmm", "unigram"], bambara.test_sents, crf=crf,  hmm=hmm, unigram=unigram)
m.final()

m = Majority([ "tnt", "hmm", "unigram"], bambara.test_sents, tnt=tnt, hmm=hmm, unigram=unigram)
m.final()

print("\n\nCalculating TotPrecision...\n")
        
tp = TotPrecision(["crf", "tnt", "hmm", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,
                 crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, unigram_totprecision=uni_avg_acc)
tp.final()

tp0 = TotPrecision(["crf", "tnt", "hmm"],bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm,
                  crf_totprecision=crf_avg_acc, tnt_totprecision = tnt_avg_acc, hmm_totprecision=hmm_avg_acc)
tp0.final()

tp3 = TotPrecision(["crf", "tnt", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram,
                  crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, unigram_totprecision=uni_avg_acc)
tp3.final()

tp1 = TotPrecision(["crf", "hmm", "unigram"], bambara.test_sents, crf=crf, hmm=hmm, unigram=unigram,
                  crf_totprecision=crf_avg_acc, hmm_totprecision=hmm_avg_acc, unigram_totprecision = uni_avg_acc)
tp1.final()
tp2 = TotPrecision(["tnt", "hmm","unigram"], bambara.test_sents, tnt=tnt, hmm=hmm, unigram=unigram,
                  tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, unigram_totprecision=uni_avg_acc)
tp2.final()


print("\n\nCalculating TagPrecision...\n")
    

p0 = TagPrecision(["crf", "tnt", "hmm", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,
                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision)
p0.final()

    
p1 = TagPrecision(["crf", "tnt", "hmm"],bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm,
                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision)
p1.final()


p2 = TagPrecision(["crf", "tnt", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram,
                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, unigram_tagprecision = unigram_tagprecision)
p2.final()


p3 = TagPrecision(["crf", "hmm", "unigram"], bambara.test_sents, crf=crf, hmm = hmm, unigram=unigram,
                  crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision)
p3.final()


p4 = TagPrecision(["tnt", "hmm","unigram"],  bambara.test_sents, tnt=tnt , hmm=hmm, unigram=unigram,
                  tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision)
p4.final()

p5 = TagPrecision(["crf", "tnt"], bambara.test_sents, crf=crf, tnt=tnt,
                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision)
p5.final()

p6 = TagPrecision(["crf", "hmm"], bambara.test_sents, crf=crf, hmm=hmm,
                  crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision)
p6.final()

p7 = TagPrecision(["crf", "unigram"],  bambara.test_sents, crf=crf, unigram=unigram,
                  crf_tagprecision=crf_tagprecision, unigram_tagprecision=unigram_tagprecision)
p7.final()

p8 = TagPrecision(["tnt", "hmm"], bambara.test_sents, tnt=tnt, hmm=hmm,
                  tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision)
p8.final()

p9 = TagPrecision(["tnt", "unigram"],  bambara.test_sents, tnt= tnt, unigram=unigram,
                  tnt_tagprecision=tnt_tagprecision, unigram_tagprecision=unigram_tagprecision)
p9.final()

p10 = TagPrecision(["hmm", "unigram"],  bambara.test_sents, hmm=hmm, unigram=unigram,
                   hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision)
p10.final()


print("\n\nCalculating PrecisionRecall...\n")


pr = PrecisionRecall(["crf", "tnt", "hmm", "unigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,unigram_tagprecision=unigram_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall,
                     unigram_tagrecall=unigram_tagrecall)
pr.final()

pr0 = PrecisionRecall(["crf", "tnt", "hmm"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall)
pr0.final()

pr1 = PrecisionRecall(["crf", "tnt", "unigram"],bambara.test_sents, crf=crf, tnt=tnt,unigram=unigram,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, unigram_tagprecision=unigram_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, unigram_tagrecall=unigram_tagrecall)
pr1.final()

pr4 = PrecisionRecall(["crf", "hmm", "unigram"], bambara.test_sents, crf=crf, hmm=hmm, unigram=unigram,
                     crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision,
                     crf_tagrecall=crf_tagrecall, hmm_tagrecall=hmm_tagrecall, unigram_tagrecall=unigram_tagrecall)
pr4.final()

pr5 = PrecisionRecall(["tnt","hmm", "unigram"], bambara.test_sents, tnt=tnt, hmm=hmm, unigram=unigram,
                    tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision,
                    tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall, unigram_tagrecall=unigram_tagrecall)
pr5.final()

pr6 = PrecisionRecall(["crf", "tnt"], bambara.test_sents, crf=crf, tnt=tnt,
                      crf_tagprecision = crf_tagprecision, tnt_tagprecision=tnt_tagprecision,
                      crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall)
pr6.final()

pr8 = PrecisionRecall(["crf", "hmm"], bambara.test_sents, crf=crf, hmm=hmm,
                     crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision,
                     crf_tagrecall=crf_tagrecall, hmm_tagrecall=hmm_tagrecall)
pr8.final()

pr7 = PrecisionRecall(["crf", "unigram"], bambara.test_sents, crf=crf, unigram=unigram,
                     crf_tagprecision=crf_tagprecision, unigram_tagprecision=unigram_tagprecision,
                     crf_tagrecall=crf_tagrecall, unigram_tagrecall = unigram_tagrecall)
pr7.final()

pr9 = PrecisionRecall(["tnt", "hmm"], bambara.test_sents, tnt=tnt, hmm=hmm,
                     tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,
                     tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall)
pr9.final()

pr10 = PrecisionRecall(["tnt", "unigram"], bambara.test_sents, tnt=tnt, unigram=unigram,
                    tnt_tagprecision = tnt_tagprecision, unigram_tagprecision=unigram_tagprecision,
                      tnt_tagrecall = tnt_tagrecall, unigram_tagrecall = unigram_tagrecall)
pr10.final()

pr11 = PrecisionRecall(["hmm", "unigram"], bambara.test_sents, hmm=hmm, unigram=unigram,
                      hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision,
                      hmm_tagrecall=hmm_tagrecall, unigram_tagrecall=unigram_tagrecall)
pr11.final()

#With Bigram

##print("\nWith Bigram")
##print("\n\nCalculating Majority...\n")
##
##m = Majority(["crf", "tnt", "hmm", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, bigram=bigram)
##m.final()
##m = Majority(["crf", "tnt", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, bigram=bigram)
##m.final()
##m = Majority(["crf",  "hmm", "bigram"], bambara.test_sents, crf=crf,  hmm=hmm, bigram=bigram)
##m.final()
##
##m = Majority([ "tnt", "hmm", "bigram"], bambara.test_sents, tnt=tnt, hmm=hmm, bigram=bigram)
##m.final()
##
##print("\n\nCalculating TotPrecision...\n")
##        
##tp = TotPrecision(["crf", "tnt", "hmm", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, bigram=bigram,
##                 crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, bigram_totprecision=bigram_avg_acc)
##tp.final()
##
##tp3 = TotPrecision(["crf", "tnt", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, bigram=bigram,
##                  crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, bigram_totprecision=bigram_avg_acc)
##tp3.final()
##
##tp1 = TotPrecision(["crf", "hmm", "bigram"], bambara.test_sents, crf=crf, hmm=hmm, bigram=bigram,
##                  crf_totprecision=crf_avg_acc, hmm_totprecision=hmm_avg_acc, bigram_totprecision = bigram_avg_acc)
##tp1.final()
##tp2 = TotPrecision(["tnt", "hmm","bigram"], bambara.test_sents, tnt=tnt, hmm=hmm, bigram=bigram,
##                  tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, bigram_totprecision=bigram_avg_acc)
##tp2.final()
##
##
##print("\n\nCalculating TagPrecision...\n")
##    
##
##p0 = TagPrecision(["crf", "tnt", "hmm", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, bigram=bigram,
##                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision)
##p0.final()
##
##p2 = TagPrecision(["crf", "tnt", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, bigram=bigram,
##                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, bigram_tagprecision = bigram_tagprecision)
##p2.final()
##
##p3 = TagPrecision(["crf", "hmm", "bigram"], bambara.test_sents, crf=crf, hmm = hmm, bigram=bigram,
##                  crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision)
##p3.final()
##
##
##p4 = TagPrecision(["tnt", "hmm","bigram"],  bambara.test_sents, tnt=tnt , hmm=hmm, bigram=bigram,
##                  tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision)
##p4.final()
##
##
##p7 = TagPrecision(["crf", "bigram"],  bambara.test_sents, crf=crf, bigram=bigram,
##                  crf_tagprecision=crf_tagprecision, bigram_tagprecision=bigram_tagprecision)
##p7.final()
##
##p9 = TagPrecision(["tnt", "bigram"],  bambara.test_sents, tnt= tnt, bigram=bigram,
##                  tnt_tagprecision=tnt_tagprecision, bigram_tagprecision=bigram_tagprecision)
##p9.final()
##
##p10 = TagPrecision(["hmm", "bigram"],  bambara.test_sents, hmm=hmm, bigram=bigram,
##                   hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision)
##p10.final()
##
##
##print("\n\nCalculating PrecisionRecall...\n")
##
##
##pr = PrecisionRecall(["crf", "tnt", "hmm", "bigram"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, bigram=bigram,
##                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,bigram_tagprecision=bigram_tagprecision,
##                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall,
##                     bigram_tagrecall=bigram_tagrecall)
##pr.final()
##
##pr1 = PrecisionRecall(["crf", "tnt", "bigram"],bambara.test_sents, crf=crf, tnt=tnt,bigram=bigram,
##                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, bigram_tagprecision=bigram_tagprecision,
##                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, bigram_tagrecall=bigram_tagrecall)
##pr1.final()
##
##pr4 = PrecisionRecall(["crf", "hmm", "bigram"], bambara.test_sents, crf=crf, hmm=hmm, bigram=bigram,
##                     crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision,
##                     crf_tagrecall=crf_tagrecall, hmm_tagrecall=hmm_tagrecall, bigram_tagrecall=bigram_tagrecall)
##pr4.final()
##
##pr5 = PrecisionRecall(["tnt","hmm", "bigram"], bambara.test_sents, tnt=tnt, hmm=hmm, bigram=bigram,
##                    tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision,
##                    tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall, bigram_tagrecall=bigram_tagrecall)
##pr5.final()
##
##pr7 = PrecisionRecall(["crf", "bigram"], bambara.test_sents, crf=crf, bigram=bigram,
##                     crf_tagprecision=crf_tagprecision, bigram_tagprecision=bigram_tagprecision,
##                     crf_tagrecall=crf_tagrecall, bigram_tagrecall = bigram_tagrecall)
##pr7.final()
##
##pr10 = PrecisionRecall(["tnt", "bigram"], bambara.test_sents, tnt=tnt, bigram=bigram,
##                    tnt_tagprecision = tnt_tagprecision, bigram_tagprecision=bigram_tagprecision,
##                      tnt_tagrecall = tnt_tagrecall, bigram_tagrecall = bigram_tagrecall)
##pr10.final()
##
##pr11 = PrecisionRecall(["hmm", "bigram"], bambara.test_sents, hmm=hmm, bigram=bigram,
##                      hmm_tagprecision=hmm_tagprecision, bigram_tagprecision=bigram_tagprecision,
##                      hmm_tagrecall=hmm_tagrecall, bigram_tagrecall=bigram_tagrecall)
##pr11.final()
##  
##


#with Regexp

print("Calculating Majority, TotPrecision, TagPrecision, PrecisionRecall with RegexTagger\n\n")

print("Majority with Regexp\n")
tp = Majority(["crf", "tnt", "hmm", "unigram", "regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,regex=regex)
tp.final()

tp = Majority(["crf", "tnt", "hmm", "regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, regex=regex)
tp.final()

tp = Majority(["crf", "tnt", "unigram", "regex"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram, regex=regex)
tp.final()

tp = Majority(["crf", "hmm", "unigram", "regex"], bambara.test_sents, crf=crf, hmm=hmm, unigram=unigram, regex=regex)
tp.final()

tp = Majority(["tnt", "hmm", "unigram", "regex"], bambara.test_sents,tnt=tnt,hmm=hmm, unigram=unigram, regex=regex)
tp.final()

tp = Majority(["crf", "tnt",  "regex"], bambara.test_sents, crf=crf, tnt=tnt, regex=regex)
tp.final()

tp = Majority(["crf", "hmm", "regex"], bambara.test_sents, crf=crf, hmm=hmm,  regex=regex)
tp.final()

tp = Majority(["crf", "unigram", "regex"], bambara.test_sents, crf=crf, unigram=unigram, regex=regex)
tp.final()

tp = Majority(["tnt", "hmm", "regex"], bambara.test_sents,tnt=tnt,hmm=hmm, regex=regex)
tp.final()

tp = Majority(["tnt", "unigram", "regex"], bambara.test_sents,tnt=tnt, unigram=unigram, regex=regex)
tp.final()

tp = Majority(["hmm", "unigram", "regex"], bambara.test_sents,hmm=hmm, unigram=unigram, regex=regex)
tp.final()


print("\n\nTotPrecision with Regexp\n")
tp = TotPrecision(["crf", "tnt", "hmm", "unigram", "regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,regex=regex,
                 crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, unigram_totprecision=uni_avg_acc,
                  regex_totprecision= regex_avg_acc)
tp.final()

tp = TotPrecision(["crf", "tnt", "hmm", "regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, regex=regex,
                 crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["crf", "tnt", "unigram", "regex"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram, regex=regex,
                 crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc,unigram_totprecision=uni_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["crf", "hmm", "unigram", "regex"], bambara.test_sents, crf=crf, hmm=hmm, unigram=unigram, regex=regex,
                 crf_totprecision=crf_avg_acc, hmm_totprecision=hmm_avg_acc,unigram_totprecision=uni_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["tnt", "hmm", "unigram", "regex"], bambara.test_sents,tnt=tnt,hmm=hmm, unigram=unigram, regex=regex,
                 tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, unigram_totprecision=uni_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["crf", "tnt",  "regex"], bambara.test_sents, crf=crf, tnt=tnt, regex=regex,
                 crf_totprecision=crf_avg_acc, tnt_totprecision=tnt_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["crf", "hmm", "regex"], bambara.test_sents, crf=crf, hmm=hmm,  regex=regex,
                 crf_totprecision=crf_avg_acc, hmm_totprecision=hmm_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["crf", "unigram", "regex"], bambara.test_sents, crf=crf, unigram=unigram, regex=regex,
                 crf_totprecision=crf_avg_acc, unigram_totprecision=uni_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["tnt", "hmm", "regex"], bambara.test_sents,tnt=tnt,hmm=hmm, regex=regex,
                 tnt_totprecision=tnt_avg_acc, hmm_totprecision=hmm_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["tnt", "unigram", "regex"], bambara.test_sents,tnt=tnt, unigram=unigram, regex=regex,
                 tnt_totprecision=tnt_avg_acc, unigram_totprecision=uni_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

tp = TotPrecision(["hmm", "unigram", "regex"], bambara.test_sents,hmm=hmm, unigram=unigram, regex=regex,
                 hmm_totprecision=hmm_avg_acc, unigram_totprecision=uni_avg_acc, regex_totprecision=regex_avg_acc)
tp.final()

print("\n\nTagPrecision with Regexp\n")

pr = TagPrecision(["crf", "tnt", "hmm", "unigram","regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram, regex=regex,
                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,
                  unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "tnt", "hmm", "regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "tnt", "unigram", "regex"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram, regex=regex,
                  crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, unigram_tagprecision=unigram_tagprecision,
                  regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "hmm", "unigram", "regex"], bambara.test_sents, crf=crf, hmm=hmm, unigram=unigram, regex=regex,
                  crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision,
                  regex_tagprecision=regex_tagprecision)                  
pr.final()

pr = TagPrecision(["tnt", "hmm", "unigram", "regex"], bambara.test_sents, tnt=tnt, hmm=hmm, unigram=unigram, regex=regex,
                  tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision,
                  regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "tnt", "regex"], bambara.test_sents, crf=crf, tnt=tnt, regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "tnt", "regex"], bambara.test_sents, crf=crf, tnt=tnt, regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "hmm", "regex"], bambara.test_sents, crf=crf, hmm=hmm, regex=regex,
                     crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "unigram", "regex"], bambara.test_sents, crf=crf, unigram=unigram, regex=regex,
                     crf_tagprecision=crf_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["tnt","hmm" ,"regex"], bambara.test_sents, tnt=tnt, hmm=hmm, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,
                  regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["tnt","unigram" ,"regex"], bambara.test_sents, tnt=tnt, unigram=unigram, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["hmm","unigram", "regex"], bambara.test_sents, hmm=hmm, unigram=unigram, regex=regex,
                     hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["crf", "regex"], bambara.test_sents, crf=crf, regex=regex,
                     crf_tagprecision=crf_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["tnt", "regex"], bambara.test_sents, tnt=tnt, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()

pr = TagPrecision(["hmm", "regex"], bambara.test_sents, hmm=hmm, regex=regex,
                     hmm_tagprecision=hmm_tagprecision, regex_tagprecision=regex_tagprecision)
pr.final()


print("\n\nPrecisionRecall with Regexp\n")

pr = PrecisionRecall(["crf", "tnt", "hmm", "unigram","regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram, regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,unigram_tagprecision=unigram_tagprecision,
                     regex_tagprecision=regex_tagprecision,crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall,
                     unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "tnt", "hmm", "regex"], bambara.test_sents, crf=crf, tnt=tnt, hmm=hmm, unigram=unigram,regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision,regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall,
                     regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "tnt", "unigram", "regex"], bambara.test_sents, crf=crf, tnt=tnt, unigram=unigram, regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall,unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "hmm", "unigram", "regex"], bambara.test_sents, crf=crf, hmm=hmm, unigram=unigram, regex=regex,
                     crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, hmm_tagrecall=hmm_tagrecall,unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["tnt", "hmm", "unigram", "regex"], bambara.test_sents, tnt=tnt, hmm=hmm, unigram=unigram, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision,
                     tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall,unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "tnt", "regex"], bambara.test_sents, crf=crf, tnt=tnt, regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "tnt", "regex"], bambara.test_sents, crf=crf, tnt=tnt, regex=regex,
                     crf_tagprecision=crf_tagprecision, tnt_tagprecision=tnt_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, tnt_tagrecall=tnt_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "hmm", "regex"], bambara.test_sents, crf=crf, hmm=hmm, regex=regex,
                     crf_tagprecision=crf_tagprecision, hmm_tagprecision=hmm_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, hmm_tagrecall=hmm_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "unigram", "regex"], bambara.test_sents, crf=crf, unigram=unigram, regex=regex,
                     crf_tagprecision=crf_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["tnt","hmm" ,"regex"], bambara.test_sents, tnt=tnt, hmm=hmm, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, hmm_tagprecision=hmm_tagprecision, regex_tagprecision=regex_tagprecision,
                     tnt_tagrecall=tnt_tagrecall, hmm_tagrecall=hmm_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["tnt","unigram", "regex"], bambara.test_sents, tnt=tnt, unigram=unigram, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision,
                     tnt_tagrecall=tnt_tagrecall, unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["hmm","unigram", "regex"], bambara.test_sents, hmm=hmm, unigram=unigram, regex=regex,
                     hmm_tagprecision=hmm_tagprecision, unigram_tagprecision=unigram_tagprecision, regex_tagprecision=regex_tagprecision,
                     hmm_tagrecall=hmm_tagrecall, unigram_tagrecall=unigram_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["crf", "regex"], bambara.test_sents, crf=crf, regex=regex,
                     crf_tagprecision=crf_tagprecision, regex_tagprecision=regex_tagprecision,
                     crf_tagrecall=crf_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["tnt", "regex"], bambara.test_sents, tnt=tnt, regex=regex,
                     tnt_tagprecision=tnt_tagprecision, regex_tagprecision=regex_tagprecision,
                     tnt_tagrecall=tnt_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()

pr = PrecisionRecall(["hmm", "regex"], bambara.test_sents, hmm=hmm, regex=regex,
                     hmm_tagprecision=hmm_tagprecision, regex_tagprecision=regex_tagprecision,
                     hmm_tagrecall=hmm_tagrecall, regex_tagrecall=regex_tagrecall)
pr.final()
