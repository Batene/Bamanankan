# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

from create_reader import create_reader
from indivTaggers import *

tone = input("nontonal/tonal? -> ")
while tone!= "nontonal" and tone !="tonal":
    print("wrong input")
    tone = input("nontonal/tonal? -> ")
   
tag = input("POS/Affixes? -> ")
while tag!= "POS" and tag !="Affixes":
    print("wrong input")
    tag = input("POS/Affixes? -> ")

def disambiguated_i(sent):
    dis_sent = []
    for w in sent:
        if len(w[1].split("/"))>1:
            return False
    return True


bambara = create_reader(tone, tag)

disamb_train_sents = [i for i in bambara.train_sents if disambiguated_i(i) == True]
disamb_test_sents = [i for i in bambara.test_sents if disambiguated_i(i) == True]#==devset
disamb_testset =  [i for i in bambara.testset if disambiguated_i(i) == True]

bambara.train_sents = disamb_train_sents
bambara.test_sents = disamb_test_sents
bambara.testset=disamb_testset

print("Größe Korpus nach Disambiguierung: ")
print(len(bambara.train_sents)+len(bambara.test_sents)+len(bambara.testset))



###to run for different taggers:

##backoff=indivDefault(bambara)
##dic = indivDic(bambara,tone)
##regex = indivRegexp(bambara,tag,tone,backoff=backoff)
##crf = indivCRF(bambara, tone, tag)
#tnt = indivTnT(bambara,backoff=backoff)
##hmm = indivHMM(bambara)
##unigram = indivUnigram(bambara,backoff=backoff)
##bigram = indivBigram(bambara,backoff=backoff)
##trigram = indivTrigram(bambara,backoff=backoff)
##affix = indivAffix(bambara,(-2),backoff=backoff)
##affix = indivAffix(bambara,3,backoff=backoff)
##affix = indivAffix(bambara,-4,backoff=None)



         
            
