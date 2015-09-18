# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

# implementation of 9-fold crossvalidation (see vanHalteren et al. (2001))
# call startCV() to run
from crf import CRFTagger
from nltk.tag import UnigramTagger, tnt, DefaultTagger, untag
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag.hmm import LidstoneProbDist
from nltk.metrics import accuracy
from create_reader import create_reader_9_1, dictionary_backoff
from collections import defaultdict, OrderedDict, Counter
from indivTaggers import *
from nltk.metrics import accuracy
from regextagger_tonal import Regexp as RegexpTonal
from regextagger_non_tonal import Regexp
from regextagger_non_tonal_SA import Regexp as RegexpSA
from regextagger_tonal_SA import Regexp as RegexpTonalSA


class CrossValidation(object): #of crf, tnt, hmm, unigram
    def __init__(self, folds, train_sents, option_tone, option_tag):
        self.option_tone=option_tone
        self.option_tag = option_tag
        self.folds = folds
        self.train_sents=train_sents # 90%of corpus
        self.foldsize=len(self.train_sents)//self.folds
        self.test_fold = None
        self.train_folds = None
        self.crf_avg_acc = None
        self.crf_tagprecision=None
        self.crf_tagrecall = None
        self.tnt_avg_acc=None
        self.tnt_tagprecision=None
        self.tnt_tagrecall=None
        self.hmm_avg_acc=None
        self.hmm_tagrecall=None
        self.hmm_tagprecision=None
        self.uni_avg_acc=None
        self.lasttagger_avg_acc=None
        self.unigram_tagprecision=None
        self.unigram_tagrecall=None
        self.bigram_avg_acc=None
        self.bigram_tagprecision=None
        self.bigram_tagrecall=None
        self.regex_avg_acc=None
        self.regex_tagprecision=None
        self.regex_tagrecall=None
        self.lasttagger_tagrecall=None
        self.lasttagger_tagprecision=None
        self.crf = None
        self.tnt=None
        self.hmm=None
        self.unigram=None
        self.bigram=None
        self.regex=None
        self.lasttagger=None
        self.foldlist=[]
        self.crf_tagged = []
        self.tnt_tagged=[]
        self.hmm_tagged = []
        self.uni_tagged = []
        self.regex_tagged = []
        self.lasttagger_tagged = []
        self.org_tagged = []
        for i in range(1, self.folds+1):
            self.foldlist.append(self.create_fold(i))

    def create_fold(self, i):
        if i ==1:
            self.fold1=[]
            return self.fold1
        if i == 2:
            self.fold2=[]
            return self.fold2
        if i == 3:
            self.fold3=[]
            return self.fold3
        if i == 4:
            self.fold4=[]
            return self.fold4
        if i ==5:
            self.fold5=[]
            return self.fold5
        if i == 6:
            self.fold6=[]
            return self.fold6
        if i == 7:
            self.fold7=[]
            return self.fold7
        if i == 8:
            self.fold8=[]
            return self.fold8
        if i == 9:
            self.fold9=[]
            return self.fold9

    def split_into_folds(self):
        n = len(self.train_sents)
        for i in range(0, n, self.folds):
            for j in self.foldlist:
                j.append(self.train_sents[i])
                i+=1
        
    
    def get_folds(self, k): #k = index of fold which serves for testing
        self.test_fold = self.train_sents[((k-1)*self.foldsize):(k*self.foldsize)]
        train_folds1 = self.train_sents[:(k-1)*self.foldsize]
        train_folds2 = self.train_sents[(k*self.foldsize):]
        self.train_folds = train_folds1+train_folds2  
        
    def trainALL(self, last):
        self.split_into_folds()
        for k in range(1,(self.folds+1)):
            train_sents = sum(self.foldlist[:(self.folds-1)],[])
            crf = CRFTagger(training_opt ={'max_iterations':100,'max_linesearch' : 10,'c1': 0.0001,'c2': 1.0})
            crf_trained = crf.train(train_sents ,'Models/model.crfCrossValidation1'+str(k)+self.option_tone+self.option_tag+'.tagger')
            print(str(k)+" fold: crf")
            tnt_tagger = tnt.TnT(unk=DefaultTagger('n'), Trained=True, N=100)
            tnt_tagger.train(train_sents)
            print(str(k)+" fold: tnt")
            tag_set= set()
            symbols=set()
            for i in train_sents:
                for j in i:
                    tag_set.add(j[1])
                    symbols.add(j[0])
            trainer = HiddenMarkovModelTrainer(list(tag_set), list(symbols))
            hmm = trainer.train_supervised(train_sents, estimator=lambda fd, bins:LidstoneProbDist(fd, 0.1, bins))
            print(str(k)+" fold: hmm")
            if last == "U":
                lasttagger = UnigramTagger(train_sents, backoff=DefaultTagger('n'))
                print(str(k)+" fold: unigram")
            if last == "B":
                if self.option_tone == "tonal" and self.option_tag == "Affixes":
                    regex=RegexpTonalSA(DefaultTagger('n'))
                if self.option_tone == "tonal" and self.option_tag == "POS":
                    regex=RegexpTonal(DefaultTagger('n'))
                if self.option_tone == "nontonal" and self.option_tag == "Affixes":
                    regex=RegexpSA(DefaultTagger('n'))
                if self.option_tone == "nontonal" and self.option_tag == "POS":
                    regex=Regexp(DefaultTagger('n'))
                dic = dictionary_backoff(self.option_tone, regex)
                affix=AffixTagger(train_sents, min_stem_length=0, affix_length=-4, backoff = dic) 
                lasttagger = BigramTagger(train_sents, backoff=affix)
                print(str(k)+" fold: bigram")
            to_tag = [untag(i) for i in self.foldlist[self.folds-1]]
            self.crf_tagged+=crf.tag_sents(to_tag)
            self.tnt_tagged+=tnt_tagger.tag_sents(to_tag)
            self.hmm_tagged+=hmm.tag_sents(to_tag)
            self.lasttagger_tagged+=lasttagger.tag_sents(to_tag)
            self.org_tagged+=self.foldlist[self.folds-1]
            self.foldlist=[self.foldlist[self.folds-1]]+self.foldlist[:(self.folds-1)]
        self.crf = crf
        self.tnt=tnt_tagger
        self.hmm=hmm
        self.lasttagger=lasttagger
        org_words=sum(self.org_tagged,[])
        self.crf_avg_acc = accuracy(org_words, sum(self.crf_tagged,[]))
        self.tnt_avg_acc = accuracy(org_words, sum(self.tnt_tagged,[]))
        self.hmm_avg_acc = accuracy(org_words, sum(self.hmm_tagged,[]))       
        self.lasttagger_avg_acc = accuracy(org_words, sum(self.lasttagger_tagged,[]))
        print("Accuracy of concatenated crf-tagged sentences: ",self.crf_avg_acc)
        print("Accuracy of concatenated tnt-tagged sentences: ",self.tnt_avg_acc)
        print("Accuracy of concatenated hmm-tagged sentences: ",self.hmm_avg_acc)
        print("Accuracy of concatenated "+last+"-tagged sentences: ", self.lasttagger_avg_acc)
        (self.crf_tagprecision, self.crf_tagrecall) = self.tagprecision_recall(crf, self.crf_tagged, self.org_tagged)
        (self.tnt_tagprecision, self.tnt_tagrecall) = self.tagprecision_recall(tnt_tagger, self.tnt_tagged, self.org_tagged)
        (self.hmm_tagprecision, self.hmm_tagrecall) = self.tagprecision_recall(hmm, self.hmm_tagged, self.org_tagged)
        (self.lasttagger_tagprecision, self.lasttagger_tagrecall) = self.tagprecision_recall(lasttagger, self.lasttagger_tagged, self.org_tagged)
        self.org_tagged = []
        self.foldlist = []
        for i in range(1, self.folds+1):
            self.foldlist.append(self.create_fold(i))

    def trainUniTnT(self):
        '''train unigram and tnt seperatly without DefaultTagger'''
        self.split_into_folds()
        for k in range(1,(self.folds+1)):
            train_sents = sum(self.foldlist[:(self.folds-1)],[])
            tnt_tagger = tnt.TnT(N=100)
            tnt_tagger.train(train_sents)
            print(str(k)+" fold: tnt evaluated")
            unigram = UnigramTagger(train_sents)
            print(str(k)+" fold: unigram evaluated")
            to_tag = [untag(i) for i in self.foldlist[self.folds-1]]
            self.tnt_tagged+=tnt_tagger.tag_sents(to_tag)
            self.uni_tagged+=unigram.tag_sents(to_tag)
            self.org_tagged+=self.foldlist[self.folds-1]
            self.foldlist=[self.foldlist[self.folds-1]]+self.foldlist[:(self.folds-1)]
        self.tnt=tnt_tagger
        self.unigram=unigram
        self.tnt_avg_acc = accuracy(sum(self.org_tagged,[]), sum(self.tnt_tagged,[]))
        self.uni_avg_acc = accuracy(sum(self.org_tagged,[]), sum(self.uni_tagged,[]))
        print("Accuracy of concatenated tnt-tagged sentences: ",self.tnt_avg_acc)
        print("Accuracy of concatenated unigram-tagged sentences: ", self.uni_avg_acc)
        (self.tnt_tagprecision, self.tnt_tagrecall) = self.tagprecision_recall(tnt_tagger, self.tnt_tagged, self.org_tagged)
        (self.unigram_tagprecision, self.unigram_tagrecall) = self.tagprecision_recall(unigram, self.uni_tagged, self.org_tagged)
        #delete following values so that trainRegexp has the inicial values
        self.org_tagged = []
        self.foldlist = []
        for i in range(1, self.folds+1):
            self.foldlist.append(self.create_fold(i))

    def trainRegexp(self, backoff):
        self.split_into_folds()
        for k in range(1,(self.folds+1)):
            train_sents = sum(self.foldlist[:(self.folds-1)],[])
            if self.option_tone == "tonal" and self.option_tag == "Affixes":
                regex=RegexpTonalSA(backoff)
            if self.option_tone == "tonal" and self.option_tag == "POS":
                regex=RegexpTonal(backoff)
            if self.option_tone == "nontonal" and self.option_tag == "Affixes":
                regex=RegexpSA(backoff)
            if self.option_tone == "nontonal" and self.option_tag == "POS":
                regex=Regexp(backoff)
            to_tag = [untag(i) for i in self.foldlist[self.folds-1]]
            self.regex_tagged+=regex.tag_sents(to_tag)
            self.org_tagged+=self.foldlist[self.folds-1]
            self.foldlist=[self.foldlist[self.folds-1]]+self.foldlist[:(self.folds-1)]
        self.regex = regex
        self.regex_avg_acc = accuracy(sum(self.org_tagged,[]), sum(self.regex_tagged,[]))
        print("Accuracy of concatenated regexp-tagged sentences: ",self.regex_avg_acc)
        (self.regex_tagprecision, self.regex_tagrecall) = self.tagprecision_recall(regex, self.regex_tagged, self.org_tagged)
        self.org_tagged =[]
        self.foldlist=[]
        for i in range(1, self.folds+1):
            self.foldlist.append(self.create_fold(i))

        
    def tagprecision_recall(self,tagger,tagger_tagged, org_tagged):
        '''For any tag X, precision measures which percentage of the tokens tagged X by
        the tagger are also tagged X in the benchmark.'''
        tagged_words_orig = sum(org_tagged, [])
        tags_orig = [i[1] for i in tagged_words_orig]
        tagset = list(set(tags_orig))
        tagged_words_tagger = sum(tagger_tagged, [])
        tags_tagger = [i[1] for i in tagged_words_tagger]
        counter_tagger = Counter(tags_tagger)
        counter_orig = Counter(tags_orig)
        precisions = dict()
        recalls = dict()
        for i in tagset:
            prec_rec = self.compare_tags_prec_rec(tags_tagger, tags_orig, i, counter_tagger, counter_orig)
            precisions[i] = prec_rec[0]
            recalls[i]=prec_rec[1]
        for j in tags_orig:
            if j not in precisions:
                precisions[j] = 0
            if j not in recalls:
                recalls[j]=0
        if "None" not in precisions:
            precisions["None"]=0
        if "None" not in recalls:
            recalls["None"]=1 #wegen 1-recall!
        print("precision, 1-recall calculated")
        return precisions, recalls

    def compare_tags_prec_rec(self, tags_tagger, tags_orig, tag_i, counter_tagger, counter_orig):
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
            recall = 1-(tagI_occ_orig_and_tagger/tagI_occ_original)#see van Halteren 1 - Recall
        return precision, recall



def startCV():
    tone = input("nontonal/tonal? -> ")
    while tone!= "nontonal" and tone !="tonal":
        print("wrong input")
        tone = input("nontonal/tonal? -> ")   
    tag = input("POS/Affixes? -> ")
    while tag!= "POS" and tag !="Affixes":
        print("wrong input")
        tag = input("POS/Affixes? -> ")
    print("Calculating CrossValidation")
    last = input("Unigram or Bigram+Affix+Dict+Regexp+Default? U/B -> ")
    while last != "U" and last != "B":
        print("wrong input")
        last = input("U/B -> ")
    bambara = create_reader_9_1(tone,tag)
    #print(len(bambara.train_sents)) #27864, 420914 words --> one fold contains about 46768 words
    #print(len(bambara.test_sents))
    cV = CrossValidation(9, bambara.train_sents, tone, tag)
    #cV.trainUniTnT() #needed to calculate tagprecision and tagrecall and totprecision of tnt and unigram without DefaultTagger 
    cV.trainALL(last)
    return cV
    
