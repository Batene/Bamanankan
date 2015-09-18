# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

#Some code from the book Python 3 Text Processing with NLTK 3 Cookbook

from nltk.tag import untag
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import ToolboxCorpusReader
from confusionmatrix import ConfusionMatrix
from htmlreaderALL import XMLCorpusReader
import codecs
import getpass
import time

class BambaraTagging(object):
    def __init__(self, root, file_list, option_tone, option_tag):
        self.root = root #only the pathname after "C:Users/<username>/nltk_data/corpora"; example: 'cookbook\\bambara' (instead of: C:/Users/<username>/nltk_data/corpora/cookbook/bambara)
        self.file_list = file_list
        self.option_tone = option_tone
        self.option_tag = option_tag
        self.reader = None
        self.anzahl_sents = 0
        self.train_sents = []
        self.test_sents = [] #=devset!
        self.unigramtagger = None
        self.bigramtagger = None
        self.trigramtagger = None
        self.contingenzliste = ["Getagged   :   Ursprüngliches Tag"+"\n"]
        self.reference_tags = []
        self.test_tags = []
        self.evaluate = 0
        self.evaluate_final = 0
        self.user = getpass.getuser()
        self.testset=[] #real test set
     
    def copy_files(self):
        """ Copies the corpus files (self.file_list) to the
        C:/Users/<username>/nltk_data/corpora/cookbook/bambara for further usage.
        If this directory does not exist yet, it will be created also.
        """
        print("Checking corpus directory...")
        if not os.path.exists("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook\\bambara"):
            print("Creating corpus directories...")
            os.mkdir("C:\\Users\\"+self.user+"\\nltk_data\\corpora")
            os.mkdir("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook")
            os.mkdir("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook\\bambara")
            os.mkdir("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook\\bambara\\Corpus")
        else:
            print("nltk_data/corpora/cookbook/bambara folder exists")
        print("Checking corpus files...")
        for file in self.file_list:
            if not os.path.exists(file):
                print(file, "This corpus file does not exist")
                raise IOError('File does not exist:', file)
            else:
                if not os.path.exists("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook\\bambara\\"+file):
                    print("Corpus file ", file, "does not exist yet.")
                    print("Copying file to nltk_data/corpora/cookbook/bambara. Please wait. This may take a while.")
                    # read the desired corpus files (given in self.file_list) from the folder
                    f = codecs.open(file, "r+", "utf-8")
                    # copies these corpus files in the right user directory
                    g = codecs.open("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook\\bambara\\"+file, "w", "utf-8")
                    lines = f.readlines()
                    output = []
                    for i in lines:
                        output.append(i)
                    g.writelines(output)
                    g.close()
                    f.close()
                    print("Copied file: ", file)
                else:
                    print("Corpus file exists")     
    
    def create_reader(self):
        """Reads the corpus files with the XMLCorpusReader. See book for further explanations"""
        self.reader = XMLCorpusReader("C:\\Users\\"+self.user+"\\nltk_data\\corpora\\cookbook\\bambara\\", self.file_list, self.option_tone, self.option_tag)
        self.reader.all_tagging_sents()
        self.reader.all_sents()
        self.anzahl_sents = len(self.reader.tagged_sents)
        return self.reader, self.anzahl_sents
    
    def sets8_1_1(self, split):
        """
        Creation of train- and test-set. Furthermore, develoment set is created that serves to optimize the trained tagger.
        In 10 sentences, 1st-4th sentences and 6th-9th sentences go to the training set, while the 5th sentence goes to the test set
        and 10the sentence goes to the dev set.
        This partition of the corpus sentences is done in steps: <split> gives the numer of the bundles of sentences that are treated
        together.
        """
        n = (self.anzahl_sents//split) # gives the number of bundles consisting of split sentences
        saetze = self.reader.sents[:(n*split)]
        saetzetagged = self.reader.tagged_sents[:(n*split)]
        for i in range(n):
            s_split= saetze[:split]
            s_tag_split = saetzetagged[:split]
            for j in range(0, split, 10):
                #print(i, j)
                self.train_sents.append(s_tag_split[j]) # tagged sentences to train the tagger
                self.train_sents.append(s_tag_split[j+1])
                self.train_sents.append(s_tag_split[j+2])
                self.train_sents.append(s_tag_split[j+3])
                self.train_sents.append(s_tag_split[(j+5)])
                self.train_sents.append(s_tag_split[j+6])
                self.train_sents.append(s_tag_split[j+7])
                self.train_sents.append(s_tag_split[j+8])
                self.test_sents.append(s_tag_split[j+4]) # tagged sentences to test and optimize the tagger
                self.testset.append(s_tag_split[j+9]) # tagged sentences to check optimized tagger
            saetze = saetze[split:]
            saetzetagged = saetzetagged[split:]
        
    def sets9_1(self, split):
        """
        Creation of train- and test-set. Furthermore, development set is created that serves to optimize the trained tagger.
        In 10 sentences, 1st-4th sentences and 6th-9th sentences go to the training set, while the 5th sentence goes to the test set
        and 10the sentence goes to the dev set.
        This partition of the corpus sentences is done in steps: <split> gives the numer of the bundles of sentences that are treated
        together.
        """
        n = (self.anzahl_sents//split) # gives the number of bundles consisting of split sentences
        saetze = self.reader.sents[:(n*split)]
        saetzetagged = self.reader.tagged_sents[:(n*split)]
        for i in range(n):
            s_split= saetze[:split]
            s_tag_split = saetzetagged[:split]
            for j in range(0, split, 10):
                #print(i, j)
                self.train_sents+=s_tag_split[j:(j+9)] # tagged sentences to train the tagger
                self.test_sents.append(s_tag_split[j+9]) # tagged sentences to test and optimize the tagger
            saetze = saetze[split:]
            saetzetagged = saetzetagged[split:]
   

    def calculate_contingenz_with_sets(self, tagger):
        """
        Compares the original tags with the tags created by the tagger.
        """
        tagger_tagged = tagger.tag_sents([untag(i) for i in self.test_sents])
        tagger_words = sum(tagger_tagged,[])
        original_tagged = self.test_sents
        original_words = sum(original_tagged,[])
        tagged_org_zip = zip([i[1] for i in original_words],[i[1] for i in tagger_words])
        contingenzliste = []
        orig_tags = []
        tag_tags = []
        for i in tagged_org_zip:
            if i[0] != i[1]:
                if i[1] == None:
                    i = (i[0], "None")
                contingenzliste.append(i[1]+"   :   "+i[0]+"\n")
                orig_tags.append(i[0])
                tag_tags.append(i[1])   

        self.contingenzliste = self.contingenzliste + contingenzliste
        self.reference_tags = self.reference_tags + orig_tags
        self.test_tags = self.test_tags + tag_tags


    def matrix(self):
        """Creates a Contingenz Matrix using ConfusionMatrix of NLTK"""
        cm = ConfusionMatrix(self.reference_tags, self.test_tags) # first reference, then test!
        #f = codecs.open("C:\\Users\\"+self.user+"\\Downloads\\continenzmatrix.txt", "w", "utf-8")
        f = codecs.open("Results\\contingenzmatrix.txt","w","utf-8")
        f.write(cm.pp())
        f.close()
        ###print contingenzliste
        #g = codecs.open("C:\\Users\\"+self.user+"\\Downloads\\contingenzliste.txt", "w", "utf-8")
        #g.writelines(self.contingenzliste)
        #g.close()
        values_not_null = cm.get_values_not_null()
        return values_not_null


#create the directory for the corpus file: (C:/Users/username/nltk_data/corpora/cookbook/bambara and add the corpus file corbama-net-non-tonal_conll.txt

import os, os.path
path = os.path.expanduser('~/nltk_data')
if not os.path.exists(path):
    os.mkdir(path)
print("Checking nltk_data directory...")
print(os.path.exists(path))
import nltk.data
print(path in nltk.data.path)
