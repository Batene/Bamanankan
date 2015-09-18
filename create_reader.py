# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

from nltk.tag import UnigramTagger, TrigramTagger
from nltk.tag import DefaultTagger
import getpass
from os import listdir
import os
from bambara_tagging_htmlreaderALL import BambaraTagging
from toolboxreaderRun import get_alt_pos
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import ToolboxCorpusReader


def create_reader_9_1(option_tones, option_tag):
    '''Creates a reader according to the options: tonal/nontonal and POS/Affixes'''
    user = getpass.getuser()
    path = "cookbook/bambara/"
    path1 = os.getcwd()
    files = ["Corpus/"+f for f in listdir(path1+"\\Corpus") if f[-4:] == "html"]
    bambara = BambaraTagging(path, files, option_tones, option_tag)
    bambara.copy_files()
    print("\nPlease wait, reader is being created ...\n")
    bambara.create_reader()
    bambara.sets9_1(90)
    print(option_tones+" "+option_tag+" 90% Train-, 10% Testset created")
    return bambara


def create_reader(option_tones, option_tag):
    '''Creates a reader according to the options: tonal/nontonal and POS/Affixes'''
    user = getpass.getuser()
    path = "cookbook/bambara/"
    path1 = os.getcwd()
    files = ["Corpus/"+f for f in listdir(path1+"\\Corpus") if f[-4:] == "html"]
    bambara = BambaraTagging(path, files, option_tones, option_tag)
    bambara.copy_files()
    print("\nPlease wait, reader is being created...\n")
    bambara.create_reader()
    bambara.sets8_1_1(90)
    print(option_tones+" "+option_tag+" 80% Train-, 10% Dev-, 10% Testset created")
    return bambara


def dictionary(option_tone):
    '''Creates a dictionary according to the option: tonal/nontonal'''
    if option_tone == "tonal":
        bambara_dict_toolbox = BambaraTagging("cookbook/bambara", ["bamadaba.txt"], option_tone, "POS")
        bambara_dict_toolbox.copy_files()
        reader = LazyCorpusLoader("cookbook/bambara/", ToolboxCorpusReader, ["bamadaba.txt"])
        entries = reader.entries("bamadaba.txt") #tonal
        words = reader.words("bamadaba.txt")#tonal
        pos = reader.words("bamadaba.txt", key="ps")#tonal
    else:
        bambara_dict_toolbox = BambaraTagging("cookbook/bambara", ["bamadaba_non_tonal.txt"], option_tone, "POS")
        bambara_dict_toolbox.copy_files()
        reader = LazyCorpusLoader("cookbook/bambara/", ToolboxCorpusReader, ["bamadaba_non_tonal.txt"])
        entries = reader.entries("bamadaba_non_tonal.txt") #tonal
        words = reader.words("bamadaba_non_tonal.txt")#tonal
        pos = reader.words("bamadaba_non_tonal.txt", key="ps")#tonal
        
    own_model = get_alt_pos(entries, pos, reader, option_tone)#tonal
    print("Dictionary created")
    dic = UnigramTagger(model=own_model, backoff=DefaultTagger('n'))
    return dic



def dictionary_backoff(option_tone, backoff):
    '''Creates a dictionary according to the option: tonal/nontonal'''
    if option_tone == "tonal":
        bambara_dict_toolbox = BambaraTagging("cookbook/bambara", ["bamadaba.txt"], option_tone, "POS")
        bambara_dict_toolbox.copy_files()
        reader = LazyCorpusLoader("cookbook/bambara/", ToolboxCorpusReader, ["bamadaba.txt"])
        entries = reader.entries("bamadaba.txt") #tonal
        words = reader.words("bamadaba.txt")#tonal
        pos = reader.words("bamadaba.txt", key="ps")#tonal
    else:
        bambara_dict_toolbox = BambaraTagging("cookbook/bambara", ["bamadaba_non_tonal.txt"], option_tone, "POS")
        bambara_dict_toolbox.copy_files()
        reader = LazyCorpusLoader("cookbook/bambara/", ToolboxCorpusReader, ["bamadaba_non_tonal.txt"])
        entries = reader.entries("bamadaba_non_tonal.txt") #tonal
        words = reader.words("bamadaba_non_tonal.txt")#tonal
        pos = reader.words("bamadaba_non_tonal.txt", key="ps")#tonal
        
    own_model = get_alt_pos(entries, pos, reader, option_tone)#tonal
    print("Dictionary created")
    dic = UnigramTagger(model=own_model, backoff=backoff)
    return dic
