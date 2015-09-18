# modified nltk.corpus.reader.xmldocs

# original author:
# Natural Language Toolkit: XML Corpus Reader
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE_Apache.TXT

####################

# modified with parts of HTMLReader (def parse_sent): https://raw.githubusercontent.com/maslinych/daba/master/formats.py
#(author: Kirill Maslinsky); daba is licensed under GNU General Public License:
# daba - suite of tools for automated interlinear glossing and manual disambiguisation
##
##(c) 2010-2011 Kirill Maslinsky <kirill@altlinux.org>
##
##This program is free software: you can redistribute it and/or modify
##it under the terms of the GNU General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##This program is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU General Public License for more details.
##
##You should have received a copy of the GNU General Public License
##along with this program (see LICENSE_GPL.txt).
##If not, see <http://www.gnu.org/licenses/>.

######################


"""
Corpus reader for daba html files
"""
from __future__ import print_function, unicode_literals

import codecs

# Use the c version of ElementTree, which is faster, if possible:
try: from xml.etree import cElementTree as ElementTree
except ImportError: from xml.etree import ElementTree

from nltk import compat
from nltk.data import SeekableUnicodeStreamReader
from nltk.tokenize import WordPunctTokenizer
from nltk.internals import ElementWrapper

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *

import unicodedata
import re
normalizeText = lambda t: unicodedata.normalize('NFKD', t)

from nltk.tag.util import untag


class XMLCorpusReader(CorpusReader):
    """
    Modified XMLCorpusReader for Corpus reader for corpora whose documents are in
    Daba´s HTML format
    """        
    def __init__(self, root, fileids, tone, tag, wrap_etree=False):
        self.fileids = fileids
        self._wrap_etree = wrap_etree
        CorpusReader.__init__(self, root, fileids)
        self.tagged_sents = []
        self.sents = []
        self.words =[]
        self.tagged_words = []
        self.option_tone = tone
        self.option_tag = tag

    def xml(self, fileid=None):
        '''Modified so that concatenating XML is possible
        (that means reading more than just one file).'''
        elt = ElementTree.parse(self.abspath(fileid).open()).getroot()
        # If requested, wrap it.
        if self._wrap_etree:
            elt = ElementWrapper(elt)
        # Return the ElementTree element.
        return elt
        
    def all_tagged_words(self):
        '''sets the class´s attribute tagged_words'''
        if len(self.tagged_words) != 0:
            print("tagged_words alreadi created!")
        else:
            self.tagged_words = sum(self.tagged_sents,[])
        
    def tagging_sents(self, fileid=None):
        '''Returns sentences of tagged words (tupel of word and ps) as lists in a list:
        [[(word1, ps1),(word2, ps2),...],[],...]'''
        elt = self.xml(fileid)
        encoding = self.encoding(fileid)
        word_tokenizer=WordPunctTokenizer()
        iterator = elt.getiterator()
        tagged_sents = []
        sents = []
        file = fileid
        for event, elem in ElementTree.iterparse(self.root+file):
            sents.append(elem)
            stags = []
        for sent in sents: # sents contains the elements marked with <>...</> in the xml file
            for sentence in sent.findall('span'):#find all <span...>...</span> - elements
                if sentence.attrib['class'] == ('sent'):
                    wtags = []
                    for span in sentence.findall('span'):
                        if span.attrib['class']=='annot':
                            for w in span.findall('span'):
                                if w.attrib['class'] == 'c':# to have a PoS-Tag for the punctuations marks also
                                    wtags.append((normalizeText((w.text).split("\n")[0]), 'c'))
                                if w.attrib['class']=='w':
                                    for lem in w.findall('span'):
                                        if lem.attrib['class']=='lemma':
                                            #1. option: POS --> only look for PoS-Tags
                                            if self.option_tag == 'POS':
                                                pstags = "_" # if there is no PoS-Tag
                                                for ps in lem.findall('sub'):
                                                    if ps.attrib['class'] == 'ps':
                                                        pstags = (normalizeText((ps.text).split("\n")[0]))
                                                        break
                                                #1.1 option: tonal
                                                if self.option_tone == "tonal":
                                                    wtags.append(((normalizeText(lem.text).replace(".", "")), pstags))
                                                #1.2 option: nontonal
                                                if self.option_tone == "nontonal":
                                                    wtags.append((self.strip_accents((normalizeText(lem.text).replace(".", ""))), pstags))
                                            #2. option: Affixes --> look for PoS-Tags and morphem markers
                                            if self.option_tag == "Affixes":
                                                pstags = "_"
                                                for ps in lem.findall('sub'):
                                                    if ps.attrib['class'] == 'ps':
                                                        pstags = (normalizeText((ps.text).split("\n")[0]))
                                                gloss = ''
                                                for m in lem.findall('span'): #only check direct span-children of lemma (do not take nesting into account)
                                                    counter = 0
                                                    for mps in m.findall('sub'): #type(m.findall('sub')) = list
                                                        if mps.attrib['class'] == 'ps' and normalizeText(mps.text) != 'mrph': #then discard 'm' - class and go to next span element
                                                            break
                                                        else:
                                                            if mps.attrib['class'] == 'gloss':
                                                                newgloss = normalizeText(mps.text).split()[0]
                                                                if re.search(r'[A-Z]', newgloss):
                                                                    gloss = gloss + '|' + newgloss
                                                                break
                                                        counter+=1
                                                #2.1 option: tonal
                                                if self.option_tone == "tonal":
                                                    wtags.append((normalizeText((lem.text.replace(".","")).split("\n")[0]), pstags+gloss))
                                                #2.2 option: nontonal
                                                if self.option_tone == "nontonal":
                                                    wtags.append((self.strip_accents(normalizeText((lem.text.replace(".","")).split("\n")[0])), pstags+gloss))                                                            
                    stags.append(wtags)  
        self.tagged_sents = self.tagged_sents+stags


    def strip_accents(self, word):
        '''delete tonal markers for nontonal reading'''
        return ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')
    

    def first_capitalize(self, sentence):
        '''Capitalize first word in sentence'''
        capitalized_sentence = [(sentence[0][0].capitalize(),sentence[0][1])]+sentence[1:]
        return capitalized_sentence
        
    def all_tagging_sents(self):
        '''set class´s attribute "tagged_sentence"'''
        if len(self.tagged_sents) != 0:
            print("tagged_sents alreadi created")
        else:
            for i in self.fileids:
                self.tagging_sents(i)
            self.tagged_sents = [self.first_capitalize(i) for i in self.tagged_sents if i != []]

    def all_sents(self):
        '''set class´s attribute "sents"'''
        self.sents = [untag(i) for i in self.tagged_sents]

    def all_words(self):
        '''set class´s attribute "words"'''
        self.words = sum(self.sents, [])

    def raw(self, fileids=None):
        if fileids is None: fileids = self._fileids
        elif isinstance(fileids, compat.string_types): fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])
