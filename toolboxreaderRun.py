# Copyright (C) 2015 Kathrin Donandt
# For license information see LICENSE.txt

# get alternative words for a dictionary entry
from nltk.corpus.reader import ToolboxCorpusReader
from bambara_tagging_htmlreaderALL import BambaraTagging


def get_alt_pos(entries, pos,reader, tone):
    dic = {}
    counter = 0
    if tone == "tonal":
        for i in reader.entries("bamadaba.txt"):
            if (i[0] in dic.keys()) == False:
                dic[i[0]] = pos[counter]
            for j in i[1]:
                if j[0] == 'va':
                    dic[j[1]] = pos[counter]
            counter+=1
    else:
        for i in reader.entries("bamadaba_non_tonal.txt"):
            if (i[0] in dic.keys()) == False:
                dic[i[0]] = pos[counter]
            for j in i[1]:
                if j[0] == 'va':
                    dic[j[1]] = pos[counter]
            counter+=1
        
    return dic
