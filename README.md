# Bamanankan

README

Contact: kathrindonandt@yahoo.de

To run the programs, the following programs have to be installed:
1. python 3;
2. NLTK (http://www.nltk.org/install.html);
3. CRFSuite https://pypi.python.org/pypi/python-crfsuite

Running the programs will create the folder "nltk_data" in C:/User directory, where the corpus files in the folder "Corpus" will be automatically copied to.


Containing Folders
==================

!Note: Corpusreader htmlreader.py is supposed to read files in Daba HTML format (Bambara Reference Corpus).

Corpus
- Folder which has to contain the corpus files

Models
- folder empty, but will be used to store models to when CRFTaggers is trained

Results
- folder empty, results of analyzeContingency.py will be saved to this folder



Containing Files 
================

analyzeContingency.py 
- calculates percentage of words tagged i which in reality are j; saves result to a file in Results folder
- looks for the words which are responsible for these errors and save each error to a file in Results folder
- saves confusionmatrix to Results folder

backoffCombi.py
- combines taggers according to the backoff-chaining given in NLTK

bambara_tagging_htmlreaderALL.py
- loads the corpus files and creates a reader needed to work with the sentences/words in the corpus
- used by create_reader

bamadaba.txt
- the Bamadaba dictionary (http://cormand.huma-num.fr/bamadaba.html)

bamadaba_non_tonal.txt
- the Bamadaba dictionary (http://cormand.huma-num.fr/bamadaba.html), but without tones

confusionmatrix.py
- slightly modified version of confusionmatrix.py of NLTK
- function was added so that switches (tagger tagged word with tag A instead of the right tag B) can be analyzed

create_reader.py
- uses bambara_tagging_htmlreaderALL to create a reader (with htmlreaderALL.py) needed to work with the corpus files

crf.py
- modified crf of NLTK (features added)

CrossValidation.py
- implementation of a 9-fold crossvalidation

disambiguation.py
- removes sentences containing words tagged ambiguously (e.g. n/v)

ensemblecombinationBrillWu_Html.py
- calculates complementarity and disagreement of the taggers CRF, TnT, HMM and either Unigram or a backoff Tagger (Bigram+Affix+Dictionary+Regexp+DefaultTagger) according to Brill & Wu (1998)
- save result to files in Results folder

ensemblecombinationBrillWu_HtmlREGEX,py
- calculates complementarity and disagreement of the taggers CRF, TnT, HMM, Unigram, Regexp
- saves result to files in Results folder

htmlreaderALL.py
- reader for Daba HTML files 
- it´s a modified xml reader of NLTK: nltk.corpus.reader.xmldocs; uses parts of HTMLReader of Kirill Maslinsky

indivTaggers.py
- trains individual taggers more easily 
- also used by other programs

patterns[_non]_tonal[_SA].py
- patterns for the RegexpTagger, see http://cormand.huma-num.fr/gloses.html 

regextagger_[non_]tonal[_SA].py
- RegexpTagger for each form of training (tonal or nontonal, with or without Affixes)

toolboxreaderRun.py
- contains function to get the alternative words to an entry in the dictionary
- use to create the DictionaryTagger

Voting.py
- implementation of several voting strategies of the ensemble combination


(for further information on the files, look into the files header)






