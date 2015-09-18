from nltk.tag import RegexpTagger
from patterns_non_tonal import patterns

class Regexp(RegexpTagger):
    def __init__(self, backoff=None):
        print("Regexp")
        RegexpTagger.__init__(self, patterns, backoff=backoff)
