from nltk.tag import RegexpTagger
from patterns_tonal_SA import patterns


class Regexp(RegexpTagger):
    def __init__(self, backoff=None):
        RegexpTagger.__init__(self, patterns, backoff=backoff)
