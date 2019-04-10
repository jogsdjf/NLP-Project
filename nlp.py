
import json
import nltk
import re

from collections import defaultdict
from nltk.corpus import stopwords
from pattern.en import parse, Sentence, mood, parsetree
from pattern.db import csv
from pattern.search import search
from pattern.vector import Document, NB

def readText():
    """
    Reads the text from a text file.
    """
    with open("730.txt", "rb") as f:
        text = f.read().decode('utf-8-sig',errors='ignore')
    return text


def chunkSentences(text):
    """
    Parses text into parts of speech tagged with parts of speech labels.
    """
    sentences = nltk.sent_tokenize(text)
    tokenizedSentences = [nltk.word_tokenize(sentence)
                          for sentence in sentences]
    taggedSentences = [nltk.pos_tag(sentence)
                       for sentence in tokenizedSentences]
    if nltk.__version__[0:2] == "2.":
        chunkedSentences = nltk.batch_ne_chunk(taggedSentences, binary=True)
    else:
        chunkedSentences = nltk.ne_chunk_sents(taggedSentences, binary=True)
    return chunkedSentences


def extractEntityNames(tree, _entityNames=None):
    """
    Creates a local list to hold nodes of tree passed through, extracting named
    entities from the chunked sentences.
    """
    if _entityNames is None:
        _entityNames = []
    try:
        if nltk.__version__[0:2] == "2.":
            label = tree.node
        else:
            label = tree.label()
    except AttributeError:
        pass
    else:
        if label == 'NE':
            _entityNames.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                extractEntityNames(child, _entityNames=_entityNames)
    return _entityNames


def buildDict(chunkedSentences, _entityNames=None):
    """
    Uses the global entity list, creating a new dictionary with the properties
    extended by the local list, without overwriting.
    """
    if _entityNames is None:
        _entityNames = []

    for tree in chunkedSentences:
        extractEntityNames(tree, _entityNames=_entityNames)

    return _entityNames


def removeStopwords(entityNames, customStopWords=None):
    """
    Brings in stopwords and custom stopwords to filter mismatches out.
    """
    # Memoize custom stop words
    if customStopWords is None:
        with open("customStopWords.txt", 'r') as f:
            customStopwords = f.read().split(', ')

    for name in entityNames:
        if name in stopwords.words('english') or name in customStopwords:
            entityNames.remove(name)


def getMajorCharacters(entityNames):
    """
    Adds names to the major character list if they appear frequently.
    """
    return {name for name in entityNames if entityNames.count(name) > 10}


def splitIntoSentences(text):
    """
    Split sentences on .?! "" and not on abbreviations of titles.
    Used for reference: http://stackoverflow.com/a/8466725
    """
    sentenceEnders = re.compile(r"""
    # Split sentences on whitespace between them.
    (?:               # Group for two positive lookbehinds.
      (?<=[.!?])      # Either an end of sentence punct,
    | (?<=[.!?]['"])  # or end of sentence punct and quote.
    )                 # End group of two positive lookbehinds.
    (?<!  Mr\.   )    # Don't end sentence on "Mr."
    (?<!  Mrs\.  )    # Don't end sentence on "Mrs."
    (?<!  Ms\.   )    # Don't end sentence on "Ms."
    (?<!  Jr\.   )    # Don't end sentence on "Jr."
    (?<!  Dr\.   )    # Don't end sentence on "Dr."
    (?<!  Prof\. )    # Don't end sentence on "Prof."
    (?<!  Sr\.   )    # Don't end sentence on "Sr."
    \s+               # Split on whitespace between sentences.
    """, re.IGNORECASE | re.VERBOSE)
    return sentenceEnders.split(text)


def compareLists(sentenceList, majorCharacters):
    """
    Compares the list of sentences with the character names and returns
    sentences that include names.
    """
    characterSentences = defaultdict(list)
    for sentence in sentenceList:
        for name in majorCharacters:
            if re.search(r"\b(?=\w)%s\b(?!\w)" % re.escape(name),
                         sentence,
                         re.IGNORECASE):
                characterSentences[name].append(sentence)
    return characterSentences


def extractMood(characterSentences):
    """
    Analyzes the sentence using grammatical mood module from pattern.
    """
    characterMoods = defaultdict(list)
    for key, value in characterSentences.items():
        for x in value:
            #print(x)
            characterMoods[key].append(mood(Sentence(parse(str(x),
                                                           lemmata=True))))
    return characterMoods


def extractSentiment(characterSentences):
    """
    Trains a Naive Bayes classifier object with the reviews.csv file, analyzes
    the sentence, and returns the tone.
    """
    nb = NB()
    characterTones = defaultdict(list)
    for review, rating in csv("reviews.csv"):
        nb.train(Document(review, type=int(rating), stopwords=True))
    for key, value in characterSentences.items():
        for x in value:
            characterTones[key].append(nb.classify(str(x)))
    return characterTones

def extractTrait(characterSentences):
    """
    Analyzes the sentence using serach module of pattern for adjective.
    """
    print(1)
    characterTrait = defaultdict(list)
    for key,value in characterSentences.items():
        for x in value:
            #print(x)
            #t=parsetree(x)
            characterTrait[key].append(search('JJ',parsetree(str(x))))
            #print(search('JJ',parsetree(str(x))))
            
    return characterTrait
def writeTraits(sentenceTrait):
    """
    Writes the traits to a text file in the same directory.
    """
    with open("traits.txt", "w") as f:
        for item in sentenceTrait.items():
            f.write("%s:%s\n" % item)


def writeAnalysis(sentenceAnalysis):
    """
    Writes the sentence analysis to a text file in the same directory.
    """
    with open("sentenceAnalysis.txt", "w") as f:
        for item in sentenceAnalysis.items():
            f.write("%s:%s\n" % item)

'''def traitJSON(sentenceTrait):
    """
    Writes the sentence analysis to a JSON file in the same directory.
    """
    with open("trait.json", "w") as f:
        json.dump(sentenceTrait, f)'''
    
def writeToJSON(sentenceAnalysis):
    """
    Writes the sentence analysis to a JSON file in the same directory.
    """
    with open("sentenceAnalysis.json", "w") as f:
        json.dump(sentenceAnalysis, f)


if __name__ == "__main__":
    text = readText()

    chunkedSentences = chunkSentences(text)
    #print(chunkedSentences)
    entityNames = buildDict(chunkedSentences)
    #print(entityNames)
    removeStopwords(entityNames)
    majorCharacters = getMajorCharacters(entityNames)
    
    sentenceList = splitIntoSentences(text)
    characterSentences = compareLists(sentenceList, majorCharacters)
    characterMoods = extractMood(characterSentences)
    characterTones = extractSentiment(characterSentences)
    characterTrait = extractTrait(characterSentences)
    #print(characterTrait)

    # Merges sentences, moods and tones together into one dictionary on each
    # character.
    sentenceAnalysis = defaultdict(list,
                                   [(k, [characterSentences[k],
                                         characterTones[k],
                                         characterMoods[k]])
                                    for k in characterSentences])

    sentenceTrait = defaultdict(list,
                                [(k,[characterTrait[k]]) for k in characterSentences])
    writeTraits(sentenceTrait)
    writeAnalysis(sentenceAnalysis)
    writeToJSON(sentenceAnalysis)
    #traitJSON(sentenceTrait)
