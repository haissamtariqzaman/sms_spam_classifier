import numpy as np
import pandas as pd
import os
import re
import csv
import contractions
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class SpamClassifier:

    fileName=None
    messages=None
    label=None

    def __init__(self,fileName):
        self.fileName=fileName
        self.label=[]
        self.messages=[]

    def read_file(self):
        with open(self.fileName, 'r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.label.append(row.get('v1'))
                self.messages.append(row.get('v2'))

    def tokenize(self):
        for index in range(len(self.messages)):
            self.messages[index] = self.messages[index].split()

    def removeURLs(self):
        for x in range(len(self.messages)):
            sentenceLen = len(self.messages[x])
            y = 0
            while y < sentenceLen:
                result = re.search("^http://|^https://|^www\.", self.messages[x][y])
                if (result != None):
                    del self.messages[x][y]
                    sentenceLen -= 1;
                else:
                    y += 1

    def fixContractions(self):
        for x in range(len(self.messages)):
            self.messages[x] = contractions.fix(self.messages[x])

    def toLowerCase(self):
        for x in range(len(self.messages)):
            for y in range(len(self.messages[x])):
                self.messages[x][y]=self.messages[x][y].lower()

    def removeStopWords(self):
        self.toLowerCase()
        stopWords=stopwords.words('english')

        for x in range(len(self.messages)):
            sentenceLen = len(self.messages[x])
            y = 0
            while y < sentenceLen:
                if self.messages[x][y] in stopWords:
                    del self.messages[x][y]
                    sentenceLen-=1
                else:
                    y+=1

    def printMessages(self):
        for sentence in self.messages:
            print(sentence)

    def printLabels(self):
        print(self.label)