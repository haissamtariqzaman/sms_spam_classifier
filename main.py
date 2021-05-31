import numpy as np
import pandas as pd
import os
import re
import csv

csv_file = 'spam.csv'

def read_file():
    label = []
    msg = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label.append(row.get('v1'))
            msg.append(row.get('v2'))

    return label, msg

def tokenize(messages):
    for index in range(len(messages)):
        messages[index]=messages[index].split()

def removeURLs(messages):
    for x in range(len(messages)):
        sentenceLen = len(messages[x])
        y=0
        while y<sentenceLen:
            result = re.search("^http://|^https://|^www\.", messages[x][y])
            if (result != None):
                del messages[x][y]
                sentenceLen-=1;
            else:
                y+=1

def main():
    label, msg = read_file()
    tokenize(msg)
    print(label)
    removeURLs(msg)

if __name__ == '__main__':
    main()