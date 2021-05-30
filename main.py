import numpy as np
import pandas as pd
import os
import csv

csv_file = 'spam.csv'

def read_file():
    label = []
    msg = []
    dataset = pd.read_csv(csv_file)
    for l in range(len(dataset)):
        label.append(row.get('v1'))
        msg.append(row.get('v2'))

    return label, msg

def tokenize(messages):
    for index in range(len(messages)):
        messages[index]=messages[index].split()

def main():
    label, msg = read_file()
    tokenize(msg)
    print(label)

    for x in range(len(msg)):
        print(msg[x])

if __name__ == '__main__':
    main()