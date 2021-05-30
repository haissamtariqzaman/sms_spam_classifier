import numpy as np
import pandas as pd
import os
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


def main():
    label, msg = read_file()
    print(label)
    print(msg)


if __name__ == '__main__':
    main()
