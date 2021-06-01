from SpamClassifier import SpamClassifier

def main():
    spam_classifier=SpamClassifier('spam.csv')
    spam_classifier.read_file()
    spam_classifier.fixContractions()
    spam_classifier.tokenize()
    spam_classifier.removeURLs()
    spam_classifier.removeStopWords()
    spam_classifier.printMessages()
    spam_classifier.printLabels()

if __name__ == '__main__':
    main()