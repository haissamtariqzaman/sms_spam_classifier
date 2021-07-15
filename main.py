from SpamClassifier import SpamClassifier

def main():
    spam_classifier=SpamClassifier('spam.csv')
    spam_classifier.read_file()
    spam_classifier.fixContractions()
    spam_classifier.tokenize()
    spam_classifier.removeURLs()
    spam_classifier.removePunc()
    #spam_classifier.removeStopWords()
    spam_classifier.labelsToNumeric()
    spam_classifier.join()
    spam_classifier.messageToNumeric()
    spam_classifier.makeSameSize()
    #spam_classifier.printMessageSequence()
    spam_classifier.LSTMModel()


if __name__ == '__main__':
    main()