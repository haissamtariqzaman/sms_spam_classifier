import contractions
import csv
import nltk
import re
from sklearn.model_selection import train_test_split

import numpy as np
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

nltk.download('stopwords')


class SpamClassifier:
    fileName = None
    messages = None
    label = None
    n_label = None
    message_sequence = None
    tokenizer = None
    VOC_SIZE = None
    max_length_sequence = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, fileName):
        self.fileName = fileName
        self.label = []
        self.messages = []
        self.n_label = []
        self.tokenizer = Tokenizer()

    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.message_sequence, self.n_label,
                                                                                test_size=0.20, random_state=42)

    def read_file(self):
        with open(self.fileName, 'r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.label.append(row.get('v1'))
                self.messages.append(row.get('v2'))

    def tokenize(self):
        for index in range(len(self.messages)):
            self.messages[index] = self.messages[index].split()

    def removePunc(self):
        test_list = []
        alpha = 'a'
        for i in range(0, 26):
            test_list.append(alpha)
            alpha = chr(ord(alpha) + 1)
        alpha = 'A'
        for i in range(0, 26):
            test_list.append(alpha)
            alpha = chr(ord(alpha) + 1)

        # print(test_list)
        counter = 0
        nullSTR = ""
        for x in range(len(self.messages)):
            for y in range(len(self.messages[x])):
                counter = len(self.messages[x][y])
                z = 0
                nullSTR = ""
                while z < counter:
                    if self.messages[x][y][z] in test_list:
                        nullSTR = nullSTR + self.messages[x][y][z]
                    z += 1
                self.messages[x][y] = nullSTR

        for x in range(len(self.messages)):
            self.messages[x] = self.remove_empty_string(self.messages[x])
        # print(self.messages)

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
                self.messages[x][y] = self.messages[x][y].lower()

    def removeStopWords(self):
        self.toLowerCase()
        stopWords = stopwords.words('english')

        for x in range(len(self.messages)):
            sentenceLen = len(self.messages[x])
            y = 0
            while y < sentenceLen:
                if self.messages[x][y] in stopWords:
                    del self.messages[x][y]
                    sentenceLen -= 1
                else:
                    y += 1

    def printMessages(self):
        for sentence in self.messages:
            print(sentence)

    def printLabels(self):
        print(self.label)

    def printn_labels(self):
        print(self.n_label)

    def labelsToNumeric(self):
        for lbl in self.label:
            if lbl == "spam":
                self.n_label.append(1)
            else:
                self.n_label.append(0)

    def printMessageSequence(self):
        print(self.message_sequence)

    def join(self):
        for x in range(len(self.messages)):
            self.messages[x] = " ".join(self.messages[x])

    def messageToNumeric(self):
        self.tokenizer.fit_on_texts(self.messages)
        self.message_sequence = self.tokenizer.texts_to_sequences(self.messages)
        self.VOC_SIZE = len(self.tokenizer.word_index) + 1

    def makeSameSize(self):

        max_len = 0
        for x in self.message_sequence:
            l = len(x)
            if l > max_len:
                max_len = l

        self.max_length_sequence = max_len

        for x in self.message_sequence:
            if len(x) != max_len:
                diff = max_len - len(x)
                for i in range(diff):
                    x.insert(0, 0)

    # -----------------------------------------------------------------------------------------------------------
    def convertMessage(self, message):
        message = contractions.fix(message)
        message = message.split()
        message = self.remove_url_msg(message)
        message = self.remove_punc_msg(message)
        message = self.remove_empty_string(message)
        # message = self.remove_stop_words_msg(message)
        message = " ".join(message)
        message = self.message_to_numeric_msg(message)
        message = self.make_same_size_msg(message)
        return message

    def remove_url_msg(self, message):
        y = 0
        sentence_len = len(message)
        while y < sentence_len:
            result = re.search("^http://|^https://|^www\.", message[y])
            if result is not None:
                del message[y]
                sentence_len -= 1
            else:
                y += 1
        return message

    def remove_punc_msg(self, message):
        test_list = []
        alpha = 'a'
        for i in range(0, 26):
            test_list.append(alpha)
            alpha = chr(ord(alpha) + 1)
        alpha = 'A'
        for i in range(0, 26):
            test_list.append(alpha)
            alpha = chr(ord(alpha) + 1)

        counter = 0
        nullSTR = ""
        for y in range(len(message)):
            counter = len(message[y])
            z = 0
            nullSTR = ""
            while z < counter:
                if message[y][z] in test_list:
                    nullSTR = nullSTR + message[y][z]
                z += 1
            message[y] = nullSTR

        return message

    def remove_empty_string(self, message):
        msg_len = len(message)
        x = 0
        while x < msg_len:
            if message[x] == "" or message[x] == " ":
                del message[x]
                msg_len -= 1
            else:
                x += 1
        return message

    def to_lower_case_msg(self, message):
        for y in range(len(message)):
            message[y] = message[y].lower()
        return message

    def remove_stop_words_msg(self, message):
        message = self.to_lower_case_msg(message)
        stopWords = stopwords.words('english')
        sentenceLen = len(message)
        y = 0
        while y < sentenceLen:
            if message[y] in stopWords:
                del message[y]
                sentenceLen -= 1
            else:
                y += 1
        return message

    # TO BE EDITED BY HAISSAM-------------------
    def message_to_numeric_msg(self, message):
        m = []
        m.append(message)
        self.tokenizer.fit_on_texts(m)
        # print(self.tokenizer.word_index)
        message = self.tokenizer.texts_to_sequences(m)
        return message[0]

    def make_same_size_msg(self, message):
        if len(message) != self.max_length_sequence:
            diff = self.max_length_sequence - len(message)
            for i in range(diff):
                message.insert(0, 0)
        return message

    # ----------------------------------------------------------------------------------------------------
    def LSTMModel(self):

        # self.message_sequence = np.array(self.message_sequence)
        # self.n_label = np.array(self.n_label)

        # print(self.n_label.shape)

        self.split_train_test()
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        model = Sequential()

        feature_num = 50
        print(self.max_length_sequence)
        model.add(
            Embedding(input_dim=self.VOC_SIZE + 20, output_dim=feature_num, input_length=self.max_length_sequence))
        model.add(LSTM(units=100))
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()

        model.fit(self.X_train, self.y_train, epochs=3, batch_size=32, validation_split=0.2)

        # x=self.convertMessage("Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GBP/mnth inc 3hrs 16 stop?txtStop www.gamb.tv")
        # z=self.convertMessage("IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.")
        # zz=self.convertMessage("Congratulations! you have won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.")
        # xxa=self.convertMessage("Did I forget to tell you ? I want you , I need you, I crave you ... But most of all ... I love you my sweet Arabian steed ... Mmmmmm ... Yummy")
        # y=self.convertMessage("Hi! I am haissam")

        y_pred = model.predict(self.X_test)
        for x in range(len(y_pred)):
            if y_pred[x] < 0.5:
                y_pred[x] = 0
            else:
                y_pred[x] = 1

        from sklearn.metrics import confusion_matrix
        # Comparing the predictions against the actual observations in y_val
        cm = confusion_matrix(y_pred, self.y_test)

        def accuracy(confusion_matrix):
            diagonal_sum = confusion_matrix.trace()
            sum_of_all_elements = confusion_matrix.sum()
            return diagonal_sum / sum_of_all_elements

        print("Accuracy is: ", accuracy(cm))