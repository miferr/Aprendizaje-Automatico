#   Librerias
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from codecs import open
import get_vocab_dict as vodi
import process_email as poem
from sklearn.model_selection import train_test_split


#   Numero de correos por carpeta
n_spam = 500
n_easy_ham = 2551
n_hard_ham = 250

dic = vodi.getVocabDict()
n_words = len(dic)

def parse_name(number):
    while(len(str(number)) < 4):
          number = '0' + str(number)
    return str(number)

def create_X(n_emails, name_folder):
    X = np.array
    for i in range(1, n_emails + 1):
        email_contents = open(name_folder+'/'+parse_name(i)+'.txt', 'r', encoding='utf-8', errors='ignore').read()
        print(name_folder+'/'+parse_name(i)+'.txt')
        email = poem.email2TokenList(email_contents)
        Xn = np.zeros(n_words)
        for word in email:
            if word in dic:
                Xn[dic[word] - 1] = 1
        if i == 1:
            X = Xn
        else:
            X = np.vstack([X, Xn])
    return X
print(create_X(n_spam, 'spam').shape)

X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split( create_X(n_easy_ham, "easy_ham"), np.zeros(n_spam), test_size=0.2, random_state=0)
X_train_aux.shape