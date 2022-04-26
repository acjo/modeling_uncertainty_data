import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return 

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        #get the unique words in the training data
        all_words = X.str.split().sum()
        unique_words =sorted(list(set(all_words)))
        #new dataframe containing word counts
        self.data = pd.DataFrame(0, columns=unique_words, index=['ham', 'spam'])
        #number of words in spam and ham
        self.N_counts = {'ham':sum(y == 'ham') ,
                         'spam': sum(y == 'spam')}
        #total number of words
        self.N_samples = len(y)
        #probability of spam and ham
        self.P_class = { 'ham': self.N_counts['ham'] / self.N_samples,
                         'spam': self.N_counts['spam'] / self.N_samples }
        #now we count the number of times each word occurs in spam and ham
        for SH, message in zip(y, X):
            for word in message.strip().split():
                if word == " ":
                    continue
                else:
                    #increment the word count
                    self.data.loc[SH, word] += 1


        #total number of times words (including mutliplicites) occur in spam/ham
        self.N_occurences = {'ham': sum(self.data.loc['ham']),
                             'spam': sum(self.data.loc['spam'])}
        return self

    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)
        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        #initialize probability array
        probabilities = np.zeros((X.shape[0], 2))
        #iterate through each message
        for N, message in enumerate(X):
            # get list of words and unique words
            # get words and unique words in a list
            words = message.strip().split(" ")
            unique_words = list(set(words))
            #iterate through each class
            for c, class_ in enumerate(['ham', 'spam']):
                #initialize the current probability
                current_prob = 1
                #iterate through all the words to
                for i, word in enumerate(unique_words):
                    #in the cases that the word is in the training set calculate it here
                    try:
                        #check if the current probability is zero so we can set the overall prob to
                        # zero and skip the rest of the iteration.
                        if self.data.loc[class_, word] == 0:
                            current_prob = 0
                            break
                        #otherwise iterate as usual
                        else:
                            current_prob *= (self.data.loc[class_, word] / self.N_occurences[class_])**int(words.count(word))
                    #otherwise the corresponding probabability is 1
                    except KeyError:
                        continue

                #calculate probabilities
                probabilities[N, c] = self.P_class[class_] * current_prob

        return probabilities

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        #create label max for indices
        label_map = {0:'ham', 1:'spam'}
        #get probabilities
        probabilities = self.predict_proba(X)
        #get corresponding indices
        indexes = np.argmax(probabilities, axis = 1)
        #return array of predictions.
        return np.array([label_map[index] for index in indexes])

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        tol = 1e-12

        #initialize probability array
        probabilities = np.zeros((X.shape[0], 2))
        #iterate through each message
        for N, message in enumerate(X):
            # get list of words and unique words
            # get words and unique words in a list
            words = message.strip().split(" ")
            unique_words = list(set(words))
            #iterate through each class
            for c, class_ in enumerate(['ham', 'spam']):
                #initialize the current probability
                current_prob = 0
                #iterate through all the words to
                for i, word in enumerate(unique_words):
                    #in the cases that the word is in the training set calculate it here
                    try:
                        current_prob += int(words.count(word)) * np.log( self.data.loc[class_, word] / self.N_occurences[class_] + tol)
                    #otherwise the corresponding log probabability is 0
                    except KeyError:
                        continue

                #calculate probabilities
                probabilities[N, c] = np.log(self.P_class[class_] + tol) + current_prob

        return probabilities


    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''

        #create label max for indices
        label_map = {0:'ham', 1:'spam'}
        #get probabilities
        probabilities = self.predict_log_proba(X)
        #get corresponding indices
        indexes = np.argmax(probabilities, axis = 1)
        #return array of predictions.
        return np.array([label_map[index] for index in indexes])

    def score_log(self, X, y):

        predictions = self.predict_log(X)
        mask = predictions == y
        return sum(mask) / len(mask)


class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like 
    Poisson random variables
    '''

    def __init__(self):
        return

    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to 
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        
        Returns:
            self: this is an optional method to train
        '''
        all_words = X.str.split().sum()
        unique_words =sorted(list(set(all_words)))
        #new dataframe containing word counts
        self.data = pd.DataFrame(0, columns=unique_words, index=['ham', 'spam'])
        #number of words in spam and ham
        self.N_counts = {'ham':sum(y == 'ham') ,
                         'spam': sum(y == 'spam')}
        #total number of words
        self.N_samples = len(y)
        #probability of spam and ham
        self.P_class = { 'ham': self.N_counts['ham'] / self.N_samples,
                         'spam': self.N_counts['spam'] / self.N_samples }
        #now we count the number of times each word occurs in spam and ham
        for SH, message in zip(y, X):
            for word in message.strip().split():
                if word == " ":
                    continue
                else:
                    #increment the word count
                    self.data.loc[SH, word] += 1


        #total number of times words (including mutliplicites) occur in spam/ham
        self.N_occurences = { 'ham' : sum(self.data.loc['ham']),
                             'spam' : sum(self.data.loc['spam']) }

        N_k_ham = self.N_occurences['ham']
        N_k_spam = self.N_occurences['spam']

        #initalize ham rate dicitionaries
        self.ham_rates = dict()
        self.spam_rates = dict()
        #iterate through all the unique words
        for word in unique_words:

            #get the corresponding number of occuerences.
            n_i_ham = self.data.loc[ 'ham', word ]
            n_i_spam = self.data.loc[ 'spam', word ]
            #note that if we take derivatives we see that
            #the optimizer is n_i / N_k
            self.ham_rates[word] = n_i_ham / N_k_ham
            self.spam_rates[word] = n_i_spam / N_k_spam

        self.rates = { 'ham' : self.ham_rates,
                      'spam' : self.spam_rates }

        return self

    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham or spam
                column 0 is ham, column 1 is spam 
        '''
        tol = 1e-12
        #initialize probability array
        probabilities = np.zeros((X.shape[0], 2))
        #iterate through each message
        for N, message in enumerate(X):
            # get list of words and unique words
            # get words and unique words in a list
            words = message.strip().split(" ")
            n = len(words)
            unique_words = list(set(words))
            #iterate through each class
            for c, class_ in enumerate(['ham', 'spam']):
                #initialize the current probability
                current_prob = 0
                #iterate through all the words to
                for i, word in enumerate(unique_words):
                    #in the cases that the word is in the training set calculate it here
                    try:
                        current_prob += np.log( poisson.pmf( int(words.count(word)), self.rates[class_][word] * n ) + tol )
                    #otherwise the corresponding log probabability is 0
                    except KeyError:
                        continue

                #calculate probabilities
                probabilities[N, c] = np.log( self.P_class[class_] + tol ) + current_prob

        return probabilities

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''

        #create label max for indices
        label_map = { 0: 'ham', 1 : 'spam' }
        #get probabilities
        probabilities = self.predict_proba(X)
        #get corresponding indices
        indexes = np.argmax(probabilities, axis = 1)
        #return array of predictions.
        return np.array([label_map[index] for index in indexes])



def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    #vectorize method
    VectorizeMethod = CountVectorizer()
    #get new x_train and x_test
    X_train = VectorizeMethod.fit_transform(X_train)
    X_test = VectorizeMethod.transform(X_test)
    #set the multinomial
    NB = MultinomialNB()
    #fit
    NB.fit(X_train, y_train)
    #return the prediction
    return NB.predict(X_test)



if __name__ == "__main__":

    """
    data = pd.read_csv('sms_spam_collection.csv')
    X = data['Message']
    y = data['Label']

    NBC =  NaiveBayesFilter()
    NBC.fit(X[:300], y[:300])
    print('Naive:',NBC.score(X[-300:], y[-300:]), sep='\n')
    print('Naive Log:', NBC.score_log(X[-300:], y[-300:]), sep='\n')

    PB = PoissonBayesFilter()
    PB.fit(X[:300], y[:300])
    print('Poisson:',PB.score(X[-300:], y[-300:]), sep='\n')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    print(sklearn_method(X_train, y_train, X_test))
    
    NBC.fit(X_train, y_train)
    print(NBC.score(X_test, y_test))
    print(NBC.predict(X[530:535]))
    print(NBC.predict_log(X[530:535]))
    print()

    print(NBC.predict_proba(X[[1085, 2010]]))
    print(NBC.predict(X[[1085, 2010]]))

    print()
    print(NBC.predict_log_proba(X[[1085, 2010]]))
    print(NBC.predict_log(X[[1085, 2010]]))
    
    Z = NBC.predict(X[530:535])
    print(Z)
    0.9581339712918661
    print(NBC.score(X_test, y_test))
    print(X[300:305])
    print(NBC.data.sample(2))
    y = sum(NBC.data.iloc[0].value_counts().iloc[1:])

    print()
    x = sum(NBC.data.iloc[1].value_counts().iloc[1:])

    print(x + y)
    
    print(PB.score(X[-300:], y[-300:]))

    PB = PoissonBayesFilter()
    PB.fit(X_train, y_train)
    print(PB.score(X_test, y_test))
    print(PB.predict_proba(X[530:535]))
    
    
    PB = PoissonBayesFilter()
    PB.fit(X[:300], y[:300])
    print(PB.score(X[-300:], y[-300:]))
    """


