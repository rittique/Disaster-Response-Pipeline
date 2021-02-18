# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:03:49 2021

@author: Tino Riebe

"""
# Basics
import os
import pandas as pd
from sqlalchemy import create_engine
import re
import sys 


# Language Toolit
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# klearn Libraries for Ml Models
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib as jbl

# sklearn classifier 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore")

# database_filepath = 'e:/github/desaster/data/DisasterResponse.db'
# model_filepath = 'e:/github/desaster/models'

def load_data(database_filepath):
    '''
    INPUT  
    database_filepath(string) - DisasterResponse.db 
        
    OUTPUT
    X - features the datset
    y - target of the dataset
    categories - the name of the different targets
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster_messages', engine)
    engine.dispose()
    
    X = df['message']
    y = df[df.columns[4:]]
    categories = y.columns.tolist()

    return X, y, categories


def tokenize(text):
    """
    INPUT
    text - messages column from the table
    
    OUTPUT
    lemmed - tokenized text after performing below actions
    
    1. Remove Punctuation and normalize text
    2. Tokenize text and remove stop words
    3. Use stemmer and Lemmatizer to Reduce words to its root form
    """
    # Remove Punctuations and normalize text by converting text into lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text and remove stop words
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    #Reduce words to its stem/Root form, Lemmatize the words
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmed]
    
    return lemmed


def build_basic_pipelines():
    """
    INPUT
    clf_name - name of a classifier
    
    OUTPUT
    pipeline - pipeline with the basic classifier
    """
    clf_choice = 0
    clf_dict={1:'RandomForest', 2:'KNeighbors',3:'GradientBoosting',
              4:'DecisionTree', 5:'AdaBoost', 6:'SGD',
              7:'MultinominalNB', 8:'SVC'}
    
    output='Available Classifier\n\n1 - RandomForest\n2 - KNeighbors\n\
3 - GradientBososting\n4 - DecisionTree\n5 - AdaBoost\n6 - SGD\n\
7 - MultinominalGB\n8 - SVC'
   
    
    
    while clf_choice < 1 or clf_choice > 8:
        print(output)
        try:
            clf_choice = int(input('Your Choice (1-8) is: '))
            if clf_choice < 1 or clf_choice > 8:
                print('wrong input, only 1 - 8\n')
        except:
                print('wrong input, only 1 - 8\n')
    else:
        print(clf_dict[clf_choice], '- pipeline will be prepared')
    
    if clf_choice==1:
        clf = RandomForestClassifier()
    elif clf_choice==2:
        clf = KNeighborsClassifier()
    elif clf_choice==3:
        clf = GradientBoostingClassifier()
    elif clf_choice==4:
        clf = DecisionTreeClassifier()
    elif clf_choice==5:
        clf = AdaBoostClassifier()
    elif clf_choice==6:
        clf = SGDClassifier()
    elif clf_choice==7:
        clf = MultinomialNB()
    elif clf_choice==8:
        clf = SVC()
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf))
        ])
    
    return pipeline


def build_model():
    """
    INPUT
    None - because we will use the selected classifier with predifined
           parameters
    
    OUTPUT
    pipeline - pipeline with the adjusted classifier
    """
    # The pipeline has tfidf, dimensionality reduction, and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(loss='modified_huber',
                                                    penalty='elasticnet',
                                                    alpha = 0.0001,
                                                    n_jobs=-1))),
        ])

    return pipeline   

    '''
    # GridsearchPipeline for modeltraining, already done
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('SGD', MultiOutputClassifier(SGDClassifier()))
        ])

    grid = {'vect__ngram_range': [(1, 1), (1, 2)],
            'SGD__estimator__loss': ['modified_huber'],
            'SGD__estimator__penalty': ['elasticnet','l2'],
            'SGD__estimator__alpha': [0.0001, 0.001],
            'SGD__estimator__n_jobs': [-1]
            }

    model = GridSearchCV(pipeline2, grid, cv=2)
    model.fit(X_train, y_train)
    return model
    '''


def evaluate_model(model, X_test, y_test, categories):
    """
    INPUT:
    model: ml -  model
    X_text: The X test set
    y_test: the y test classifications
    category_names: the category names
    
    OUTPUT:
    None
    """
        
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=categories))


def save_model(model, model_filepath):
    """
    INPUT: 
    model - the fitted model
    model_filepath (str) -  filepath to save model
    
    OUTPUT:
    None
    """
    
    
    jbl.dump(model, model_filepath)
    print('modell is saved:', model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)
        print('\nChoose:\n\nTrain with a basic classifer (1)\nTrain with the predefined classifier(2)?')
        choice=0
        
        while choice != 1 and choice != 2:
            try:
                choice = int(input('Your Choice is: '))
                if choice != 1 and choice != 2:
                    print('wrong input, only (1) or (2)')
                    
            except:
                print('wrong input, only (1) or (2)')
        
        
        print('Building model...')
        if choice ==1:
            model = build_basic_pipelines()
        else: 
            model = build_model()
            
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
