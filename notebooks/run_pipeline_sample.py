#!/usr/bin/env python
# coding: utf-8
# Notebook to drive the LDA model creation on the unsupervised parts of this project, based on refactored functions, rather than everything being crowded into one notebook.

import os
import sys
src_dir = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_dir)

import pandas as pd
import numpy as np
from sklearn.datasets import load_files

from datacode.retrieve_data import pull_data
from datacode.download_data import download_file, unzip_data
from datacode.retrieve_data import pull_data

from nltk.corpus import stopwords
import nltk; nltk.download('stopwords')
import gensim.corpora as corpora
import spacy
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim

from features.pre_process import set_stop_words, sent_to_words, remove_stopwords, bigrams, get_corpus, get_test_bigram
import pickle
import warnings
from models.text_model import gen_lda_model, train_vectors
import pyLDAvis.gensim


from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

root_dir = os.path.dirname(os.path.dirname(os.path.abspath("LICENSE")))
interim_data_path = os.path.join(root_dir, "data/interim")
processed_data_path = os.path.join(root_dir, "data/processed")
raw_data_path = os.path.join(root_dir, "data/raw")
raw_data_loc = os.path.join(raw_data_path, "imdb_raw.tar.gz")
topics_list_output = os.path.join(processed_data_path, "found_topics.csv")

source_file_ulr = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
imbd_unpack_dir = "../data/raw"

corpus_path = os.path.join(interim_data_path, "train_corpus4.pkl")
id2word_path = os.path.join(interim_data_path, "train_id2word4.pkl")
bigram_train_path = os.path.join(interim_data_path, "bigram_train4.pkl")

model_log = os.path.join(processed_data_path, "logs/lda_model.log")
model_data = os.path.join(processed_data_path, "model_data/lda_train4.model")

# Download the source data and unzip
download_file(source_file_ulr, raw_data_loc)
unzip_data(raw_data_loc, imbd_unpack_dir)

# Pull in training data, split off features and label data and save the features array as a df
imdb_train = pull_data(os.path.join(raw_data_path, "aclImdb/train"))
text_train, y_train = imdb_train.data, imdb_train.target
text_train_df = pd.DataFrame({'text': text_train})


# Better to add to a YAML or other text file in due course
added_stop_words = ['film','films','movie','picture','review','watch','movies','see',
                    'xc','seems','think','would','could','get', 'however','people','many',
                    'us','jane','also','jones','know','even','great','good','bad','poor',
                   'terrible','awful','stink','brilliant','lame','stupid','loved','hate','hated',
                   'enjoy','enjoyed','garbage','really','best','wonderful','much','make','well','man',
                   'woman','much','actually','little','small','guess','never','always', 'joy',
                   'love','english','french','quite','beautiful','hit','joe','james','adam','crap',
                   'worst','best','jesus']


# # SAMPLE A SUBSET JUST TO GET IT TO RUN THROUGH
text_train_df_samp = text_train_df.sample(frac=0.05).copy()


# Returns the following
# train_id2word4: Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for debugging and topic printing.
# train_corpus4: Stream of document vectors or sparse matrix 
# bigram_train4: Grouping of related phrases i.e. sci fi is converted to sci_fi 
#train_corpus4, train_id2word4, bigram_train4 = get_corpus(text_train_df, added_stop_words) 

# RUN ON SAMPLE
train_corpus4, train_id2word4, bigram_train4 = get_corpus(text_train_df_samp, added_stop_words) 

# Keep the files
# with open(corpus_path, 'wb') as f:
#     pickle.dump(train_corpus4, f)
# with open(id2word_path, 'wb') as f:
#     pickle.dump(train_id2word4, f)
# with open(bigram_train_path, 'wb') as f:
#     pickle.dump(bigram_train4, f)
# Create the LDA model - i.e. create the topics
lda_train4 = gen_lda_model(train_corpus4, train_id2word4, model_log, model_data)

topic_list = []
for topic in lda_train4.show_topics(num_topics=20, num_words=10, log=False, formatted=True):
    topic_list.append([train_id2word4[id[0]] for id in lda_train4.get_topic_terms(topic[0])])

topic_list_df = pd.DataFrame({'topics': topic_list})

print(topic_list_df)

topic_list_df.to_csv(topics_list_output)

