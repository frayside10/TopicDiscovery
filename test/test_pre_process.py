from src.features.pre_process import set_stop_words, sent_to_words, remove_stopwords, bigrams

import pytest


def test_set_stop_words(stop_word_list):
    result = set_stop_words(stop_word_list)
    # Check if extra words appear in the stop list
    assert all(w for w in result if w in stop_word_list)


def test_sent_to_words(raw_words_for_tokenizing):
    result = list(sent_to_words(raw_words_for_tokenizing.text))

    # Check tokens are all lower case
    assert [str(w).islower() for w in result]


#def test_remove_stopwords(stop_word_list, word_list):
#    result = remove_stopwords(stop_list)
    # not complete


# def test_bigrams(raw_words_for_bigram):
#     b = bigrams(raw_words_for_bigram)
#     b = [b[word] for word in raw_words_for_bigram]
#
#     # Fixture input should be coverted into 'sci_fi' from 'sci fi'
#     assert 'sci_fi' in b[0]


