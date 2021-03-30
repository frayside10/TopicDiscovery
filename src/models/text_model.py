import logging
import warnings
import gensim

def gen_lda_model(corpus, id2word, model_log, model_data):
    """ Constructor that estimates Latent Dirichlet Allocation model parameters
    based on the supplied training corpus"""
    logging.basicConfig(filename=model_log, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train4 = gensim.models.ldamulticore.LdaMulticore(
                               corpus=corpus,
                               num_topics=20,
                               id2word=id2word,
                               chunksize=100,
                               workers=2, # Num. Processing Cores - 1
                               passes=10,
                               #eval_every = 1,
                               per_word_topics=True)
        lda_train4.save(model_data)
        return lda_train4

def train_vectors(lda_model, train_corp, text_train_df):
    """Create a feature vector for every review"""
    train_vecs = []
    for i in range(len(text_train)):
        top_topics = lda_model.get_document_topics(train_corp[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(10)]
        topic_vec.extend([len(text_train_df.iloc[i].text)]) # length review
        train_vecs.append(topic_vec)
    return train_vecs

