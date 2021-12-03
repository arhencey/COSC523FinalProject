import pandas as pd
import numpy as np
import textstat
import spacy

# Feature engineering found at https://www.kaggle.com/pdan93/clrp-features-only-model
nlp = spacy.load('en_core_web_sm')

def get_words(text):
    text = textstat.remove_punctuation(text)
    return text.split()
def long_words(text):
    count = 0
    for w in get_words(text):
        if len(w)>6:
            count += 1
    return count
def difficult_words(text):
    count = 0
    for w in get_words(text):
        if textstat.is_difficult_word(w):
            count += 1
    return count
def get_pronouns(text, doc):
    pronouns = []
    for sent in doc.sents:
        count = 0
        for token in sent:
            if token.pos_ == "PRON":
                count += 1
        pronouns.append(count)
    return np.mean(pronouns)
def get_lexical_diversity(text):
    words = get_words(text)
    unique_words = []
    for w in words:
        if w not in unique_words:
              unique_words.append(w)
    if len(unique_words)>0 and len(words)>0:
        return len(unique_words)/len(words)
    return 0
def content_diversity(text, doc):
    words = get_words(text)
    content_words = 0
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "ADV":
            content_words += 1
    if content_words>0 and len(words)>0:
        return content_words/len(words)
    return 0
def word_incidence(text, pos, doc):
    words = get_words(text)
    nr = 0
    for token in doc:
        if token.pos_ == pos:
            nr += 1
    if nr>0 and len(words)>0:
        return nr/(len(words)/1000)
    return 0

def add_df_features(df):
    df['flesch_reading_ease'] = -1
    df['smog_index'] = -1
    df['flesch_kincaid_grade'] = -1
    df['coleman_liau_index'] = -1
    df['automated_readability_index'] = -1
    df['dale_chall_readability_score'] = -1
    df['textstat_difficult_words'] = -1
    df['linsear_write_formula'] = -1
    df['gunning_fog'] = -1
    df['text_standard'] = -1
    df['avg_character_per_word'] = -1
    df['avg_letter_per_word'] = -1
    df['avg_sentence_length'] = -1
    df['avg_sentence_per_word'] = -1
    df['avg_syllables_per_word'] = -1
    df['rix'] = -1
    df['lix'] = -1
    df['lexicon_count'] = -1
    df['long_words'] = -1
    df['difficult_words'] = -1
    df['get_pronouns'] = -1
    df['get_lexical_diversity'] = -1
    df['content_diversity'] = -1
    df['word_incidence_adj'] = -1
    df['word_incidence_adv'] = -1
    df['word_incidence_noun'] = -1
    df['word_incidence_pron'] = -1
    df['word_incidence_verb'] = -1
    for idx, row in df.iterrows():
        df.loc[idx, 'flesch_reading_ease'] = textstat.flesch_reading_ease(row['excerpt'])
        df.loc[idx, 'smog_index'] = textstat.smog_index(row['excerpt'])
        df.loc[idx, 'flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(row['excerpt'])
        df.loc[idx, 'coleman_liau_index'] = textstat.coleman_liau_index(row['excerpt'])
        df.loc[idx, 'automated_readability_index'] = textstat.automated_readability_index(row['excerpt'])
        df.loc[idx, 'dale_chall_readability_score'] = textstat.dale_chall_readability_score(row['excerpt'])
        df.loc[idx, 'textstat_difficult_words'] = textstat.difficult_words(row['excerpt'])
        df.loc[idx, 'linsear_write_formula'] = textstat.linsear_write_formula(row['excerpt'])
        df.loc[idx, 'gunning_fog'] = textstat.gunning_fog(row['excerpt'])
        df.loc[idx, 'text_standard'] = textstat.text_standard(row['excerpt'], float_output=True)
        df.loc[idx, 'avg_character_per_word'] = textstat.avg_character_per_word(row['excerpt'])
        df.loc[idx, 'avg_letter_per_word'] = textstat.avg_letter_per_word(row['excerpt'])
        df.loc[idx, 'avg_sentence_length'] = textstat.avg_sentence_length(row['excerpt'])
        df.loc[idx, 'avg_sentence_per_word'] = textstat.avg_sentence_per_word(row['excerpt'])
        df.loc[idx, 'avg_syllables_per_word'] = textstat.avg_syllables_per_word(row['excerpt'])
        df.loc[idx, 'rix'] = textstat.rix(row['excerpt'])
        df.loc[idx, 'lix'] = textstat.lix(row['excerpt'])
        df.loc[idx, 'lexicon_count'] = textstat.lexicon_count(row['excerpt'])
        df.loc[idx, 'long_words'] = long_words(row['excerpt'])
        df.loc[idx, 'difficult_words'] = difficult_words(row['excerpt'])
        doc = nlp(row['excerpt'])
        df.loc[idx, 'get_pronouns'] = get_pronouns(row['excerpt'], doc)
        df.loc[idx, 'get_lexical_diversity'] = get_lexical_diversity(row['excerpt'])
        df.loc[idx, 'content_diversity'] = content_diversity(row['excerpt'], doc)
        df.loc[idx, 'word_incidence_adj'] = word_incidence(row['excerpt'], 'ADJ', doc)
        df.loc[idx, 'word_incidence_adv'] = word_incidence(row['excerpt'], 'ADV', doc)
        df.loc[idx, 'word_incidence_noun'] = word_incidence(row['excerpt'], 'NOUN', doc)
        df.loc[idx, 'word_incidence_pron'] = word_incidence(row['excerpt'], 'PRON', doc)
        df.loc[idx, 'word_incidence_verb'] = word_incidence(row['excerpt'], 'VERB', doc)

def get_feature_data():
    # read in dataset
    training_data = pd.read_csv('data/train.csv')
    #training_data.drop(columns=['id', 'excerpt', 'url_legal', 'license'], inplace=True)
    testing_data = pd.read_csv('data/test.csv')
    #testing_data.drop(columns=['id', 'url_legal', 'license'], inplace=True)

    # add the CLRP features
    add_df_features(training_data)
    add_df_features(testing_data)

    # drop unnecessary columns
    training_data.drop(columns=['id', 'excerpt', 'url_legal', 'license'], inplace=True)
    testing_data.drop(columns=['id', 'url_legal', 'license'], inplace=True)

    # read in the pre-calculated features and drop unnecessary columns
    cleanedFeatures = pd.read_csv('cleanedfeatures.csv')
    cleanedFeatures.drop(columns=['Unnamed: 0'], inplace=True)

    # concatenate the dataframes
    training_data = pd.concat([training_data, cleanedFeatures], axis=1)

    # normalize the data along the columns
    training_data = (training_data - training_data.min())/(training_data.max() - training_data.min())

    y_train = training_data[["target"]].to_numpy()
    training_data.drop(columns=['target'], inplace=True)
    X_train = training_data.to_numpy()

    return X_train, y_train

def get_excerpt_data():
    # read in dataset and drop unnecessary columns
    training_data = pd.read_csv('data/train.csv')
    #training_data.drop(columns=['id', 'excerpt', 'url_legal', 'license'], inplace=True)
    testing_data = pd.read_csv('data/test.csv')
    #testing_data.drop(columns=['id', 'url_legal', 'license'], inplace=True)

    # normalize the target data
    training_data["target"] = (training_data["target"] - training_data["target"].min())/(training_data["target"].max() - training_data["target"].min())

    X_train = training_data[["excerpt"]].to_numpy()
    y_train = training_data[["target"]].to_numpy()

    return X_train, y_train
