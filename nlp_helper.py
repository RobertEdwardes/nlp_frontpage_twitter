import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')
pos_to_lemmatize={'NN':'n','NNS':'n','NNP':'n','NPPS':'n','WP':'n','WP$':'n',
                 'VB':'v','VBD':'v','VBG':'v','VBN':'v','VBP':'v','VBZ':'v',
                 'JJ':'a','JJR':'a','JJS':'a',
                 'RB':'r','RBR':'r','RBS':'r','WRB':'r'}

stop_words = set(nltk.corpus.stopwords.words("english"))

def remove_stop_words(text, stop_words=stop_words):
    tokenized_word = nltk.tokenize.word_tokenize(text)
    filtered_sent=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

def unescapematch(matchobj):
     escapesequence = matchobj.group(0)
     digits = escapesequence[2:]
     ordinal = int(digits, 16)
     char = chr(ordinal)
     return char

def processed_feature(text, pos_clean=True):
    
    # Removing URLS
    processed_feature = re.sub(r'https?:\S+', '', text)
    processed_feature = re.sub(r'^(.+?).\s', '', processed_feature)
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', processed_feature)
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', ' ', processed_feature)
    # Remove unicode
    processed_feature = re.sub(r'(\\u[0-9A-Fa-f]+)', unescapematch, processed_feature)
    # Converting to Lowercase remove RT for retweets
    processed_feature = processed_feature.lower().replace('RT ','')
    # Remove paper page Numbers
    processed_feature = re.sub(r'[a-zA-Z][0-9]', '', processed_feature)
    processed_feature = re.sub(r'[0-9][a-zA-Z]', '', processed_feature)
    # Remove numbers
    processed_feature = re.sub(r'[\d.]*\d+', '', processed_feature)
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', '', processed_feature)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', '', processed_feature) 
    if pos_clean:
        h=[]
        tokens = nltk.word_tokenize(text)
        tag = nltk.pos_tag(tokens)
        #Remove words with unimportant Part of Speech
        [h.append(word) for (word,pos) in tag if pos!='VB' and pos!='VBD'and pos!='VBG'and pos!='VBN'and 
        pos!='VBP' and pos!='RB' and pos!='RBR' and pos!='RBS' and pos!='WRB' and pos!='DT' and pos!='TO'
        and pos!='MO'and pos!='VBZ' and pos!='CC'and pos!='IN' and pos!='FW' and pos!='RP' and pos!='PRP$' 
        and pos!='PRP' and pos not in ['JJ','JJR','JJS'] and pos!='CD']
        text = ' '.join(h)
    return text

def Tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    transformed_data = vectorizer.fit_transform(text)
    return zip(vectorizer.get_feature_names_out(), np.ravel(transformed_data.sum(axis=0)))

def create_bag_of_words(text, lemmatize=True):
    filtered_sent=remove_stop_words(text)
    pos_taged = nltk.pos_tag(filtered_sent)
    ps  = nltk.stem.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    bag_of_words = []
    if lemmatize:
        for tag in pos_taged:
            if tag[1] in pos_to_lemmatize:
                bag_of_words.append(lem.lemmatize(tag[0],pos_to_lemmatize[tag[1]]))
            else:
                bag_of_words.append(ps.stem(tag[0]))
    else:
        for w in filtered_sent:
            bag_of_words.append(ps.stem(w))
    return bag_of_words

def create_bag_of_phrases(text, n_gram_range=(2,3), stop_words=stop_words):
    count = CountVectorizer(ngram_range=n_gram_range,stop_words=stop_words).fit(text)
    bag_of_phrases = count.get_feature_names_out()
    bag_of_phrases_out = []
    for i in range(len(bag_of_phrases)):
        if text[0].lower().find(bag_of_phrases[i]) > 0:
            tokens = nltk.word_tokenize(bag_of_phrases[i])
            tag = nltk.pos_tag(tokens)
            if tag[-1][1] != 'JJ' and tag[-1][1] != 'JJR'and tag[-1][1] != 'JJS': 
                bag_of_phrases_out.append(bag_of_phrases[i])
    return bag_of_phrases

def vadar(text):
    out_put={'neg':None,
             'neu':None,
             'pos':None,
             'comp':None}
    sia = SentimentIntensityAnalyzer()
    out_put['neg'] = sia.polarity_scores(text)['neg']
    out_put['neu'] = sia.polarity_scores(text)['neu']
    out_put['pos'] = sia.polarity_scores(text)['pos']
    out_put['comp'] = sia.polarity_scores(text)['compound']
    return out_put

def cos_similarity(textlist):
    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()