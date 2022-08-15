import datetime
import requests
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json 
import networkx
import os
import spacy
from nlp_helper import *
from pdfminer.high_level import extract_text
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm")

def front_page_build_tables(con):
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS Front_Page_text (id INTEGER PRIMARY KEY AUTOINCREMENT, front_page_text TEXT, date_yyyy_mm_dd TEXT, slug TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Front_Page_vader (front_page_text_id INTEGER, sediment_neg REAL, sediment_neu REAL, sediment_pos REAL, sediment_comp REAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Front_Page_bag (front_page_text_id INTEGER, bag_a_words_json BLOB, Tfidf_json BLOB, phrase_list BLOB)""")
    cur.execute("""CREATE VIEW IF NOT EXISTS Front_Page_Report
                AS
                SELECT a.slug, a.date_yyyy_mm_dd, b.sediment_comp, b.sediment_neg, b.sediment_neu, b.sediment_pos, c.bag_a_words_json, c.Tfidf_json
                    FROM Front_Page_text a
                    JOIN Front_Page_vader b ON a.id = b.front_page_text_id
                    JOIN Front_Page_bag c ON a.id = c.front_page_text_id""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Front_Page_nea (front_page_text_id INT, nea_text TEXT, start_char TEXT, end_char TEXT, label TEXT)""") 
    cur.execute("""CREATE TABLE IF NOT EXISTS Front_Page_slug (slug TEXT)""")
    con.commit()
    df = pd.read_json('base_urls.json',orient='index')
    entries = df['entries'][0]
    for i in entries:
        file_name = i['request']['url'].split('/')[-1].replace('.jpg','')
        cur.execute(f"""INSERT INTO Front_Page_slug (slug) VALUES ('{file_name}')""")
        con.commit()
    cur.execute(f"""DELETE FROM Front_Page_slug
                    WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM Front_Page_slug 
                    GROUP BY slug
                    )
                    """)
    con.commit()
    

def run_front_page(con):
    day = datetime.datetime.today()
    cur = con.cursor()
    slugs = pd.read_sql_query("""SELECT * FROM Front_Page_slug""",con)
    slugs = slugs['slug'].tolist()
    for slug in slugs:
        url_temp =  f'https://cdn.freedomforum.org/dfp/pdf{day.day}/{slug}.pdf'
        r = requests.get(url_temp)
        if r.status_code == 200:
            with open(f'{slug}.pdf', 'wb') as f:
                f.write(r.content)
            text = extract_text(f'{slug}.pdf')
            os.remove(f'{slug}.pdf')
            cur.execute(f"""INSERT INTO Front_Page_text (front_page_text, date_yyyy_mm_dd, slug) VALUES ('{text.replace("'","''")}', DATE(), '{slug}')""")
            con.commit()

def text_vadar_nea(con):
    cur = con.cursor()
    id_list = pd.read_sql_query('SELECT * FROM Front_Page_text ',con)
    id_list = id_list['id'].tolist()
    for idnum in id_list:
        try:
            row = pd.read_sql_query(f"SELECT * FROM Front_Page_text WHERE id = {idnum} ",con)
            idx = row['id'].values[0]
            text = row['front_page_text'].values[0]
            text_nlp = nlp(row['front_page_text'].values[0]) 
            p_tag_block_t = processed_feature(text)
            p_tag_block_t = remove_stop_words(p_tag_block_t)
            p_tag_block=[]
            stuborn_words = ['com','city','state','page','county','fi','national','cid','year','month','day','week']
            for w in p_tag_block_t:
                if w.lower() not in stuborn_words:
                    p_tag_block.append(w)        
            p_vocab = create_bag_of_words(' '.join(p_tag_block))
            p_Tfidf = Tfidf(p_tag_block)
            p_phrases = create_bag_of_phrases([text], n_gram_range=(3,5), stop_words=stop_words)
            dict_Tfidf = {key:value for key, value in p_Tfidf}
            dict_Tfidf = dict(sorted(dict_Tfidf.items(),key = lambda x:x[1], reverse = True))
            avg = sum(dict_Tfidf.values())/len(dict_Tfidf.keys())
            dict_Tfidf_insert = {}
            for key, value in dict_Tfidf.items():
                if value > avg:
                    dict_Tfidf_insert[key] =value 
            dict_vocab_insert = dict(nltk.FreqDist(p_vocab).most_common(len(dict_Tfidf_insert.keys())))
            vadar_out = vadar(' '.join(p_tag_block))
            cur.execute(f"""INSERT INTO Front_Page_vader (front_page_text_id , sediment_neg , sediment_neu , sediment_pos , sediment_comp ) VALUES ({idx},{vadar_out['pos']},{vadar_out['neu']},{vadar_out['neg']},{vadar_out['comp']})""")
            cur.execute(f"""INSERT INTO Front_Page_bag (front_page_text_id , bag_a_words_json , Tfidf_json, Phrase_List ) VALUES ({idx},'{json.dumps(dict_vocab_insert)}','{json.dumps(dict_Tfidf_insert)}','{','.join(p_phrases)}')""")
            con.commit()
            for ent in text_nlp.ents:
                cur.execute(f"""INSERT INTO Front_Page_nea VALUES ({idx}, '{ent.text.replace("'", "''")}', '{ent.start_char}', '{ent.end_char}', '{ent.label_}')""")
                con.commit()
        except Exception as e:
            continue