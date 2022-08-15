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
import time
import random as rnd

def init_users(con, headers):
    base_handels = {'NRSC':'NRSC',
        'DSCC':'DSCC',
        'DCCC':'DCCC',
        'NRCC':'NRCC',
        'RNC' :'GOP',
        'DNC':'DNC',
        'Twitter Verified':'verified'}
    base_handels = ','.join(base_handels.values())
    BASE_URL_QUERY = f'https://api.twitter.com/2/users/by?usernames={base_handels}&user.fields=created_at&expansions=pinned_tweet_id&tweet.fields=author_id,created_at'
    auth_response_QUERY = requests.get(BASE_URL_QUERY,  headers=headers)
    auth_response_RESPONSE = json.loads(auth_response_QUERY.text)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS twitter_base_users (user_id INTEGER, user_name TEXT, category TEXT)
    """)
    con.commit()
    for i in auth_response_RESPONSE['data']:
        cur.execute(f"""INSERT INTO twitter_base_users (user_id, user_name) VALUES ({i['id']},'{i['username']}')""")
        con.commit()
    groups = {'v':'verified', 'd':['DNC', 'dccc', 'dscc'], 'r':['NRSC', 'NRCC', 'GOP']}
    for key, value in groups.items():
        if isinstance(value,list):
            value = "','".join([f'{i}' for i in value])
        cur.execute(f"""UPDATE twitter_base_users SET category = '{key}' WHERE user_name IN ('{value}')""")
        con.commit()

def get_follows(base_id, con, headers, next_token=None):
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS twitter_base_users (user_id INTEGER, user_name TEXT, parent_id INTEGER)
    """)
    con.commit()
    retry = True 
    while retry:   
        if next_token:
            BASE_URL_QUERY = f'https://api.twitter.com/2/users/{base_id}/following?user.fields=created_at&expansions=pinned_tweet_id&tweet.fields=created_at&max_results=1000&pagination_token={next_token}'
        else:
            BASE_URL_QUERY = f'https://api.twitter.com/2/users/{base_id}/following?user.fields=created_at&expansions=pinned_tweet_id&tweet.fields=created_at&max_results=1000'
        auth_response_QUERY = requests.get(BASE_URL_QUERY,  headers=headers)
        auth_response_RESPONSE = json.loads(auth_response_QUERY.text)
        if auth_response_QUERY.status_code == 429:
            time.sleep(900)
            auth_response_QUERY = requests.get(BASE_URL_QUERY,  headers=headers)
            auth_response_RESPONSE = json.loads(auth_response_QUERY.text)
        if auth_response_RESPONSE['data']:
            for i in auth_response_RESPONSE['data']:
                cur.execute(f"""INSERT INTO twitter_users (user_id, user_name, parent_id) VALUES ({i['id']},'{i['username']}', {base_id})""")
                con.commit()
        if 'next_token' in auth_response_RESPONSE['meta'].keys():
            get_follows(base_id, con, headers, auth_response_RESPONSE['meta']['next_token'])
            retry = False
        else:
            retry = False
    
def twitter_build_tables(con):
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_forgien_lang_user (lang TEXT, user_id INT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_noAcesses_user (user_id INT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_to_user (tweet_id INT, user_id INT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet (tweet_id INT, conersation_id INT, text TEXT, likes INT, retweets INT, replay INT, quote INT, created TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_nea (tweet_id INT, text TEXT, entity TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_mention (tweet_id INT, mention_id INT, mention_name TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_hash_tags (tweet_id INT, hash_tags TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_urls (tweet_id INT, url TEXT, title TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_conText_entity (tweet_id INT, entity_id INT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_conText_domain (tweet_id INT, domain_id INT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS twitter_tweet_conText_def (id INT, name TEXT, description TEXT, type TEXT, PRIMARY KEY(id,type)) ''')
    con.commit()

def get_tweet_data(con, headers):
    cur = con.cursor()
    groups = pd.read_sql_query("SELECT category FROM twitter_base_users",con)
    groups = list(set(groups['category'].tolist()))
    date = datetime.datetime.now() - datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%dT00:00:00Z") 
    for group in groups:
        limit = 300
        user_ids_list = pd.read_sql_query(f"""SELECT a.user_id
                                            FROM twitter_users a
                                            JOIN twitter_base_users b ON a.parent_id = b.user_id
                                            WHERE b.category = '{group}' AND a.user_id NOT IN (SELECT user_id FROM twitter_noAcesses_user)
                                            ORDER BY RANDOM()
                                            LIMIT {limit}
                                            """,con)
        sample_set = list(set(user_ids_list['user_id'].tolist()))
        
        for s_id in sample_set:
            try:
                BASE_URL = f'https://api.twitter.com/2/users/{s_id}/tweets?max_results=100&start_time={date}&tweet.fields=id,author_id,context_annotations,conversation_id,created_at,entities,lang,public_metrics,text'
                auth_response_QUERY = requests.get(BASE_URL,  headers=headers)
                auth_response_RESPONSE = json.loads(auth_response_QUERY.text)
                if auth_response_RESPONSE['meta']['result_count'] > 0:
                    for j in auth_response_RESPONSE['data']:
                        input_dict = {
                        'lang' : None,
                        'hashtags' : [],
                        'url' : [],
                        'title' : [],
                        'tweet_domain' : [],
                        'tweet_entity' : [],
                        'tweet_mentions' : [],
                        'conversation_id' : None,
                        'tweet_id' : j['id'],
                        'text' : j['text'].replace("'","''"),
                        'author_id' : j['author_id'],
                        'created_at' : j['created_at'],
                        'public_metrics': j['public_metrics'],
                        'nea' : []
                        }
                        if 'lang' in list(j.keys()):
                            input_dict['lang'] = j['lang']
                            if input_dict['lang'] != 'en':
                                cur.execute(f"DELETE FROM twitter_users WHERE user_id = {input_dict['author_id']}")
                                cur.execute(f"INSERT INTO twitter_forgien_lang_user (lang, user_id) VALUES ('{input_dict['lang']}', {input_dict['author_id']}) ")
                                con.commit()
                                continue
                        if 'hashtags' in list(j['entities'].keys()):
                            for i in j['entities']['hashtags']:
                                hash_query = f"INSERT INTO twitter_tweet_hash_tags VALUES ({input_dict['tweet_id']}, '{i['tag']}')"
                                cur.execute(hash_query)
                                con.commit()
                        if 'urls' in list(j['entities'].keys()):
                            for i in j['entities']['urls']:
                                if 'unwound_url' in list(i.keys()):
                                    url_query = f"INSERT INTO twitter_tweet_urls VALUES ({input_dict['tweet_id']}, '{i['unwound_url']}', '{i['title']}')"
                                    cur.execute(url_query)
                                    con.commit()
                        if 'annotations' in list(j['entities'].keys()):
                            for i in j['entities']['annotations']:
                                nea_query = f"INSERT INTO twitter_nea VALUES ({input_dict['tweet_id']}, '{i['normalized_text']}', '{i['type']}')"
                                cur.execute(nea_query)
                                con.commit()
                        if 'context_annotations' in list(j.keys()):
                            for i in j['context_annotations']:
                                d_name = i['domain']['name'].replace("'","''")
                                d_des = i['domain']['description'].replace("'","''")
                                conText_d = f"INSERT INTO twitter_tweet_conText_domain VALUES ({input_dict['tweet_id']}, {i['domain']['id']})"
                                conDef_d = f"INSERT INTO twitter_tweet_conText_def VALUES ({i['domain']['id']}, '{d_name}', '{d_des}', 'domain') ON CONFLICT(id,type) DO NOTHING"
                                conText_e = f"INSERT INTO twitter_tweet_conText_entity VALUES ({input_dict['tweet_id']}, {i['entity']['id']})"
                                e_name = i['entity']['name'].replace("'","''")
                                if 'description' not in  list(i.keys()):
                                    e_des=  None
                                else:
                                    e_des = i['entity']['description'].replace("'","''")
                                conDef_e = f"INSERT INTO twitter_tweet_conText_def VALUES ({i['entity']['id']}, '{e_name}', '{e_des}', 'entity') ON CONFLICT(id,type) DO NOTHING"
                                cur.execute(conText_d)
                                cur.execute(conDef_d)
                                cur.execute(conText_e)
                                cur.execute(conDef_e)
                                con.commit()
                        if 'mentions' in list(j['entities'].keys()):
                            for i in j['entities']['mentions']:
                                mention_query = f"INSERT INTO twitter_tweet_mention VALUES ({input_dict['tweet_id']}, {i['id']}, '{i['username']}')"
                                cur.execute(mention_query)
                                con.commit()
                        if 'conversation_id' in list(j.keys()):
                            input_dict['conversation_id'] = j['conversation_id']
                        tweet_query = f"INSERT INTO twitter_tweet VALUES ({input_dict['tweet_id']},{input_dict['conversation_id']},'{input_dict['text']}', {input_dict['public_metrics']['like_count']}, {input_dict['public_metrics']['retweet_count']}, {input_dict['public_metrics']['reply_count']}, {input_dict['public_metrics']['quote_count']},'{input_dict['created_at']}')"
                        user_tweet_join_query = f"INSERT INTO twitter_tweet_to_user VALUES ({input_dict['tweet_id']},{input_dict['author_id']})"
                        cur.execute(tweet_query)
                        cur.execute(user_tweet_join_query)
                        con.commit()
            except:
                cur.execute(f"INSERT INTO twitter_noAcesses_user VALUES ({s_id})")
                con.commit()
                continue
                
def tw_cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
         if w.lower() in words or not w.isalpha())
    return tweet