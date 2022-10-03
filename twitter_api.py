# encoding:utf-8
import tweepy
import configparser
import pandas as pd

#read config
config = configparser.ConfigParser()
config.read('config.ini') 

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config ['twitter']['access_token_secret']


# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token = (access_token ,access_token_secret)

api =tweepy.API(auth)


#search tweets about Fanatik
keywords = 'Fanatik'

          
           
limit=1000

tweets = tweepy.Cursor(api.search_tweets, lang="tr", q=keywords, count=100, tweet_mode = 'extended').items(limit)
                       
#create DataFrame
columns =['User','Tweet']
data = []
for tweet in tweets :
    
    data.append([tweet.user.screen_name,tweet.full_text])

df =pd.DataFrame(data, columns= columns)

#df.to_csv('tweetsset.csv')  #saving data to csv with pandas               

print(df)