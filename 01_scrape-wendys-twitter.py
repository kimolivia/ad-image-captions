import tweepy, json

access_token = r'1057005035232604160-pVOp37d0C6ND3Cdka15OFUnoh94OHH'
access_token_secret = r'X15XCCZSqPquCgtwB70uayulGzHAtkWYTlOlt9q6uFTSu'
consumer_key = r'kFeFUxECg61jEIjpYnjLDDvoV'
consumer_secret = r'SgQOy5SfR7PBz2vp3eFl9sYRPnJXWpIR5IS3LJf05f68mn3dor'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)