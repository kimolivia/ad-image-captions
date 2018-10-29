import tweepy, json

access_token = r'1057005035232604160-pVOp37d0C6ND3Cdka15OFUnoh94OHH'
access_token_secret = r'X15XCCZSqPquCgtwB70uayulGzHAtkWYTlOlt9q6uFTSu'
consumer_key = r'kFeFUxECg61jEIjpYnjLDDvoV'
consumer_secret = r'SgQOy5SfR7PBz2vp3eFl9sYRPnJXWpIR5IS3LJf05f68mn3dor'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# print tweets on my home page
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open('tweets.txt', 'w')
    
    def on_status(self, status):
        tweet = status._json
        self.file.write(json.dumps(tweet) + '\n')
        tweet_list.append(status)
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()
























