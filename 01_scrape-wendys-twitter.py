import tweepy, json


with open('twitter.pw', 'r') as f:
    twitter = f.read().split('\n')


access_token = twitter[0]
access_token_secret = twitter[1]
consumer_key = twitter[2]
consumer_secret = twitter[3]

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
























