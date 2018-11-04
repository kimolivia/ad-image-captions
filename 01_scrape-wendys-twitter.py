import tweepy, json
import requests
import csv
import re

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
""" public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
 """


# Getting more tweets
# wendys_tw = api.user_timeline(screen_name = 'Wendys', count = 300, include_rts = True)
tweets = api.user_timeline(screen_name = 'Wendys', count = 300, include_rts = False, exclude_replies=False)
last_id = tweets[-1].id
while True:
    more_tweets = api.user_timeline(screen_name='Wendys',
    count=200,
    include_rts=False,
    exclude_replies=True,
    max_id=last_id-1)

    # There are no more tweets
    if len(more_tweets) == 0:
        break
    else:
        last_id = more_tweets[-1].id - 1
        tweets = tweets + more_tweets


def img_dl(img_url, id):

        # Download image in link
        img_data = requests.get(img_url).content
        file_loc = r'./media/i{}.jpg'.format(id)
        with open(file_loc, 'wb') as handler:
            handler.write(img_data)



media_dict = {}
for status in tweets:
    # getting id, text, media_link
    media_id = status.id_str
    media_text = status.text
    media = status.entities.get('media', [])
    if len(media) > 0:
        media_link = media[0]['media_url']
        img_dl(media_link, media_id)
    else:
        media_link = ''
    media_dict[media_id] = (media_text, media_link)


with open('./output/wendys-tweets-1104.txt', 'w+', encoding='utf-8') as file: 
    for k,(v1, v2) in media_dict.items():
        l = '{}\t{}\t{}\n'.format(k, v1, v2)
        l_print = re.sub(r'[^\x00-\x7f]', r' ', l)
        file.write(l)
        # print(l)

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
























