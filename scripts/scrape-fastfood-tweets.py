import tweepy, json
import requests
import csv
import re
import datetime
import sys


class MyTweetMediaDownloader():

    def __init__(self, api, user):
        self.api = api
        self.user = user
        self.start()
        
    def start(self):
        tweets = self.get_tweets()
        print('Extracted tweets from ', self.user)
        media_dict = self.extract_text_media(tweets)
        now = datetime.datetime.now()
        csv_filename = r'./output/tweet_info/{}-tweets-{}.csv'.format(self.user, now.strftime(r'%m-%d-%H-%M'))
        self.tweet_to_csv(media_dict, csv_filename)
        print('Saved tweets from ', self.user)

    def img_dl(self, img_url, idx):

        # Download image in link
        img_data = requests.get(img_url).content
        file_loc = r'./output/imgs/{}-i{}.jpg'.format(self.user, idx)
        with open(file_loc, 'wb') as handler:
            handler.write(img_data)

    def get_tweets(self):

        # Extract as many tweets as possible from the specified user
        tweets = self.api.user_timeline(screen_name = self.user, count = 200, 
        include_rts = False, exclude_replies=False)
        last_id = tweets[-1].id
        while True:
            more_tweets = self.api.user_timeline(screen_name=self.user,
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
        
        return tweets

    def extract_text_media(self, tweets):

        # getting id, text, media_link
        media_dict = {}
        for status in tweets:
            media_id = status.id_str
            media_text = status.text
            media = status.entities.get('media', [])
            if len(media) > 0:
                media_link = media[0]['media_url']
                self.img_dl(media_link, media_id)
            else:
                media_link = ''
            media_dict[media_id] = (media_text, media_link)
        return media_dict

    def make_printable(self, txt):

        # remove emoticons, commas, and carriage returns
        pat = r'[^\x00-\x7F]+'
        pat2 = r','
        pat3 = r'\n'
        txt = re.sub(pat, r' ', txt)
        txt = re.sub(pat2, r' ', txt)
        txt = re.sub(pat3, r' ', txt)
        return txt
                 
        
    def tweet_to_csv(self, media_dict, csv_filename):

        # write tweet info into a csv
        with open(csv_filename, 'w') as f: 
            f.write('ID, Media, Link\n')
            for k,(text, link) in media_dict.items():
                text = self.make_printable(text)
                link = self.make_printable(link)
                row = '{},{},{}\n'.format(k, text, link)
                # print(row)
                f.write(row)


def twitter_access(credentials):
    with open(credentials, 'r') as f:
        twitter = f.read().split('\n')

    access_token = twitter[0]
    access_token_secret = twitter[1]
    consumer_key = twitter[2]
    consumer_secret = twitter[3]

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api


def main():
    
    # Read Twitter credentials
    credentials = './data/twitter.pw'
    if len(sys.argv) > 1:
        credentials = sys.argv[1]

    api = twitter_access(credentials)
    # https://people.com/food/best-fast-food-tweets/
    # https://spoonuniversity.com/lifestyle/best-fast-food-chains-to-follow-on-twitter
    # https://www.myrecipes.com/news/funny-fast-food-twitter-accounts 
    users = [
        'Wendys', 'DennysDiner', 'IHOP', 'redlobster', 'tacobell', 'DiGiorno',
        'BurgerKing', 'Oreo', 'kitkat', 'Arbys', 'ChickfilA', 'dominos',
        'WhiteCastle', 'ChipotleTweets', 'kfc'] 
    # user = 'Wendys'
    # MyTweetMediaDownloader(api, user)
    [MyTweetMediaDownloader(api, u) for u in users]
    


if __name__ == '__main__':
    main()
















