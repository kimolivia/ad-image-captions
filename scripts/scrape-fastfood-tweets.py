import tweepy, json
import requests
import csv
import re
from datetime import datetime
import sys


class TweetMediaDownloader():
    '''Downloads tweets and attached images.

    Attributes:
        api: A tweepy object w/ authentification and access handled.
    '''

    def __init__(self, api):
        self.api = api
            
    def get_tweets(self, user):
        '''Download as many recent tweets as possible from user.
        
        Args:
            user: Twitter handle of desired entity.
        
        Returns:
            A list of status objects from the tweepy package 
            (works like a dict). Each status object contains 
            all the metadata for 1 tweet. For example:

            [
                {
                    'id_str': '1064083674935816192',
                    'text': '@RealKevinKuehne Well that’s just too long.',
                    'link': 'http://pbs.twimg.com/tweet_video_thumb/DnZH404WsAAXuJL.jpg'
                },
                {
                    'id_str': '1064083674935816192',
                    'text': 'Wazzzupsdf my dude',
                    'link': 'http://pbs.twimg.com/tweet_video_thumb/DnZH404WsAAXuJL.jpg'
                }
            ]         
        '''

        tweets = self.api.user_timeline(screen_name = user, count = 200, 
        include_rts = False, exclude_replies=False)
        last_id = tweets[-1].id
        while True:
            more_tweets = self.api.user_timeline(screen_name=user,
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
        print(f'\n\nFrom {user}: downloaded {len(tweets)} tweets')
        
        return tweets

    def parse_tweets(self, tweets):
        '''Extract tweet id, text, and image link (if applicable).

        Args:
            tweets: List of status objects.
        
        Returns:
            A dict mapping tweet-id to a tuple of the tweet text and link.
            For example:

            {
                '1042834936938550000': ('Whats up',	http://twimg.com/AEx2V1.jpg),
                '1042834936938551111': ('Nothing much',	http://twimg.com/faf1.jpg)
            }        
        '''
        media_dict = {}
        media_link_n = 0
        for status in tweets:
            media_id = status.id_str
            media_text = status.text
            media = status.entities.get('media', [])
            if len(media) > 0:
                media_link = media[0]['media_url']
                media_link_n += 1
            else:
                media_link = ''
            media_dict[media_id] = (media_text, media_link)

        print(f'There are {media_link_n} images in {len(tweets)} tweets')

        return media_dict

    def make_printable(self, txt):
        '''Remove emoticons, commas, and carriage returns.'''

        pat = r'[^\x00-\x7F]+'
        pat2 = r','
        pat3 = r'\n'
        txt = re.sub(pat, r' ', txt)
        txt = re.sub(pat2, r' ', txt)
        txt = re.sub(pat3, r' ', txt)

        return txt

    def img_dl(self, img_url, file_loc):
        '''Download image in link.'''

        img_data = requests.get(img_url).content
        with open(file_loc, 'wb') as handler:
            handler.write(img_data)

    def dwnld_images(self, media_dict, user, folder_root='./output/'):
        '''Download all images in media dict to folder.'''

        img_n = 0
        for k, (_, link) in media_dict.items():
            if link != '':
                file_loc = f'{folder_root}{user}-i{k}.jpg'
                self.img_dl(link, file_loc)
                img_n += 1

        print(f'Successfully downloaded {img_n} images to {folder_root} folder')
        
    def tweet_to_csv(self, media_dict, csv_filename):
        '''Write all tweet information into a csv.'''

        with open(csv_filename, 'w') as f: 
            f.write('ID,Media,Link\n')
            for k,(text, link) in media_dict.items():
                text = self.make_printable(text)
                link = self.make_printable(link)
                row = '{},{},{}\n'.format(k, text, link)
                f.write(row)
        
        print('Tweets info is now a CSV in your folder')


def twitter_access(credentials):
    '''Return tweepy API object using API keys.'''

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
    
    '''
    Sources: 
        https://people.com/food/best-fast-food-tweets/
        https://spoonuniversity.com/lifestyle/best-fast-food-chains-to-follow-on-twitter
        https://www.myrecipes.com/news/funny-fast-food-twitter-accounts 
    '''
    # users = [
    #     'Wendys', 'DennysDiner', 'IHOP', 'redlobster', 'tacobell', 'DiGiorno',
    #     'BurgerKing', 'Oreo', 'kitkat', 'Arbys', 'ChickfilA', 'dominos',
    #     'WhiteCastle', 'ChipotleTweets', 'kfc'] 

    users = ['Wendys']

    tweetMuncher = TweetMediaDownloader(api)
    time_stamp = datetime.now().strftime(r'%m-%d-%H-%M')

    for u in users:
        twts_raw = tweetMuncher.get_tweets(u)
        twts_info = tweetMuncher.parse_tweets(twts_raw)

        csv_filename = f'./output/tweet_info/{u}-tweets-{time_stamp}.csv'
        # tweetMuncher.tweet_to_csv(twts_info, csv_filename)

        img_folder = './output/imgs'
        # tweetMuncher.dwnld_images(twts_info, u, img_folder)

 

if __name__ == '__main__':
    main()
















