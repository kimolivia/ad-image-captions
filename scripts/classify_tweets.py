import pandas
import os
import sys
import string

# Naively tries to classify tweets based on common words used in advertising etc.

class Classify:

    def __init__(self, file_path):
        self.file_path = file_path
        self.words = {"user": {"sorry", "dm", "suggestions"},
                      "ads": {"fresh", "delicious", "get", "download", "deal", "app", "purchase"}}
        self.df = pandas.read_csv(file_path)
        self.start()

    def start(self):
        self.classify_type()

    def word_counter(self):
        """
        args:
            None
        return:
            word_counts - a dictionary with the count of each word in a tweet for each tweet
        """
        exclude = set(string.punctuation)
        tweets = self.df.loc[:, " Media"]
        word_counts = dict()
        for i in range(len(tweets)):
            current_word_dict = dict()
            for word in tweets[i].split():
                word = (''.join(ch for ch in word if ch not in exclude)).lower()
                if word in current_word_dict:
                    current_word_dict[word] += 1
                else:
                    current_word_dict[word] = 1
            word_counts[i] = current_word_dict
        return word_counts

    def count_type(self):
        """
        args:
            None
        return:
            classified_tweets: a dictionary containing a count of how many words
            are ad related or user interaction related
        """

        word_counts = self.word_counter()
        classified_tweets = {}

        for num, tweet in word_counts.items():
            classified = {"user": 0, "ads": 0}
            for word, count in tweet.items():
                if word in self.words["user"]:
                    classified["user"] += count
                elif word in self.words["ads"]:
                    classified["ads"] += count
            classified_tweets[num] = classified

        return classified_tweets

    def classify_type(self):
        """
        args:
            None
        return:
            df - the original dataframe with a column adding the classification type
        """

        types = self.count_type()
        result = list()
        for k, v in types.items():
            if v["user"] > v["ads"]:
                result.append("user")
            elif v["user"] < v["ads"]:
                result.append("ads")
            else:
                result.append("funny")
        self.df["tweet type"] = result
        print(self.df.to_string())
        return self.df


def main():

    current_dir = os.getcwd()
    file_path = os.path.join(str(current_dir), "output", "tweet_info", "wendys-tweets-11-04-22-20.csv")

    if len(sys.argv) > 1:
        file_path = str(sys.argv[1])

    classified = Classify(file_path)


if __name__ == '__main__':
    main()