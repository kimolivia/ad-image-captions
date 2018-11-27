import os
import pandas
import sys
import random


class Generate:

    def __init__(self, food_file_path, nn_file_path):
        self.df = pandas.read_csv(food_file_path)
        self.nn = pandas.read_csv(nn_file_path)
        self.start()

    def start(self):
        self.tweet_by_category()
        self.get_tweet()

    def tweet_by_category(self):

        food_dfs = {"Bread": self.df[self.df['food type'] == "Bread"],
                    "Dairy product": self.df[self.df['food type'] == "Dairy product"],
                    "Dessert": self.df[self.df['food type'] == "Dessert"],
                    "Egg": self.df[self.df['food type'] == "Egg"],
                    "Fried food": self.df[self.df['food type'] == "Fried food"],
                    "Meat": self.df[self.df['food type'] == "Meat"],
                    "Noodles/Pasta": self.df[self.df['food type'] == "Noodles/Pasta"],
                    "Rice": self.df[self.df['food type'] == "Rice"],
                    "Seafood": self.df[self.df['food type'] == "Seafood"],
                    "Soup": self.df[self.df['food type'] == "Soup"],
                    "Vegetable/Fruit": self.df[self.df['food type'] == "Vegetable/Fruit"]}

        return food_dfs

    def map_cat(self, category):

        categories = {"bread": "Bread",
         "dairy-product": "Dairy product",
         "dessert":"Dessert",
         "egg":"Egg",
         "fried-food": "Fried food",
         "meat":"Meat",
         "noodles/pasta":"Noodles/Pasta",
         "rice":"Rice",
         "seafood":"Seafood",
         "soup":"Soup",
         "vegetable":"Vegetable/Fruit",
         "fruit":"Vegetable/Fruit"
         }

        return categories[category]

    def get_tweet(self):

        nn_tweets = list()
        food_dfs = self.tweet_by_category()

        print(len(self.nn))
        for index, row in self.nn.iterrows():

            current_cat = row["rand_cats"]
            current_cat = self.map_cat(current_cat)

            cat_tweets = food_dfs[current_cat] # this will be all rows with same food type
            # print(cat_tweets)
            # print(type(cat_tweets))
            size = len(cat_tweets)
            rand_index = random.randint(0, size-1)

            i = 0
            for _, row1 in cat_tweets.iterrows():
                if i == rand_index:
                    nn_tweets.append(row1[" Media"])
                    break
                i += 1

        filler = ["No tweet :("]*(len(self.nn)-len(nn_tweets))
        nn_tweets += filler
        self.nn["nn_tweet"] = nn_tweets
        self.nn.to_csv("dummy_result.csv", sep=',')



def main():
    current_dir = os.getcwd()
    food_file_path = os.path.join(str(current_dir), "dummy_food.csv")
    nn_file_path = os.path.join(str(current_dir), "dummy-nnet-out.csv")

    if len(sys.argv) > 1:
        file_path = str(sys.argv[1])

    tweet = Generate(food_file_path, nn_file_path)

if __name__ == "__main__":
    main()

