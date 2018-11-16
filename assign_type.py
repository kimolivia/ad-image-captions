import pandas
import os
import sys
import random


class Type:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pandas.read_csv(file_path)
        self.food_types = ["Bread", "Dairy product", "Dessert",
                           "Egg", "Fried food", "Meat", "Noodles/Pasta",
                           "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
        self.start()

    def start(self):
        self.random_assign()

    def keep_links(self):
        """
        args:
            None
        return:
            None
        Renames a column to remove whitespace and removes rows that do not have a link
        """
        self.df = self.df.rename(columns={' Link': 'Link'})
        self.df = self.df[pandas.notnull(self.df['Link'])]

    def random_assign(self):
        """
        args:
            None
        return:
            None
        Randomly generates a food type and assigns the result to a new column
        """
        self.keep_links()
        size = len(self.df.index)
        generated_types = list()
        for i in range(size):
            rand_index = random.randint(0, 10)
            generated_types.append(self.food_types[rand_index])
        self.df["food type"] = generated_types
        print(self.df.to_string())


def main():

    current_dir = os.getcwd()
    file_path = os.path.join(str(current_dir), "output", "tweet_info", "Wendys-tweets-11-14-16-42.csv")

    if len(sys.argv) > 1:
        file_path = str(sys.argv[1])

    classified = Type(file_path)

if __name__ == "__main__":
    main()
