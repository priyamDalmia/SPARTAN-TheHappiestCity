import json
import re

n_tasks = 1


def calculate_sentiment(tweets, score_table):
    for index, tweet in enumerate(tweets):

        tweet_score = 0

        tweet_location = tweet['value']['geometry']['coordinates']
        print("Tweet coordinate : {0}".format(tweet_location))

        tweet_text = tweet['value']['properties']['text']
        print("Tweet text : {0}".format(tweet_text))

        pattern = re.compile('^[A-Za-z]+(?=[ .!,]*$)')
        tweet_list = list(filter(pattern.match, tweet_text.split()))

        # Tweet sentiment Score
        for i in tweet_list:
            if score_table.get((str(i)).lower()):
                tweet_score = tweet_score + int(score_table[(str(i)).lower()])

        print("Tweet Score", tweet_score)

        # Tweet location group
        X_coords = [144.7, 144.85, 145.0, 145.15, 145.3]
        Y_coords = [-37.65, -37.8, -37.95, -38.1]
        grid_rows = {1: 'A', 2: 'B', 3: 'C', 4: "D"}

        cell = (grid_rows[sum([1 if tweet_location[1] < i else 0 for i in Y_coords])]
                + str(sum([1 if tweet_location[0] > i else 0 for i in X_coords])))
        print("Tweet Cell ", cell)

        labels = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
                  "C1", "C2", "C3", "C4", "C5", "D3", "D4", "D5"]
        i_score = [0 for i in range(16)]
        score_dict = dict(zip(labels, i_score))

        score_dict[cell] = score_dict.get(cell) + tweet_score

    print(score_dict)


if __name__ == '__main__':

    try:
        with open("AFINN.txt", 'r') as f:
            data = [i.split() for i in f.readlines()]
            score_table = {i[0]: i[-1] for i in data}
    except FileNotFoundError:
        print("AFINN file not found!")

    try:
        with open('melbGrid2.json', encoding='utf-8') as f:
            grid = []
            data = json.load(f)
            grid_data = data.pop('features')

            for index, item in enumerate(grid_data):
                square = list((item['properties'].values()))
                grid.append(square)
    except FileNotFoundError:
        print("melbGrid2 file not found!")

    try:
        with open('tinyTwitter.json', encoding='utf-8') as f:
            data = json.load(f)
            tweets = data.pop('rows')

            for i in range(n_tasks):
                calculate_sentiment(tweets, score_table)
    except FileNotFoundError:
        print("File not found!")
