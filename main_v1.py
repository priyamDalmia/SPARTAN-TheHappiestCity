import json
from mpi4py import MPI
from timer import Timer
import numpy as np
import re

def calculate_sentiment(tweets):

    try:
        with open("AFINN.txt", 'r') as f:
            data = [i.split() for i in f.readlines()]
            score_table = {i[0]: i[-1] for i in data}
    except FileNotFoundError:
        print("AFINN file not found!")

    labels = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
              "C1", "C2", "C3", "C4", "C5", "D3", "D4", "D5"]
    i_score = [0 for i in range(16)]
    score_dict = dict(zip(labels, i_score))


    for index, tweet in enumerate(tweets):

        tweet_score = 0

        tweet_location = tweet['value']['geometry']['coordinates']

        tweet_text = tweet['value']['properties']['text']
        #print("Tweet text : ",tweet_text)

        pattern = re.compile('^[A-Za-z]+(?=[ .!,]*$)')
        tweet_list = list(filter(pattern.match, tweet_text.split()))

        # Tweet sentiment Score
        for i in tweet_list:
            if score_table.get((str(i)).lower()):
                tweet_score = tweet_score + int(score_table[(str(i)).lower()])

        #
        #print("Tweet Score", tweet_score)

        # Tweet location group
        X_coords = [144.7, 144.85, 145.0, 145.15, 145.3]
        Y_coords = [-37.65, -37.8, -37.95, -38.1]
        grid_rows = {1: 'A', 2: 'B', 3: 'C', 4: "D"}

        cell = (grid_rows[sum([1 if tweet_location[1] < i else 0 for i in Y_coords])]
                + str(sum([1 if tweet_location[0] > i else 0 for i in X_coords])))
        #print("Tweet Cell ", cell)

        #print("Adding score to cell {}, old {}.".format(cell, score_dict.get(cell)))

        score_dict[cell] = score_dict.get(cell) + tweet_score

    return score_dict

def run_job(job_rank):
    print(f'This is job - , {job_rank}')


if __name__ == '__main__':

    t = Timer()

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()


    try:
        with open('smallTwitter.json', encoding='utf-8') as f:
            data = json.load(f)
            tweets = data.pop('rows')
            num_tweets = len(tweets)

    except FileNotFoundError:
        print("File not found!")


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


    start_index = int(((num_tweets)/p)*(my_rank))
    end_index = int(((num_tweets)/p)*(my_rank+1))

    t.start("Process : {0}. Calculating sentiment."
        .format(my_rank))

    score = calculate_sentiment(tweets[start_index:end_index])
    total = list(score.values())
    print("Sentiment :", total)


    total = comm.gather(total, root = 0)

    if my_rank ==0:
        print("Process 0")
        print("total score : ", total)
        print([sum(i) for i in zip(*total)])



    t.stop()
