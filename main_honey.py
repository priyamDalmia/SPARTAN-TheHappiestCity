import json
from mpi4py import MPI
from timer import Timer
import numpy as np
import re


t = Timer()
t_start = t.start()


def main():
    """Main function"""

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()
    print("Rank:", my_rank)

    #Displaying where the current calculations sit in SPARTAN
    response = (MPI.Get_processor_name())
    print("responses from: ", response)

    #Parametrize inputs
    melb_grid_file = 'melbGridv2.json'
    twitter_file = 'smallTwitter.json'

    scores, tweets,line_count, labels = process_tweets(comm,my_rank,size,\
                                    twitter_file,melb_grid_file)

    print_results(my_rank,line_count,scores,tweets,labels)

    if my_rank == 0:
        t.stop()
        

def get_melbGrid(melb_file):
    """Read melbGrid file and establish coordinates."""

    try:
        with open(melb_file, encoding='utf-8') as f:
            melb_grid = []
            labels = []
            data = json.load(f)
            grid_data = data.pop('features')
            for index, item in enumerate(grid_data):
                square = list((item['properties'].values()))
                melb_grid.append(square)
                labels.append(melb_grid[index][0])
                
            #Create lists of X and Y coordinates to be used in identifying grid boundaries
            X_coords = []
            Y_coords = []
            for grid_box in melb_grid:
                for i in range(1,3): #collecting min and max of x-coordinates and adding to list
                    if grid_box[i] not in X_coords: 
                        X_coords.append(grid_box[i])
                for j in range(3,5): #collecting min and max of y-coordinates and adding to list
                    if grid_box[j] not in Y_coords:
                        Y_coords.append(grid_box[j])

            sorted(X_coords, reverse = False)
            sorted(Y_coords, reverse = True)    

    except FileNotFoundError:
        print("melbGrid2 file not found!")

    return melb_grid, X_coords, Y_coords, labels


def get_Afinn():
    """ Read and return AFINN sentiment score reference table """
    try:
        with open("AFINN.txt", 'r') as f:
            data = [i.split() for i in f.readlines()]
            score_table = {i[0]: i[-1] for i in data}

    except FileNotFoundError:
        print("AFINN file not found!")

    return score_table


def process_tweets(comm, my_rank, size, twitter_file,melb_grid_file):
    """ Load and process twitter file line by line. """
    
    melb_grid,X_coords,Y_coords,labels = get_melbGrid(melb_grid_file)
    score_table = get_Afinn()

    #Identify the respective indices on which each of the processes starts processing.
    try:
        if my_rank == 0:
            line_count = 0
            for line_count,line in enumerate(open(twitter_file)) :
                line_count += 1
                pass
        else:
            line_count = None

        line_count = comm.bcast(line_count, root=0) 
        job_count, start_index, end_index = get_job_index(line_count, size, my_rank)

        end = False
        pointer = 0
        scores =[]
        tweets =[]

        ## Open Twitter file. Each of the processes starts and ends at corresponding 
        #   start_line/end_line in the smaller chunk of Twitter file
        with open(twitter_file, 'r') as twitter:
            for line in twitter:
                pointer += 1
                while pointer > start_index and end == False:
                    if 'id' in line:
                        coordinates = json.JSONDecoder().raw_decode(line)[0]['doc']\
                            ['coordinates']['coordinates']
                        #print("coordinates:", coordinates[0], " ", coordinates[1])
                        
                        # Case 1: when the tweet is not within max and min boundaries of 
                        #   X and Y coordinates: to ignore tweet
                        if coordinates[0] < min(X_coords) or coordinates[0] > max(X_coords) \
                            or coordinates[1] < min(Y_coords) or coordinates[1] > max(Y_coords):
                            pass
                        
                        # Case 2: when the tweet is within boundaries: Begin processing. 
                        #      Each process/rank to process respective assigned lines from 
                        #       json file. For each line:
                        #           1. Get text element from each of the line.
                        #           2. Identify grid cell to which the tweet belongs to.
                        #           3. Count and sum sentiment score for each tweet.
                        #           4. Append scores into a list of all scores within the process.
                        else:
                            tweet = json.JSONDecoder().raw_decode(line)[0]['value']\
                                    ['properties']['text']


                            calculated_cell = get_cell(melb_grid,coordinates, X_coords,Y_coords)
                        
                            score,tweet_count = calculate_sentiment(tweet,calculated_cell,\
                                        score_table,labels)
                            
                            #Counts total sentiment score for the tweet
                            scores.append(list(score.values()))

                            #Placeholder for the count of tweets, regardless of whether or not 
                            #   the tweet has sentiment score
                            tweets.append(list(tweet_count.values()))   

                    if pointer == end_index:
                        end = True
                    break

        #Summing up totals within a process
        scores = [sum(i) for i in zip(*scores)]
        tweets = [sum(i) for i in zip(*tweets)]
        #print("Tweet counts:", tweets)

    except FileNotFoundError:
        print("File not found!")

    #Gathering results at master level, rank = 0. 
    scores = comm.gather(scores, root = 0)
    tweets = comm.gather(tweets, root=0)

    return scores, tweets, line_count, labels


def get_job_index(line_count,size,my_rank):
    """ Determine job indices for each process to be used for breaking down tasks 
        of reading and processing twitter file into smaller chunks"""

    job_count = line_count/size
    start_index = round(job_count) * my_rank
    end_index = 0

    if my_rank == (size-1):  
        if (line_count % size) != 0:
            remainder = 0 
            remainder = line_count % size
            end_index = start_index + round(job_count) + remainder
    else: 
        end_index = start_index + round(job_count)

    return  job_count, start_index,end_index


def calculate_sentiment(tweet,calculated_cell,score_table,labels):
    """Calculation of Sentiment score for each tweet"""
    
    #Initialize score dictionary where scores will be collected
    i_score = [0 for i in range(len(labels))]
    score_dict = dict(zip(labels, i_score))

    j_score = [0 for i in range(len(labels))]
    tweet_count = dict(zip(labels, j_score))
    
    tweet_score = 0

    #Regex pattern for exact matches
    pattern = re.compile('^[A-Za-z]+(?=[ \'\".!,]*$)')

    #pattern = re.compile('^[A-Za-z]+(?=[ .!,'"]*$)')
    tweet_list = list(filter(pattern.match, tweet.split()))

    #Summing up Tweet sentiment Scores
    for i in tweet_list:
        if score_table.get((str(i)).lower()):
            tweet_score = tweet_score + int(score_table[(str(i)).lower()])

   # print("tweet_score:", tweet_score)

    #print("calculated cell:", calculated_cell)
    # Recording tweet sentiment scores in score dictionary
    score_dict[calculated_cell] = score_dict.get(calculated_cell) + tweet_score
    #print("score_dict[calc cell]:",  score_dict[calculated_cell])
    tweet_count[calculated_cell] = tweet_count.get(calculated_cell) + 1
    #print("tweet_count[calculated_cell]",tweet_count[calculated_cell] )


    return score_dict,tweet_count


def get_cell(melb_grid, coordinates, X_coords,Y_coords):
    """Identification of cell to which tweet belongs to
        if the point lies on the grid coordinates"""

    #Initialize labels for grid rows
    grid_rows = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

    list_match = []
    cell = ""

    # Case 1: tweet lies ALONG the boundaries on any cell; 
    #       If so, the tweet score will be added either to the left and/or the below adjacent cell
    if coordinates[0] in X_coords or coordinates[1] in Y_coords:
        for grid_box in melb_grid:
            if (coordinates[1] >= grid_box[3] and coordinates[1] <= grid_box[4]) \
             and (coordinates[0] >= grid_box[1] and coordinates[0] <= grid_box[2]):
                list_match.append(grid_box[0]) #id

        #case 1.1 - when the tweet point lies ON the intersecting points of 4 cells
        if(len(list_match)>2): #matches 4 grid boxes
            cell = sorted(list_match, reverse = False)[3]

        #case 1.2 - when the tweet point lies ALONG the boundary connecting 2 grid cells: 
        #       select either left and/or below cell
        elif len(list_match) == 2:
            if list_match[0][1] == list_match[1][1]: #comparison between top and bottom boxes
                cell = max(sorted(list_match, reverse = False))
            elif list_match[0][0] == list_match[1][0]:  #comparison between left and right boxes
                cell = min(sorted(list_match, reverse = False))
        elif len(list_match) == 1:
            cell = list_match[0] 

    #Case 2: when the point doesn't lie on the grid lines but lies within each cell
    else: 
        cell = (grid_rows[sum([1 if coordinates[1] < i else 0 for i in Y_coords])]
            + str(sum([1 if coordinates[0] > i else 0 for i in X_coords])))
        
    #print("Tweet Cell ", cell)
    #To test, point [144.9,-37.8] should lie on C2 and not B2 

    return cell




def print_results(my_rank,line_count,scores,tweets,labels):
    """Prints results in columnar format"""
    if my_rank ==0:
        print("--- Summary of Results---")
        print("Line_count:", line_count)
        total_score = [sum(i) for i in zip(*scores)]
        tweets_total_count = [sum(i) for i in zip(*tweets)]
        total_score_dict = dict(zip(labels,total_score))
        tweets_total_dict = dict(zip(labels,tweets_total_count))
        print("Cells: \t", "Total Tweets: \t", "Overall Sentiment Score:")
        for label in labels:
            print(label,"\t\t", tweets_total_dict.get(label),"\t\t",total_score_dict.get(label),)
        max_city = max(total_score_dict,key = total_score_dict.get)

        print("Happiest City is: ", max_city, ", highest score:", max(total_score_dict.values()))



if __name__ == '__main__':
    """ Main function """
    main()
