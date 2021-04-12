import json
from mpi4py import MPI
from timer import Timer
import numpy as np
import re

t = Timer()
t.start()

def get_melbGrid(melb_file):
    try:
        with open(melb_file, encoding='utf-8') as f:
            grid = []
            labels = []
            data = json.load(f)
            grid_data = data.pop('features')
            for index, item in enumerate(grid_data):
                square = list((item['properties'].values()))
                grid.append(square)
                labels.append(grid[index][0])
                
            #Create lists of X and Y coordinates to be used in identifying grid boundaries
            X_coords = []
            Y_coords = []
            for grid_box in grid:
                for i in range(1,3): #collecting min and max of x-coordinates and adding to list
                    if grid_box[i] not in X_coords:
                        X_coords.append(grid_box[i])
                for j in range(3,5): #collecting min and max of y-coordinates and adding to list
                    if grid_box[j] not in Y_coords:
                        Y_coords.append(grid_box[j])

            sorted(X_coords, reverse = False)
            sorted(Y_coords, reverse = True)    

            # Tweet location group 
            # X_coords = [144.7, 144.85, 145.0, 145.15, 145.3]
            # Y_coords = [-37.5,-37.65, -37.8, -37.95]

        #    print(labels)
        # ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
        #      "C1", "C2", "C3", "C4", "C5", "D3", "D4", "D5"]
 

    except FileNotFoundError:
        print("melbGrid2 file not found!")

    return grid, X_coords, Y_coords, labels


def get_Afinn():
    """ Read and return AFINN sentiment score reference table """
    try:
        with open("AFINN.txt", 'r') as f:
            data = [i.split() for i in f.readlines()]
            score_table = {i[0]: i[-1] for i in data}

    except FileNotFoundError:
        print("AFINN file not found!")

    return score_table


def calculate_sentiment(tweet,coordinates, melb_grid, score_table):
    """Calculation of Sentiment score for each tweet"""
    
    #Initialize score dictionary where scores will be collected
    i_score = [0 for i in range(len(labels))]
    score_dict = dict(zip(labels, i_score))

    tweet_score = 0

    #Regex pattern for exact matches
    pattern = re.compile('^[A-Za-z]+(?=[ .!,]*$)')
    tweet_list = list(filter(pattern.match, tweet.split()))
#    print("tweet_list:", tweet_list)

    #Summing up Tweet sentiment Scores
    for i in tweet_list:
        if score_table.get((str(i)).lower()):
            tweet_score = tweet_score + int(score_table[(str(i)).lower()])

   # print("tweet_score:", tweet_score)

    cell = get_cell(coordinates, X_coords,Y_coords)

    # Recording tweet sentiment scores in score dictionary
    score_dict[cell] = score_dict.get(cell) + tweet_score
    #print("score dictionary", score_dict)

    return score_dict,cell


def get_cell(coordinates, X_coords,Y_coords):
    """Identification of cell to which tweet belongs to
        if the point lies on the grid coordinates"""

    #Initialize labels for grid rows
    grid_rows = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

    list_match = []

    # Case 1: tweet lies ALONG the boundaries on any cell; 
    #       If so, the tweet score will be added either to the left and/or the below adjacent cell.
    if coordinates[0] in X_coords or coordinates[1] in Y_coords:
        for grid_box in melb_grid:
            if (coordinates[1] >= grid_box[3] and coordinates[1] <= grid_box[4]) \
             and (coordinates[0] >= grid_box[1] and coordinates[0] <= grid_box[2]):
                list_match.append(grid_box[0]) #id

        #case 1.1 - when the tweet point lies ON the intersecting points of 4 cells
        if(len(list_match)>2): #matches 4 grid boxes
            cell = sorted(list_match, reverse = False)[3]

        #case 1.2 - when the tweet point lies ALONG the boundary connecting 2 grid cells: select either left and/or below cell
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


if __name__ == '__main__':
    """ Main function """

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    print("Rank:", my_rank)

    #Displaying where the current calculations sit in SPARTAN
    response = (MPI.Get_processor_name())
    print("responses from: ", response)


    #parametrize
    melb_grid,X_coords,Y_coords,labels = get_melbGrid('melbGridv2.json')
    twitter_file = 'smallTwitter.json'

    score_table = get_Afinn()


    """ Load and process twitter file line by line. """
    
    #Identify the respectivevindices on which each of the processes starts processing.
    try:
        if my_rank == 0:
            line_count = 0
            for line_count,line in enumerate(open(twitter_file)) :
                line_count += 1
                pass
            #print("line_count ", line_count)
        else:
            line_count = None

        line_count = comm.bcast(line_count, root=0) 
        job_count, start_index, end_index = get_job_index(line_count, size, my_rank)

        end = False
        pointer = 0
        total =[]

        ## Open Twitter file. Each of the processes starts and ends at corresponding start_line/end_line 
        #   in the smaller chunk of Twitter file
        with open(twitter_file, 'r') as twitter:
            for line in twitter:
                pointer += 1
                while pointer > start_index and end == False:
                    if 'id' in line:
                        coordinates = json.JSONDecoder().raw_decode(line)[0]['doc']['coordinates']['coordinates']
                        #print("coordinates:", coordinates[0], " ", coordinates[1])
                        
                        # Case 1: when the tweet is not within max and min boundaries of X and Y coordinates: to ignore tweet
                        if coordinates[0] < min(X_coords) or coordinates[0] > max(X_coords) \
                            or coordinates[1] < min(Y_coords) or coordinates[1] > max(Y_coords):
                            continue
                        
                        # Case 2: when the tweet is within boundaries: Begin processing. 
                        #      Each process/rank to process respective assigned lines from json file. For each line:
                        #           1. Get text element from each of the line.
                        #           2. Identify grid cell to which the tweet belongs to.
                        #           3. Count and sum sentiment score for each tweet.
                        #           4. Append scores into a list that collects all scores within the process.
                        else:
                             tweet = json.JSONDecoder().raw_decode(line)[0]['value']['properties']['text']
                        
                             score,cell = calculate_sentiment(tweet,coordinates,melb_grid, score_table)
                         
                             total.append(list(score.values()))   
                             #print("Cell ",cell, "print score: ", total)

                       
                    if pointer == end_index:
                        end = True
                    break

        #Summing up total within a process
        total = [sum(i) for i in zip(*total)]
        #print("summed total:", total)

    except FileNotFoundError:
        print("File not found!")

    total = comm.gather(total, root = 0)
    
    if my_rank ==0:
        print("--- Summary ---")
        print("Line_count:", line_count)
        total_score = [sum(i) for i in zip(*total)]
        total_score_dict = dict(zip(labels,total_score))
        print("total score:", total_score_dict)
        max_city = max(total_score_dict,key = total_score_dict.get)

        print("Happiest City is: ", max_city)
        print("with highest score:", max(total_score_dict.values()))
        t.stop()


