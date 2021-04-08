
import json
from mpi4py import MPI
import time
start = time.clock()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Create a grid section object and set properties 
class Grid_box:
    def __init__(self, id, xmin, xmax, ymin, ymax, tweet_count):
        self.id = id
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.tweet_count = tweet_count


## Read coordinates of each section from the melbGrid file. Return list grid objects and total number of grid sections
def get_grid(grid, section_num):
    with open('melbGridv2.json','r') as file:
        for line in file:
            if 'id' in line:
                data = json.loads(line.rstrip(',\r\n'))
                properties = data['properties']
                
                grid.append(Grid_box(properties['id'], properties['xmin'],properties['xmax'],properties['ymin'],properties['ymax'],0))

                section_num += 1

    return grid, section_num


## Get number of tweets within each grid section.
## Twitter file is divided into smaller chunks: 
##   - each process calculates its own job counts, and its start/end lines where processing begins/ends within the big Twitter file;
##   - the grid sections where tweets belong to were finally identified.
def process_tweet(grid,section_num):
    
    #Rank 0 counts the number of lines in Twitter file
    if rank == 0:
        line_count = 0
        for line_count,line in enumerate(open("tinyTwitter.json","r")) :
            line_count += 1
        print("line_count ", line_count)
    else:
        line_count = None

    
    
    ## This is where the allocation of processes begins. The twitter file will be divided into chunks and allocated into the cores.
    ## The line_count will be used to determine how big each core will be processing.
    line_count = comm.bcast(line_count, root=0)
    job_count = line_count/size
    start_line = round(job_count) * rank
    end_line = 0

    if rank == (size-1):
        if (line_count % size) != 0:
            remainder = 0
            remainder = line_count % size
            end_line = start_line + round(job_count) + remainder

    else: 
        end_line = start_line + round(job_count)


    end = False
    pointer = 0

    ## Open Twitter file. Each of the processes starts and ends at corresponding start_line/end_line in the smaller chunk of Twitter file
    with open("tinyTwitter.json", 'r') as twitter:
        for line in twitter:
            pointer += 1
            while pointer > start_line and end == False:
                if 'id' in line:
                    coordinates = json.JSONDecoder().raw_decode(line)[0]['doc']['coordinates']['coordinates']
                   
                    ## Identification of which grid section the tweet belongs to.
                    for i in range(0,len(grid)):
                        if coordinates[0] >= grid[i].xmin and coordinates[0] < grid[i].xmax and coordinates[1]>=grid[i].ymin and coordinates[1]<grid[i].ymax:
                                grid[i].tweet_count += 1
                                break
                if pointer == end_line:
                    end = True
                break


## Combined results from all processes and stored in rank = 0; Printed results
def print_result(grid, section_num):

    all_grid = comm.gather(grid,root=0)    

    if rank == 0:
        for i in range(1,size):
            for j in range(section_num):
                if all_grid[0][j].id == all_grid[i][j].id:
                    all_grid[0][j].tweet_count += all_grid[i][j].tweet_count

        print ('\nTotal number of Tweets within each grid section:')
        for i in range(section_num):
            print ("Grid section ",all_grid[0][i].id,' - ',all_grid[0][i].tweet_count,'tweets' )
   

## Main function
def main():
    grid = []
    section_num = 0
    grid,section_num = get_grid(grid,section_num)
    process_tweet(grid,section_num)
    print_result(grid,section_num)
    if rank == 0:
        user_elapsed = (time.clock() - start)
        print ('\nTime elapsed:',user_elapsed)

    
 
main()

 
