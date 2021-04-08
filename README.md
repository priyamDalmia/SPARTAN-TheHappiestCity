# SPARTAN-TheHappiestCity
A simple, parallelized application leveraging the University of Melbourne HPC facility SPARTAN. 

The application uses a large Twitter dataset, a grid/mesh for Melbourne and a simple dictionary of terms related to sentiment scores. 
The objective is to calculate the sentiment score for the given cells and hence to calculate the area of Melbourne that has the happiest/most miserable people!

Cluster and Cloud Computing Assignment 1


1. json_analysis.py - contains the methods for reading the melbgrid , AFFIN, and the twitter data file.
                    - calculate_sentiment() method computes the sentiment score of a given list of tweets. 
                    - Not MPI compatible yet.

2. SLURM_1node_1core - contains SLURM submission script for running a single task in a single node.

3. SLURM_1node_8cores - contains SLURM submission script for running 8 processes in a single node.

4. SLURM_2nodes_4cores - contains SLURM submission script for running total of 8 processes: run in 2 nodes, with 4 tasks run per node.

5. mpi_tinytweet.py - focuses mainly on parallelizing the code using MPI. 
                    - calculates the number of tweets within a grid section (melbgrid). 
