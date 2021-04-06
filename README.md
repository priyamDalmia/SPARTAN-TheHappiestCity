# SPARTAN-TheHappiestCity
A simple, parallelized application leveraging the University of Melbourne HPC facility SPARTAN. 

The application uses a large Twitter dataset, a grid/mesh for Melbourne and a simple dictionary of terms related to sentiment scores. 
The objective is to calculate the sentiment score for the given cells and hence to calculate the area of Melbourne that has the happiest/most miserable people!

Cluster and Cloud Computing Assignment 1


1. json_analysis.py - contains the methods for reading the melbgrid , AFFIN, and the twitter data file.
                    - calculate_sentiment() method computes the sentiment score of a given list of tweets. 
                    - Not MPI compatible yet.
