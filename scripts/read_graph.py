#----------------------------------------------------------------------------#
#                              Gestion de projet :                           #
#             Group n°5 : Protein-Protein Interaction Network        #
#                                                                            #
# Coordinator : Emmanuelle Becker                                              #
# https://github.com/isDuke/Biological-Networks-Project-in-Python.git        #
#----------------------------------------------------------------------------#

## CHAPTER N°1 : INTERPRETING A PROTEIN-PROTEIN INTERACTION NETWORK

#----------------------------------------------------------------------------#
import os
import numpy as np
import sys

WORKING_DIR = os.path.abspath(".")

DATA_PATH = os.path.join(WORKING_DIR, "data/")
SCRIPT_PATH = os.path.join(WORKING_DIR, "scripts/")
TEST_PATH = os.path.join(WORKING_DIR, "tests/")

TEST_FILE = os.path.join(DATA_PATH, "toy_example.txt")
HUMAN_HIGH_QUALITY = os.path.join(DATA_PATH, "Human_HighQuality.txt")
#----------------------------------------------------------------------------#

# 1.2 - Data structure for storing the graph

## Dictionnary (1.2.1)

def read_interaction_file_dict(file : str) -> dict :
    """
    A function that reads a protein-protein interaction graph from a tabular 
    file and stores it in a dictionary.

    Argument:
    file (str) : a tabulate file

    Output: 
    A dictionary where the key is one protein and the value is another protein.
    """

    with open(file, "r", encoding='utf-8') as fh :
        interactions_count_int = int(next(fh)) # Skip the first line, which corresponds to the number of interactions
        interactions_dict = {}

        for line in fh:
            key, value = line.split()
            interactions_dict.setdefault(value, []).append(key)

    print(f"Number of interactions: {interactions_count_int} \nDictionary obtained: {interactions_dict}")
    return interactions_dict

## List (1.2.2)

def read_interaction_file_list(file : str) -> list :
    """
    Function that reads a protein-protein interaction graph from a tabular file 
    and stores it in a list.

    Argument:
    file (str): the tabular file

    Output:
    A list of tuples (protein_a, protein_b), one per interaction, in the order of the 
    file.
    """

    with open(file, "r", encoding="utf-8") as fh :
        interactions_count = int(fh.readline())
        interactions_list = [tuple(line.split()) for line in fh if line.strip()]

    print(f"Number of interactions: {interactions_count} \nList of tupples obtained: {interactions_count}")
    return interactions_list

## Adjacency Matrix (1.2.3)

def read_interaction_file_mat(file_name):
    """
    Reads a tab-separated file containing a protein interaction graph and represents it as an adjacency matrix.

    The file must contain:
    - the number of interactions on the first line;
    - a header line on the second line;
    - one interaction per line in the following format:
      protein_1 protein_2.

    Argument: 
    file_name : str
        Path to the tab-separated file containing the interactions.

    Output: 
        A tuple containing:
        - matrix : numpy.ndarray
            The adjacency matrix of the graph.
        - vertices : list
            The sorted list of proteins present in the graph.
    """
    with open(file_name) as file:
        interactions_count_int = int (next(file))
        lines = [line.split() for line in file if line.strip()]


    edges = [line[:2] for line in lines[1:]]

    vertices = sorted({vertex for edge in edges for vertex in edge})
    indices = {vertex: i for i, vertex in enumerate(vertices)}

    matrix = np.zeros((len(vertices), len(vertices)), dtype=int)

    for vertex_1, vertex_2 in edges:
        i, j = indices[vertex_1], indices[vertex_2]
        matrix[i, j] = matrix[j, i] = 1

    print(f"Number of interactions: {interactions_count_int}\n Resulting matrix:\n{matrix}")
    return matrix, vertices

## Triplet (1.2.4)


def is_interaction_file(file_str):
    """
    Check that a file follows the interaction file format.

    Expected format: the first line is the number of interactions,
    then one interaction per line with exactly 2 columns
    (separated by tabs or spaces). Blank lines are ignored.

    Argument:
    file (str) -> 

    Ouput: 

    A boolean (True/False).
    """
    with open(file_str, "r") as file_obj:
        line_list = [line.strip() for line in file_obj if line.strip()]

    if not line_list:
        return False  # empty file
    
    if not line_list[0].isdigit():
        return False  # first line missing or not a number
    
    if int(line_list[0]) != len(line_list) - 1:
        return False  # first line is not the number of interactions
    
    for line_str in line_list[1:]:
        if len(line_str.split()) != 2:
            return False  # wrong number of columns
        
    return True


if __name__ == "__main__" :
    print(read_interaction_file_dict(TEST_FILE))
    print(read_interaction_file_list(TEST_FILE))