#----------------------------------------------------------------------------#
#                              Gestion de projet :                           #
#             Group n°5 : Protein-Protein Interaction Network                #
#                                                                            #
# Coordinator : Emmanuelle Becker                                            #
# https://github.com/isDuke/Biological-Networks-Project-in-Python.git        #
#----------------------------------------------------------------------------#

## CHAPTER N°2 : EXPLORING THE PROTEIN-PROTEIN INTERACTION GRAPH

#----------------------------------------------------------------------------#
"""
Unit tests (pytest) for chapter 2: exploration of the graph and degrees.

Run from the folder that contains read_graph.py with:
    pytest -v test_graph_exploration_pytest.py

No file of data/ is needed: each test writes its own graph in a temporary
folder given by pytest (tmp_path).

Graphs used:
    toy graph: 7 interactions (A-B A-F B-C B-D B-F C-D D-E), no duplicate
            degrees: A=2 B=4 C=2 D=3 E=1 F=2
    dirty graph: 9 interactions with 3 redundant ones (B-A, a 2nd A-B, D-C)
            and 2 homodimers (A-A, C-C), 4 interactions once cleaned
"""

import pytest

from read_graph import (is_interaction_file,
                        count_vertices,
                        count_edges,
                        clean_interactome,
                        get_degree,
                        get_max_degree,
                        get_ave_degree,
                        count_degree,
                        histogram_degree)

# Toy graph: A is linked to B and F as in the example of the TP.
# A is only in the 1st column and E only in the 2nd one.
TOY_EDGES_LIST = [("A", "B"), ("A", "F"), ("B", "C"), ("B", "D"),
                  ("B", "F"), ("C", "D"), ("D", "E")]
TOY_DEGREES_DICT = {
    "A": 2, 
    "B": 4, 
    "C": 2, 
    "D": 3, 
    "E": 1, 
    "F": 2
}

# number of proteins for each degree
TOY_HISTO_DICT = {
    1: 1, 
    2: 3, 
    3: 1, 
    4: 1
}

# Dirty graph and what should remain after clean_interactome
DIRTY_EDGES_LIST = [("A", "B"), ("B", "A"), ("A", "B"), ("A", "A"),
                    ("B", "C"), ("C", "C"), ("C", "D"), ("D", "C"),
                    ("A", "D")]

CLEAN_EDGES_SET = {("A", "B"), ("A", "D"), ("B", "C"), ("C", "D")}

#----------------------------------------------------------------------------#
## Helper functions

def write_test_file(path_str: str, edges_list: list) -> None:
    """
    Write an interaction file in the format of the TP: the number of
    interactions on the first line, then one interaction per line.

    Arguments:
    path_str (str) : path of the file to create
    edges_list (list) : list of tuples (protein_a, protein_b)

    Output:
    None (the file is written on the disk).
    """
    with open(path_str, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{len(edges_list)}\n")
        for prot_a_str, prot_b_str in edges_list:
            file_obj.write(f"{prot_a_str}\t{prot_b_str}\n")


def read_edges_list(path_str: str) -> list:
    """
    Read the interactions of a file without using read_graph.py, and sort
    the two proteins of each one so that A-B and B-A look the same.

    Argument:
    path_str (str) : path of the interaction file

    Output:
    A list of sorted tuples (protein_a, protein_b), in the file order.
    """
    with open(path_str, "r", encoding="utf-8") as file_obj:
        next(file_obj)  # skip the number of interactions
        return [tuple(sorted(line_str.split()))
                for line_str in file_obj if line_str.strip()]

def parse_histogram(output_str: str) -> dict:
    """
    Turn the text printed by histogram_degree into a dictionary: each
    line "d ***" becomes an item {d: number of stars}. The printed lines
    that do not start with a number are ignored.

    Argument:
    output_str (str) : text printed by histogram_degree

    Output:
    A dictionary {degree: number of stars}.
    """
    histo_dict = {}
    for line_str in output_str.splitlines():
        words_list = line_str.split()
        if words_list and words_list[0].isdigit():
            histo_dict[int(words_list[0])] = line_str.count("*")
    return histo_dict

## Fixtures

@pytest.fixture
def toy_file_str(tmp_path) -> str:
    """
    Fixture that writes the toy graph in the temporary folder of the test.

    Argument:
    tmp_path (pathlib.Path) : temporary folder created by pytest

    Output:
    The path (str) of the toy graph file.
    """
    path_str = str(tmp_path / "toy.txt")
    write_test_file(path_str, TOY_EDGES_LIST)
    return path_str


@pytest.fixture
def dirty_file_str(tmp_path) -> str:
    """
    Fixture that writes the dirty graph in the temporary folder of the test.

    Argument:
    tmp_path (pathlib.Path) : temporary folder created by pytest

    Output:
    The path (str) of the dirty graph file.
    """
    path_str = str(tmp_path / "dirty.txt")
    write_test_file(path_str, DIRTY_EDGES_LIST)
    return path_str


@pytest.fixture
def clean_file_str(tmp_path, dirty_file_str) -> str:
    """
    Fixture that runs clean_interactome on the dirty graph.

    Arguments:
    tmp_path (pathlib.Path) : temporary folder created by pytest
    dirty_file_str (str) : path of the dirty graph (fixture)

    Output:
    The path (str) of the cleaned file.
    """
    path_str = str(tmp_path / "clean.txt")
    clean_interactome(dirty_file_str, path_str)
    return path_str

#----------------------------------------------------------------------------#
# 2.1 - Exploration of the global graph

## count_vertices (2.1.1) and count_edges (2.1.2)

class TestGlobalExploration:
    """Tests for count_vertices and count_edges."""

    def test_count_vertices(self, toy_file_str):
        """
        Check that the toy graph has 6 proteins (A, B, C, D, E and F).

        Argument:
        toy_file_str (str) : path of the toy graph (fixture)

        Output:
        None, the assert fails if a protein is forgotten or counted twice.
        """
        assert count_vertices(toy_file_str) == 6

    def test_count_edges(self, toy_file_str):
        """
        Check that the toy graph has 7 interactions.

        Argument:
        toy_file_str (str) : path of the toy graph (fixture)

        Output:
        None, the assert fails if the number of edges is not 7.
        """
        assert count_edges(toy_file_str) == 7

    def test_smallest_graph(self, tmp_path):
        """
        Check the smallest possible graph: 1 interaction, so 2 vertices
        and 1 edge.

        Argument:
        tmp_path (pathlib.Path) : temporary folder created by pytest

        Output:
        None, the assert fails if one of the two counts is wrong.
        """
        small_str = str(tmp_path / "small.txt")
        write_test_file(small_str, [("A", "B")])
        assert count_vertices(small_str) == 2
        assert count_edges(small_str) == 1

## clean_interactome (2.1.3)

class TestCleanInteractome:
    """Tests for clean_interactome, run on the dirty graph."""

    def test_output_format(self, clean_file_str):
        """
        Check that the cleaned file has the same format as the input file,
        using is_interaction_file from chapter 1.

        Argument:
        clean_file_str (str) : path of the cleaned file (fixture)

        Output:
        None, the assert fails if the format is not respected.
        """
        assert is_interaction_file(clean_file_str)

    def test_no_homodimer(self, clean_file_str):
        """
        Check that no protein interacts with itself after cleaning.

        Argument:
        clean_file_str (str) : path of the cleaned file (fixture)

        Output:
        None, the assert fails at the first homodimer found.
        """
        for prot_a_str, prot_b_str in read_edges_list(clean_file_str):
            assert prot_a_str != prot_b_str

    def test_no_redundancy(self, clean_file_str):
        """
        Check that each interaction appears only once after cleaning.
        A-B and B-A are the same interaction since the graph is undirected.

        Argument:
        clean_file_str (str) : path of the cleaned file (fixture)

        Output:
        None, the assert fails if an interaction is still duplicated.
        """
        edges_list = read_edges_list(clean_file_str)
        assert len(edges_list) == len(set(edges_list))

    def test_expected_edges(self, clean_file_str):
        """
        Check that the cleaned file contains exactly the 4 interactions
        of CLEAN_EDGES_SET.

        Argument:
        clean_file_str (str) : path of the cleaned file (fixture)

        Output:
        None, the assert fails if an interaction is missing or added.
        """
        assert set(read_edges_list(clean_file_str)) == CLEAN_EDGES_SET

    def test_input_unchanged(self, dirty_file_str, clean_file_str):
        """
        Check that clean_interactome does not modify the input file,
        which must still contain its 9 interactions.
        clean_file_str is asked only so that the cleaning runs first.

        Arguments:
        dirty_file_str (str) : path of the dirty graph (fixture)
        clean_file_str (str) : path of the cleaned file (fixture)

        Output:
        None, the assert fails if the input file was changed.
        """
        edges_list = read_edges_list(dirty_file_str)
        assert len(edges_list) == len(DIRTY_EDGES_LIST)

    def test_count_after_cleaning(self, clean_file_str):
        """
        Check that count_vertices and count_edges work on a cleaned file:
        4 proteins (A, B, C, D) and 4 interactions.

        Argument:
        clean_file_str (str) : path of the cleaned file (fixture)

        Output:
        None, the assert fails if one of the two counts is wrong.
        """
        assert count_vertices(clean_file_str) == 4
        assert count_edges(clean_file_str) == 4

#----------------------------------------------------------------------------#
# 2.2 - Work on the degree of the vertices

## get_degree, get_max_degree, get_ave_degree, count_degree (2.2.1 to 2.2.4)

class TestDegree:
    """Tests for the degree functions, run on the toy graph."""

    @pytest.mark.parametrize("prot_str, degree_int", TOY_DEGREES_DICT.items())
    def test_get_degree(self, toy_file_str, prot_str, degree_int):
        """
        Check the degree of one protein. Thanks to parametrize, pytest
        runs this test 6 times, once per item of TOY_DEGREES_DICT.

        Arguments:
        toy_file_str (str) : path of the toy graph (fixture)
        prot_str (str) : name of the protein
        degree_int (int) : expected degree of this protein

        Output:
        None, the assert fails if the degree is not the expected one.
        """
        assert get_degree(toy_file_str, prot_str) == degree_int

    def test_get_max_degree(self, toy_file_str):
        """
        Check that B is the protein with the highest degree, which is 4.
        The result is unpacked, so a tuple or a list are both accepted.

        Argument:
        toy_file_str (str) : path of the toy graph (fixture)

        Output:
        None, the assert fails if the protein or the degree is wrong.
        """
        prot_str, degree_int = get_max_degree(toy_file_str)
        assert prot_str == "B"
        assert degree_int == 4

    def test_get_ave_degree(self, toy_file_str):
        """
        Check the average degree: sum of the degrees / number of proteins,
        so 14 / 6. pytest.approx accepts a result rounded to 2 decimals.

        Argument:
        toy_file_str (str) : path of the toy graph (fixture)

        Output:
        None, the assert fails if the average is wrong.
        """
        ave_float = get_ave_degree(toy_file_str)
        assert ave_float == pytest.approx(14 / 6, abs=0.01)

    @pytest.mark.parametrize("degree_int, count_int", TOY_HISTO_DICT.items())
    def test_count_degree(self, toy_file_str, degree_int, count_int):
        """
        Check the number of proteins having a given degree. The test is
        run once per item of TOY_HISTO_DICT (degrees 1 to 4).

        Arguments:
        toy_file_str (str) : path of the toy graph (fixture)
        degree_int (int) : degree to look for
        count_int (int) : expected number of proteins with this degree

        Output:
        None, the assert fails if the number of proteins is wrong.
        """
        assert count_degree(toy_file_str, degree_int) == count_int

    def test_count_degree_absent(self, toy_file_str):
        """
        Check that count_degree returns 0 for a degree that no protein has.

        Argument:
        toy_file_str (str) : path of the toy graph (fixture)

        Output:
        None, the assert fails if the result is not 0.
        """
        assert count_degree(toy_file_str, 10) == 0

## histogram_degree (2.2.5)


class TestHistogramDegree:
    """Tests for the histogram printed by histogram_degree."""

    def test_histogram_values(self, toy_file_str, capsys):
        """
        Check the histogram between degrees 1 and 4: each line must have
        one star per protein having this degree.

        Arguments:
        toy_file_str (str) : path of the toy graph (fixture)
        capsys : pytest fixture that captures what the function prints

        Output:
        None, the assert fails if a line of the histogram is wrong.
        """
        histogram_degree(toy_file_str, 1, 4)
        output_str = capsys.readouterr().out
        assert parse_histogram(output_str) == TOY_HISTO_DICT

    def test_histogram_bounds(self, toy_file_str, capsys):
        """
        Check that only the degrees between dmin and dmax (both included)
        are printed: here 2 and 3.

        Arguments:
        toy_file_str (str) : path of the toy graph (fixture)
        capsys : pytest fixture that captures what the function prints

        Output:
        None, the assert fails if a degree out of the range is printed.
        """
        histogram_degree(toy_file_str, 2, 3)
        output_str = capsys.readouterr().out
        assert parse_histogram(output_str) == {2: 3, 3: 1}

#----------------------------------------------------------------------------#
# Main

if __name__ == "__main__":
    pytest.main(["-v", __file__])