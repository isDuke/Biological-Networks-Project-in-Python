#----------------------------------------------------------------------------#
#                              Gestion de projet :                           #
#             Group n°5 : Protein-Protein Interaction Network                #
#                                                                            #
# Coordinator : Emmanuelle Becker                                            #
# https://github.com/isDuke/Biological-Networks-Project-in-Python.git        #
#----------------------------------------------------------------------------#

## CHAPTER N°1 : INTERPRETING A PROTEIN-PROTEIN INTERACTION NETWORK

#----------------------------------------------------------------------------#
"""
Unit tests (pytest) for chapter 1: reading a protein-protein interaction file.

Run from the folder that contains the test files with:
    pytest -v test_read_graph.py

Test files used (in the same folder as this file):
    toy_example.txt   valid graph, 6 interactions (A-B A-C B-C B-D D-E D-F)
    empty.txt         empty file
    no_header.txt     first line (number of interactions) is missing
    bad_header.txt    first line is not a number
    wrong_count.txt   first line says 5 but the file has 6 interactions
    wrong_columns.txt one line has 3 columns instead of 2

All of the test functions are in a Python class for each questions. 
"""

import pytest

from read_graph import (read_interaction_file_dict,
                        read_interaction_file_list,
                        read_interaction_file_mat,
                        read_interaction_file,
                        is_interaction_file)

TOY_STR = "toy_example.txt"

# Expected values for toy_example.txt
# each couple is written in alphabetical order, so A-B and B-A are the same
EDGES_SET = {("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("D", "E"), ("D", "F")}
NEIGHBOUR_DICT = {
    "A": {"B", "C"}, 
    "B": {"A", "C", "D"}, 
    "C": {"A", "B"},
    "D": {"B", "E", "F"}, 
    "E": {"D"}, 
    "F": {"D"}
    }

#----------------------------------------------------------------------------#
## Dictionary (1.2.1)

class TestDict:
    """Tests for read_interaction_file_dict."""

    @pytest.fixture
    def int_dict(self):
        """
        Fixture that reads toy_example.txt with read_interaction_file_dict.
        It replaces the setUp method of unittest: pytest calls it again
        before each test of the class that asks for int_dict.

        Argument:
        None

        Output:
        The dictionary {protein: list of neighbours} of the toy graph.
        """
        return read_interaction_file_dict(TOY_STR)

    def test_type(self, int_dict):
        """
        Check that read_interaction_file_dict returns a dictionary.

        Argument:
        int_dict (dict) : toy graph read as a dictionary (fixture)

        Output:
        None, the assert fails if the result is not a dict.
        """
        assert isinstance(int_dict, dict)

    def test_keys(self, int_dict):
        """
        Check that the keys of the dictionary are exactly the 6 proteins
        of the toy graph (A to F), no more and no less.

        Argument:
        int_dict (dict) : toy graph read as a dictionary (fixture)

        Output:
        None, the assert fails if a protein is missing or added.
        """
        assert set(int_dict.keys()) == set(NEIGHBOUR_DICT.keys())

    def test_neighbours(self, int_dict):
        """
        Check the neighbours of every protein. The neighbours are compared
        as sets, so their order in the list does not matter.

        Argument:
        int_dict (dict) : toy graph read as a dictionary (fixture)

        Output:
        None, the assert fails if a neighbour is missing or added.
        """
        for prot_str, neigh_set in NEIGHBOUR_DICT.items():
            assert set(int_dict[prot_str]) == neigh_set

    def test_undirected(self, int_dict):
        """
        Check that the graph is undirected: each interaction must be
        stored in both directions in the dictionary.

        Argument:
        int_dict (dict) : toy graph read as a dictionary (fixture)

        Output:
        None, the assert fails at the first interaction stored only once.
        """
        # if B is a neighbour of A, then A must be a neighbour of B
        for prot_str, neigh_lst in int_dict.items():
            for neigh_str in neigh_lst:
                assert prot_str in int_dict[neigh_str]

    def test_clean_names(self, int_dict):
        """
        Check that the protein names were cleaned when the file was read.

        Argument:
        int_dict (dict) : toy graph read as a dictionary (fixture)

        Output:
        None, the assert fails if a name still contains blank characters.
        """
        # no "\r", "\n" or spaces in protein names
        for prot_str in int_dict:
            assert prot_str == prot_str.strip()

#----------------------------------------------------------------------------#
## List (1.2.2)

class TestReadList:
    """Tests for read_interaction_file_list."""

    @pytest.fixture
    def int_list(self):
        """
        Fixture that reads toy_example.txt with read_interaction_file_list.
        It is called again before each test that asks for int_list.

        Argument:
        None

        Output:
        The list of couples (protein_a, protein_b) of the toy graph.
        """
        return read_interaction_file_list(TOY_STR)

    def test_type(self, int_list):
        """
        Check that read_interaction_file_list returns a list.

        Argument:
        int_list (list) : toy graph read as a list of couples (fixture)

        Output:
        None, the assert fails if the result is not a list.
        """
        assert isinstance(int_list, list)

    def test_length(self, int_list):
        """
        Check that the list holds the 6 interactions announced on the
        first line of toy_example.txt.

        Argument:
        int_list (list) : toy graph read as a list of couples (fixture)

        Output:
        None, the assert fails if an interaction is lost or added.
        """
        assert len(int_list) == 6

    def test_couples(self, int_list):
        """
        Check that each element of the list contains exactly 2 proteins.

        Argument:
        int_list (list) : toy graph read as a list of couples (fixture)

        Output:
        None, the assert fails at the first element that is not a couple.
        """
        for couple_tup in int_list:
            assert len(couple_tup) == 2

    def test_content(self, int_list):
        """
        Check that the list contains the 6 expected interactions, whatever
        the order of the two proteins inside each couple.

        Argument:
        int_list (list) : toy graph read as a list of couples (fixture)

        Output:
        None, the assert fails if the interactions are not the expected ones.
        """
        # sort each couple so that ("B", "A") becomes ("A", "B")
        found_set = {tuple(sorted(c)) for c in int_list}
        assert found_set == EDGES_SET

#----------------------------------------------------------------------------#
## Adjacency matrix (1.2.3)

class TestReadMat:
    """Tests for read_interaction_file_mat."""

    @pytest.fixture
    def mat_tup(self):
        """
        Fixture that reads toy_example.txt with read_interaction_file_mat.
        The file is read only once per test, even when the test asks for
        both int_mat and prot_list.

        Argument:
        None

        Output:
        The tuple (adjacency matrix, ordered list of proteins).
        """
        return read_interaction_file_mat(TOY_STR)

    @pytest.fixture
    def int_mat(self, mat_tup):
        """
        Fixture that gives the first element of mat_tup.

        Argument:
        mat_tup (tuple) : result of read_interaction_file_mat (fixture)

        Output:
        The adjacency matrix (numpy.ndarray) of the toy graph.
        """
        return mat_tup[0]

    @pytest.fixture
    def prot_list(self, mat_tup):
        """
        Fixture that gives the second element of mat_tup.

        Argument:
        mat_tup (tuple) : result of read_interaction_file_mat (fixture)

        Output:
        The ordered list of proteins, i.e. the rows of the matrix.
        """
        return mat_tup[1]

    def test_vertices(self, prot_list):
        """
        Check that the list of vertices contains the 6 proteins A to F.

        Argument:
        prot_list (list) : ordered list of proteins (fixture)

        Output:
        None, the assert fails if a protein is missing or added.
        """
        assert sorted(prot_list) == ["A", "B", "C", "D", "E", "F"]

    def test_shape(self, int_mat):
        """
        Check that the matrix is square, with one row and one column
        per protein (6 x 6).

        Argument:
        int_mat (numpy.ndarray) : adjacency matrix (fixture)

        Output:
        None, the assert fails if the matrix does not have the right size.
        """
        assert int_mat.shape == (6, 6)

    def test_symmetric(self, int_mat, prot_list):
        """
        Check that the matrix is symmetric (m[i][j] == m[j][i]), which is
        the case for an undirected graph.

        Arguments:
        int_mat (numpy.ndarray) : adjacency matrix (fixture)
        prot_list (list) : ordered list of proteins (fixture)

        Output:
        None, the assert fails at the first cell different from its mirror.
        """
        size_int = len(prot_list)
        for i_int in range(size_int):
            for j_int in range(size_int):
                assert int_mat[i_int][j_int] == int_mat[j_int][i_int]

    def test_diagonal(self, int_mat):
        """
        Check that the diagonal only contains 0: no protein of the toy
        graph interacts with itself.

        Argument:
        int_mat (numpy.ndarray) : adjacency matrix (fixture)

        Output:
        None, the assert fails if a cell of the diagonal is not 0.
        """
        # no homodimer in the toy p-p interaction file
        for i_int in range(6):
            assert int_mat[i_int][i_int] == 0

    def test_edges(self, int_mat, prot_list):
        """
        Check the number of 1 in the matrix, then check that each expected
        interaction is at the right place.

        Arguments:
        int_mat (numpy.ndarray) : adjacency matrix (fixture)
        prot_list (list) : ordered list of proteins (fixture)

        Output:
        None, the assert fails if an interaction is missing or misplaced.
        """
        # each interaction is counted twice in a symmetric matrix
        assert int_mat.sum() == 12
        for x_str, y_str in EDGES_SET:
            i_int = prot_list.index(x_str)
            j_int = prot_list.index(y_str)
            assert int_mat[i_int][j_int] == 1

    def test_no_edge(self, int_mat, prot_list):
        """
        Check that two proteins that do not interact (A and F) have a 0
        in the matrix.

        Arguments:
        int_mat (numpy.ndarray) : adjacency matrix (fixture)
        prot_list (list) : ordered list of proteins (fixture)

        Output:
        None, the assert fails if the cell A-F is not 0.
        """
        i_int = prot_list.index("A")
        j_int = prot_list.index("F")
        assert int_mat[i_int][j_int] == 0

#----------------------------------------------------------------------------#
## Quadruplet (1.2.4)

class TestReadAll:
    """Tests for read_interaction_file (the quadruplet)."""

    def test_quadruplet(self):
        """
        Check that read_interaction_file returns 4 elements, and that they
        are the same as the ones given by the other reading functions.

        Argument:
        None

        Output:
        None, the assert fails at the first element that does not match.
        """
        res_tup = read_interaction_file(TOY_STR)
        assert len(res_tup) == 4
        d_int, l_int, m_int, l_som = res_tup
        assert d_int == read_interaction_file_dict(TOY_STR)
        assert l_int == read_interaction_file_list(TOY_STR)
        assert m_int.shape == (6, 6)
        assert sorted(l_som) == ["A", "B", "C", "D", "E", "F"]

#----------------------------------------------------------------------------#
## File format (1.2.7)

class TestIsInteractionFile:
    """Tests for is_interaction_file."""

    def test_valid(self):
        """
        Check that a file in the right format is accepted.

        Argument:
        None

        Output:
        None, the assert fails if toy_example.txt is rejected.
        """
        assert is_interaction_file(TOY_STR)

    def test_empty(self):
        """
        Check that an empty file is rejected.

        Argument:
        None

        Output:
        None, the assert fails if empty.txt is accepted.
        """
        assert not is_interaction_file("empty.txt")

    def test_no_header(self):
        """
        Check that a file without the number of interactions on its first
        line is rejected.

        Argument:
        None

        Output:
        None, the assert fails if no_header.txt is accepted.
        """
        assert not is_interaction_file("no_header.txt")

    def test_bad_header(self):
        """
        Check that a file whose first line is not a number is rejected.

        Argument:
        None

        Output:
        None, the assert fails if bad_header.txt is accepted.
        """
        assert not is_interaction_file("bad_header.txt")

    def test_wrong_count(self):
        """
        Check that a file whose first line does not match the real number
        of interactions is rejected.

        Argument:
        None

        Output:
        None, the assert fails if wrong_count.txt is accepted.
        """
        assert not is_interaction_file("wrong_count.txt")

    def test_wrong_columns(self):
        """
        Check that a file with a line of 3 columns is rejected.

        Argument:
        None

        Output:
        None, the assert fails if wrong_columns.txt is accepted.
        """
        assert not is_interaction_file("wrong_columns.txt")

#----------------------------------------------------------------------------#
# Main

if __name__ == "__main__":
    pytest.main(["-v", __file__])