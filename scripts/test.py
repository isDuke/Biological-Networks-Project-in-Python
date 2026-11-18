"""
Unit tests for chapter 1: reading a protein-protein interaction file.

Run from the folder that contains the test files with:
    python3 -m unittest test_interaction.py -v

Test files used (in the same folder as this file):
    toy_example.txt   valid graph, 6 interactions (A-B A-C B-C B-D D-E D-F)
    empty.txt         empty file
    no_header.txt     first line (number of interactions) is missing
    bad_header.txt    first line is not a number
    wrong_count.txt   first line says 5 but the file has 6 interactions
    wrong_columns.txt one line has 3 columns instead of 2
"""

import unittest

from read_graph import (read_interaction_file_dict,
                  read_interaction_file_list,
                  read_interaction_file_mat,
                  read_interaction_file,
                  is_interaction_file)

TOY_STR = "toy_example.txt"

# Expected values for toy_example.txt
# each couple is written in alphabetical order, so A-B and B-A are the same
EDGES_SET = {("A", "B"), ("A", "C"), ("B", "C"),
             ("B", "D"), ("D", "E"), ("D", "F")}
NEIGHBOUR_DICT = {"A": {"B", "C"}, "B": {"A", "C", "D"}, "C": {"A", "B"},
                  "D": {"B", "E", "F"}, "E": {"D"}, "F": {"D"}}


class TestDict(unittest.TestCase):
    """Tests for read_interaction_file_dict."""

    def setUp(self):
        self.int_dict = read_interaction_file_dict(TOY_STR)

    def test_type(self):
        self.assertIsInstance(self.int_dict, dict)

    def test_keys(self):
        self.assertEqual(set(self.int_dict.keys()),
                         set(NEIGHBOUR_DICT.keys()))

    def test_neighbours(self):
        for prot_str, neigh_set in NEIGHBOUR_DICT.items():
            self.assertEqual(set(self.int_dict[prot_str]), neigh_set)

    def test_undirected(self):
        # if B is a neighbour of A, then A must be a neighbour of B
        for prot_str, neigh_lst in self.int_dict.items():
            for neigh_str in neigh_lst:
                self.assertIn(prot_str, self.int_dict[neigh_str])

    def test_clean_names(self):
        # no "\r", "\n" or spaces in protein names
        for prot_str in self.int_dict:
            self.assertEqual(prot_str, prot_str.strip())


class TestReadList(unittest.TestCase):
    """Tests for read_interaction_file_list."""

    def setUp(self):
        self.int_list = read_interaction_file_list(TOY_STR)

    def test_type(self):
        self.assertIsInstance(self.int_list, list)

    def test_length(self):
        self.assertEqual(len(self.int_list), 6)

    def test_couples(self):
        for couple_tup in self.int_list:
            self.assertEqual(len(couple_tup), 2)

    def test_content(self):
        # sort each couple so that ("B", "A") becomes ("A", "B")
        found_set = {tuple(sorted(c)) for c in self.int_list}
        self.assertEqual(found_set, EDGES_SET)


class TestReadMat(unittest.TestCase):
    """Tests for read_interaction_file_mat."""

    def setUp(self):
        self.int_mat, self.prot_list = read_interaction_file_mat(TOY_STR)

    def test_vertices(self):
        self.assertEqual(sorted(self.prot_list), ["A", "B", "C", "D", "E", "F"])

    def test_shape(self):
        self.assertEqual(self.int_mat.shape, (6, 6))

    def test_symmetric(self):
        size_int = len(self.prot_list)
        for i_int in range(size_int):
            for j_int in range(size_int):
                self.assertEqual(self.int_mat[i_int][j_int], self.int_mat[j_int][i_int])

    def test_diagonal(self):
        # no homodimer in da p-p interaction file
        for i_int in range(6):
            self.assertEqual(self.int_mat[i_int][i_int], 0)

    def test_edges(self):
        # each interaction is counted twice in a symmetric matrix
        self.assertEqual(self.int_mat.sum(), 12)
        for x_str, y_str in EDGES_SET:
            i_int = self.prot_list.index(x_str)
            j_int = self.prot_list.index(y_str)
            self.assertEqual(self.int_mat[i_int][j_int], 1)

    def test_no_edge(self):
        i_int = self.prot_list.index("A")
        j_int = self.prot_list.index("F")
        self.assertEqual(self.int_mat[i_int][j_int], 0)


class TestReadAll(unittest.TestCase):
    """Tests for read_interaction_file (the quadruplet)."""

    def test_quadruplet(self):
        res_tup = read_interaction_file(TOY_STR)
        self.assertEqual(len(res_tup), 4)
        d_int, l_int, m_int, l_som = res_tup
        self.assertEqual(d_int, read_interaction_file_dict(TOY_STR))
        self.assertEqual(l_int, read_interaction_file_list(TOY_STR))
        self.assertEqual(m_int.shape, (6, 6))
        self.assertEqual(sorted(l_som), ["A", "B", "C", "D", "E", "F"])


class TestIsInteractionFile(unittest.TestCase):
    """Tests for is_interaction_file."""

    def test_valid(self):
        self.assertTrue(is_interaction_file(TOY_STR))

    def test_empty(self):
        self.assertFalse(is_interaction_file("empty.txt"))

    def test_no_header(self):
        self.assertFalse(is_interaction_file("no_header.txt"))

    def test_bad_header(self):
        self.assertFalse(is_interaction_file("bad_header.txt"))

    def test_wrong_count(self):
        self.assertFalse(is_interaction_file("wrong_count.txt"))

    def test_wrong_columns(self):
        self.assertFalse(is_interaction_file("wrong_columns.txt"))


if __name__ == "__main__":
    unittest.main()