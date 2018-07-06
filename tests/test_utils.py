import unittest

import torch

from utils import cartesian_view


class TestCartesianView(unittest.TestCase):
    def test_cartesian_view_0(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 0)
        self.assertEqual(c.size(), (2, 3, 4, 7))
        self.assertEqual(d.size(), (2, 3, 5, 9))

    def test_cartesian_view_1(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 1)
        self.assertEqual(c.size(), (2, 3, 4, 7))
        self.assertEqual(d.size(), (2, 3, 5, 9))

    def test_cartesian_view_2(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 2)
        self.assertEqual(c.size(), (2, 3, 4, 7))
        self.assertEqual(d.size(), (2, 3, 5, 9))

    def test_cartesian_view_3(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 3)
        self.assertEqual(c.size(), (2, 3, 4, 5, 7))
        self.assertEqual(d.size(), (2, 3, 4, 5, 9))

    def test_cartesian_view_4(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 4)
        self.assertEqual(c.size(), (2, 3, 4, 5, 7, 9))
        self.assertEqual(d.size(), (2, 3, 4, 5, 7, 9))

    def test_cartesian_view_5(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 5)
        self.assertEqual(c.size(), (2, 3, 4, 5, 7, 9))
        self.assertEqual(d.size(), (2, 3, 4, 5, 7, 9))
