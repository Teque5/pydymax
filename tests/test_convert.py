#!/usr/bin/env python3
'''Test cases for pydymax conversion utilities.'''
import unittest
import numpy as np

import dymax

class RunTypical(unittest.TestCase):
    def test_ll2spherical(self):
        '''south pole check'''
        theta, phi = dymax.lonlat2spherical(0, -90)
        self.assertAlmostEqual(theta, np.pi)
        self.assertAlmostEqual(phi, 0)

if __name__ == '__main__':
    unittest.main()
