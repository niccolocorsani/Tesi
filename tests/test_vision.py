import unittest

import os
from not_for_thesis.vision import get_end_points

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VisionTestCase(unittest.TestCase):

    ####TODO appena torna l'internt
    def test_intersection_always_be_or_2_or_4_for_each_image_in_folder(self):
        pass
        # directory = ROOT_DIR  + './tests/test_intersections_image'
        #
        # for filename in os.listdir(directory):
        #     f = os.path.join(directory, filename)
        #     # checking if it is a file
        #     if os.path.isfile(f):
        #         get_end_points(f)

    def test_get_end_points(self):
        get_end_points(ROOT_DIR + '/tests/test_intersections_image/molte_forme_need_for_git.png')
