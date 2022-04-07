import unittest

import os
from business_logic.vision import save_image_to_edged_black_on_white_background_and_return_vertex_of_white, \
    get_end_points

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VisionTestCase(unittest.TestCase):

    ####TODO appena torna l'internt
    def test_intersection_always_be_or_2_or_4_for_each_image_in_folder(self):

        directory = './test_intersections_image'

        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                get_end_points(f)



    def test_get_end_points(self):
        get_end_points(ROOT_DIR + '/molte_forme_need_for_git.png')

