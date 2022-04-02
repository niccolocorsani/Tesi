import unittest

import os
from business_logic.vision import save_image_to_edged_black_on_white_background_and_return_vertex_of_white, \
    get_end_points

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





class VisionTestCase(unittest.TestCase):


    def test_get_end_points(self):
        get_end_points(ROOT_DIR + '/molte_forme_need_for_git.png')

    def test_save_image_to_edged_black_on_white_background_and_return_vertex_of_white(self):
        save_image_to_edged_black_on_white_background_and_return_vertex_of_white(ROOT_DIR + '/pagine/Schermata_need_on_git.png'
                                                          ,ROOT_DIR + '/output_files/black_and_white_image_need_on_git.png')
        self.assertTrue('black_and_white_image_need_on_git.png' in os.listdir(ROOT_DIR + "/output_files"))



