import os
from business_logic.vision import save_image_to_edged_black_on_white_background_and_return_vertex_of_white
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_save_image_to_edged_black_on_white_background_and_return_vertex_of_white():
    save_image_to_edged_black_on_white_background_and_return_vertex_of_white(ROOT_DIR + '/pagine/Schermata 2022-03-20 alle 14.25.17.png'
                                                      ,ROOT_DIR + '/output_files/black_and_white_image.png')
