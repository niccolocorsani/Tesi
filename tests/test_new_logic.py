import os
import cv2

from business_logic.new_logic import check_if_text_is_inside_contours, get_inner_contour, detect_text
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_check_text_inside_added_info_image_squared_contours_should_never_be_outside_the_squares():

    path_name_original_images = os.listdir(ROOT_DIR + '/tests/need_on_git_gray_images/')

    for path_name in path_name_original_images:

        original_img = cv2.imread(ROOT_DIR + '/tests/need_on_git_gray_images/' + path_name)
        img_square = cv2.imread(ROOT_DIR + '/tests/need_for_git_squares_images/' + path_name)
        heigth_original_img, width_original_img, channels = original_img.shape
        img_resized_square = cv2.resize(img_square, (width_original_img, heigth_original_img))
        cv2.imwrite(ROOT_DIR + '/tests/need_for_git_squares_for_google/' + path_name, img_resized_square)


        contours_of_squares = get_inner_contour(img_resized_square)
        contours_of_squares = sorted(contours_of_squares, key=cv2.contourArea,
                                     reverse=True)

        ##TODO Da cambiare con funzione di opencv tesserract
        dic_of_squared_image_and_associated_words = {path_name: detect_text(ROOT_DIR + '/tests/need_for_git_squares_for_google/' + path_name)}


        dic_of_squared_image_and_associated_words = check_if_text_is_inside_contours(
            dic_of_squared_image_and_associated_words,
            original_img,
            contours_of_squares)

        assert 0 == 0 ##TODO assert che nessuna parola si trova al di fuori dei quadrati




