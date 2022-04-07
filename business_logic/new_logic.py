import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from termcolor import colored

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ROOT_DIR + "/resources/google-auth.json"


def detect_text(path):
    """

    :rtype: return a dictionary where the keys are the text scanned and the values are the vertex of each detected text
    """
    ##  https://cloud.google.com/vision/docs/ocr
    from google.cloud import vision  # pip install --upgrade google-cloud-vision (se ci sono problemi di import)
    import io
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    try:
        response = client.text_detection(image=image)
    except:
        pass
    texts = response.text_annotations
    my_list = []
    text_vertex_dic = {}
    i = 0
    for text in texts:
        if i == 0:
            i = i + 1
            continue
        my_list.append(text.description)
        vertices = ([(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])
        text_vertex_dic[text.description] = vertices
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return text_vertex_dic


def find_center_of_one_contour(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]


##TODO vedere se è possibile mettere precisione maggiore con float....
def find_center_of_rectangle(top_left, bottom_right):
    x = (top_left[0] + bottom_right[0]) / 2
    y = (top_left[1] + bottom_right[1]) / 2

    return int(x), int(y)


def read_text_and_put_rectangle_over_it_on_withe_img(input_path):
    path_name = os.listdir(ROOT_DIR + input_path)
    ### Questo mi servira dopo per verificare che tutto stia effettivamente dentro i quadrati

    for path_names in path_name:
        text_vertex_dic = detect_text(ROOT_DIR + "/pagine/" + path_names)
        image = cv2.imread(ROOT_DIR + "/pagine/" + path_names)
        all_words_of_image = text_vertex_dic.keys()

        height, width, channels = image.shape

        white_img = np.ones([height, width, channels], dtype=np.uint8)
        white_img.fill(255)  # or img[:] = 255

        for word in all_words_of_image:
            # Start coordinate, here (100, 50)
            # represents the top left corner of rectangle
            top_left = text_vertex_dic.get(word)[3]

            # Ending coordinate, here (125, 80)
            # represents the bottom right corner of rectangle
            bottom_right = text_vertex_dic.get(word)[1]

            color = (0, 0, 0)

            white_img = cv2.rectangle(white_img, top_left, bottom_right, color, 1)

            center_of_rectangle = find_center_of_rectangle(top_left=top_left, bottom_right=bottom_right)

            x = int(center_of_rectangle[0])
            y = int(center_of_rectangle[1])

            cv2.putText(white_img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 10)


def read_text_and_put_rectangle_over_it_on_original_img(input_path, output_path):
    ## input path should be  read_text_and_put_rectangle_over_it_on_original_img("/pagine_piu_pulite/",'/images_with_rectangle/')

    path_name = os.listdir(ROOT_DIR + input_path)
    ### Questo mi servira dopo per verificare che tutto stia effettivamente dentro i quadrati

    for path_names in path_name:
        text_vertex_dic = detect_text(ROOT_DIR + input_path + path_names)
        image = cv2.imread(ROOT_DIR + input_path + path_names)
        all_words_of_image = text_vertex_dic.keys()

        for word in all_words_of_image:
            # Start coordinate, here (100, 50)
            # represents the top left corner of rectangle
            top_left = text_vertex_dic.get(word)[3]

            # Ending coordinate, here (125, 80)
            # represents the bottom right corner of rectangle
            bottom_right = text_vertex_dic.get(word)[1]

            color = (0, 0, 255)

            image = cv2.rectangle(image, top_left, bottom_right, color, 1)

        cv2.imwrite(ROOT_DIR + output_path + path_names, image)


def convert_all_path_images_to_edged_and_save_it(input_path, output_path):
    path_name = os.listdir(ROOT_DIR + input_path)

    for path_names in path_name:
        # format of path should be "/pagine_piu_pulite/"
        img = cv2.imread(ROOT_DIR + input_path + path_names, 0)
        edges = cv2.Canny(img, 100, 255)  # --- image containing edges ---
        inverted_image = (255 - edges)  # --- inverte image ---
        # format of path should be "/edged_images/"
        cv2.imwrite(ROOT_DIR + output_path + path_names, inverted_image)


def detect_text_of_folder_label_it(path_folder):
    # path_folder should be in format /pathhhh/
    from google.cloud import vision  # pip install --upgrade google-cloud-vision (se ci sono problemi di import)
    import io
    client = vision.ImageAnnotatorClient()
    ## input path should be  read_text_and_put_rectangle_over_it_on_original_img("/pagine_piu_pulite/",'/images_with_rectangle/')

    path_name = os.listdir(ROOT_DIR + path_folder)

    dic_path_name_sub_dic = {}

    for path_names in path_name:
        with io.open(ROOT_DIR + path_folder + path_names, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        try:
            response = client.text_detection(image=image)
        except:
            pass
        texts = response.text_annotations
        my_list = []
        text_vertex_dic = {}
        i = 0
        for text in texts:
            if i == 0:
                i = i + 1
                continue
            my_list.append(text.description)
            vertices = ([(vertex.x, vertex.y)
                         for vertex in text.bounding_poly.vertices])
            text_vertex_dic[text.description] = vertices

        dic_path_name_sub_dic[path_names] = text_vertex_dic

    return dic_path_name_sub_dic


def check_if_text_is_inside_contours(dic_of_output_of_detect_text, img, contours_of_squares=None):

    # # remove biggest contour (the one of all image)
    # contours_of_squares = sorted(contours_of_squares, key=cv2.contourArea, reverse=True)
    # contours_of_squares.pop(0)

    for img_name in dic_of_output_of_detect_text.copy().keys():
        print(colored(img_name, 'red'))
        dic_of_word = dic_of_output_of_detect_text[img_name].copy()
        for word in dic_of_word.copy().keys():
            last_contour = 0
            print(word)
            for contour in contours_of_squares:
                last_contour = last_contour + 1
                top_left = dic_of_word[word][3]
                bottom_right = dic_of_word[word][1]
                center_of_rectangle = find_center_of_rectangle(top_left, bottom_right)
                distance = cv2.pointPolygonTest(contour, center_of_rectangle, True)
                if distance > -12:
                    break
                if last_contour == len(contours_of_squares):
                    # print(word)
                    # print('is not inside a contour')
                    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 1)
                    dic_of_output_of_detect_text[img_name].pop(word)
                    plt.imshow(img)
                    plt.show()

    return dic_of_output_of_detect_text


def get_inner_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == '__main__':

    #  original_img = cv2.imread(ROOT_DIR + '/gray_images/Schermata 2022-03-20 alle 14.18.23.png')
    original_img = cv2.imread(ROOT_DIR + '/img.png')

    img_square = cv2.imread(ROOT_DIR + '/squares_image/ooo.png')

    heigth_original_img, width_original_img, channels = original_img.shape

    img_resized_square = cv2.resize(img_square, (width_original_img, heigth_original_img))

    contours_of_squares = get_inner_contour(img_resized_square)

    # remove biggest contour (the one of all image) automaticamente -- se faccio "cmd option L" e ho questo formato di commento me lo lascia cosi
    contours_of_squares = sorted(contours_of_squares, key=cv2.contourArea, reverse=True)
    # contours_of_squares.pop(0) In questo caso, stranamente non serve.....
    # remove biggest contour (the one of all image)

    ### Le parole qui trovate a questo punto possono essere riscritte con il metodo put text nell'altre immagini
    # prima di chiamare nuovamente ill metdo detect_text....
    dic_of_squared_and_added_info_images_name_with_all_words_associated_with_it = detect_text_of_folder_label_it(
        '/squares_image/')

    dic_of_original_images_name_with_all_words_associated_with_it = detect_text_of_folder_label_it('/gray_images/')

    n = 0
    for contour in contours_of_squares:
        n = n + 1
        cv2.drawContours(original_img, [contour], 0,
                         (0, 0, 255), 5)
        center = find_center_of_one_contour(contour)
        cv2.putText(original_img, str(n), center,
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2)
        plt.imshow(original_img)
        plt.show()

plt.imshow(original_img)
plt.show()

#### Da mettere in test
check_if_text_is_inside_contours(dic_of_original_images_name_with_all_words_associated_with_it, original_img,
                                 contours_of_squares)
print('fine')
#### Da mettere in test
