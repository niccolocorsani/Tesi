import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytesseract import Output
from termcolor import colored
# Import required packages
import pytesseract

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ROOT_DIR + "/resources/google-auth.json"

import re


def check_name(name: str):
    return re.match(r"^[\-'a-zA-Z ]+$", name) is not None


#### Quando faccio draw, putText, anche se l'immagine è passata come riferimento.... L'immagine viene modificata..

def detect_text_best_accuracy(path_of_img):

    dic_opencv = detect_text_opencv(path_of_img)

    dic_google = detect_text(path_of_img)

    new_dic = merge_two_dictionaies(dic_opencv, dic_google)

    return new_dic



## TODO raffinare algoritmo in modo che se due parole sono distanti meno di n pixel allora vanno nello stesso contorno
def detect_text(path_of_img):
    """

    :rtype: return a dictionary where the keys are the text scanned and the values are the vertex of each detected text
    """
    ##  https://cloud.google.com/vision/docs/ocr
    from google.cloud import vision  # pip install --upgrade google-cloud-vision (se ci sono problemi di import)
    import io
    client = vision.ImageAnnotatorClient()
    with io.open(path_of_img, 'rb') as image_file:
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
            continue  ## TODO da rivedere
        my_list.append(text.description)
        vertices = ([(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])
        text_vertex_dic[text.description] = vertices
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    ####TO debug
    immmm = cv2.imread(path_of_img)
    for word in text_vertex_dic.keys():
        top_left = text_vertex_dic[word][3]
        bottom_right = text_vertex_dic[word][1]
        cv2.rectangle(immmm, top_left, bottom_right, (255, 0, 0), 1)

    ####TO debug
    plt.imshow(immmm)
    plt.show()
    return text_vertex_dic


def detect_text_opencv(path_of_img):
    # Read image from which text needs to be extracted
    img = cv2.imread(path_of_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    im2 = img.copy()

    text_vertex_dic = {}

    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    n_boxes = len(d['level'])

    for i in range(n_boxes):

        text = d['text'][i]
        if len(d['text'][i]) != 0 and d['text'][i] != ' ' and d['text'][i] != '—_|':
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            bottom_left = (x, y)
            bottom_right = (x + w, y)
            top_right = (x + w, y + h)
            top_left = (x, y + h)

            text_vertex_dic[text] = [bottom_left, bottom_right, top_right, top_left]
            cv2.rectangle(im2, top_left, bottom_right, (255, 0, 0), 2)
            center = find_center_of_rectangle(top_left, bottom_right)
            cv2.circle(im2, (center), 10, (0, 0, 255), -1)
    plt.imshow(im2)
    plt.show()

    return text_vertex_dic


def merge_two_dictionaies(google_dic, cv2_dic):
    merged_dic = google_dic.copy()  # start with keys and values of x
    merged_dic.update(cv2_dic)
    return merged_dic


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


def check_if_text_is_inside_contours(dic_of_output_of_detect_text, img, contours_of_squares=None):
    # # remove biggest contour (the one of all image)
    # contours_of_squares = sorted(contours_of_squares, key=cv2.contourArea, reverse=True)
    # contours_of_squares.pop(0)

    for img_name in dic_of_output_of_detect_text.copy().keys():
        print(colored(img_name, 'red'))
        dic_of_word = dic_of_output_of_detect_text[img_name].copy()
        for word in dic_of_word.copy().keys():
            last_contour = 0
            for contour in contours_of_squares:
                last_contour = last_contour + 1
                top_left = dic_of_word[word][3]
                bottom_right = dic_of_word[word][1]
                center_of_rectangle = find_center_of_rectangle(top_left, bottom_right)
                distance = cv2.pointPolygonTest(contour, center_of_rectangle, True)
                if distance > -12:
                    break
                if last_contour == len(contours_of_squares):
                    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 1)
                    dic_of_output_of_detect_text[img_name].pop(word)
                    cv2.imwrite(ROOT_DIR + '/not_inside_contours/' + img_name, img)

    return dic_of_output_of_detect_text


# {img_name: {node_name: list_of_words}...{node_name: list_of_words}}
def give_semantic_to_nodes(dic_of_original_image_and_associated_words, dic_added_info,
                           contours_of_squares, img_to_debug):
    dic_node_elements = {}
    node = 0

    for img_name in dic_of_original_image_and_associated_words.copy().keys():
        print(colored(img_name, 'red'))
        dic_of_word = dic_of_original_image_and_associated_words[img_name].copy()
        for contour in contours_of_squares:
            last_word = 0
            node = node + 1
            # cv2.putText(img_to_debug, str(node), find_center_of_one_contour(contour),
            #             cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2)
            # cv2.drawContours(img_to_debug, [contour], 0,
            #                  (0, 0, 255), 5)
            # plt.imshow(img_to_debug)
            # plt.show()

            for word in dic_of_word.copy().keys():
                last_word = last_word + 1
                top_left = dic_of_word[word][3]
                bottom_right = dic_of_word[word][1]
                center_of_rectangle = find_center_of_rectangle(top_left, bottom_right)
                distance = cv2.pointPolygonTest(contour, center_of_rectangle, True)
                if distance > -12:
                    if node not in dic_node_elements:  # check if key does not exist
                        dic_node_elements[node] = []
                    dic_node_elements[node].append(word)

    node = 0

    for img_name in dic_added_info.copy().keys():
        print(colored(img_name, 'red'))
        dic_of_word = dic_added_info[img_name].copy()
        for contour in contours_of_squares:
            last_word = 0
            node = node + 1
            for word in dic_of_word.copy().keys():

                last_word = last_word + 1
                top_left = dic_of_word[word][3]
                bottom_right = dic_of_word[word][1]
                center_of_rectangle = find_center_of_rectangle(top_left, bottom_right)
                distance = cv2.pointPolygonTest(contour, center_of_rectangle, True)

                if distance > -12:
                    if node not in dic_node_elements:  # check if key does not exist
                        dic_node_elements[node] = []
                    dic_node_elements[node].append(word)
                    print('added')
                    print(word)

    return dic_node_elements


def get_inner_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == '__main__':
    path_name_original_images = os.listdir(ROOT_DIR + '/gray_images/')



    print('do')

    # mergiare l'output di google con quello di tesseract

    print('fine')

    for path_name in path_name_original_images:

        ## Leggo immagini con STESSO nome da cartelle: immagini originali e immagini modificate
        original_img = cv2.imread(ROOT_DIR + '/gray_images/' + path_name)
        original_with_nothing_to_debug = original_img.copy()
        img_square = cv2.imread(ROOT_DIR + '/squares_image/' + path_name)
        ## Ridimensiono le immagini con stessa dimensione (quadrati e originale)
        heigth_original_img, width_original_img, channels = original_img.shape
        img_resized_square = cv2.resize(img_square, (width_original_img, heigth_original_img))
        ## Riscrivo immagine modificata su cartella apposita per google, perchè la funzione detect_text accetta solo path come argomento
        cv2.imwrite(ROOT_DIR + '/squares_for_google/' + path_name, img_resized_square)
        ## Prendo i contorni dei quadrati
        contours_of_squares = get_inner_contour(img_resized_square)
        contours_of_squares = sorted(contours_of_squares, key=cv2.contourArea,
                                     reverse=True)  # remove the biggest contour (the one of all image) qui non serve    # contours_of_squares.pop(0) In questo caso, stranamente non serve.....remove biggest contour (the one of all image)remove biggest contour (the one of all image)
        ## creo dizionario formato: {img_name: {parola: coordinate_parola}}
        dic_of_original_image_and_associated_words = {path_name: detect_text(ROOT_DIR + '/gray_images/' + path_name)}
        dic_of_squared_image_and_associated_words = {
            path_name: detect_text(ROOT_DIR + '/squares_for_google/' + path_name)}
        #### Da mettere in test
        dic_of_original_image_and_associated_words = check_if_text_is_inside_contours(
            dic_of_original_image_and_associated_words,
            original_img,
            contours_of_squares)

        n = 0
        for contour in contours_of_squares:
            n = n + 1
            cv2.drawContours(original_img, [contour], 0,
                             (0, 0, 255), 5)
            center = find_center_of_one_contour(contour)
            cv2.putText(original_img, str(n), center,
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2)

        # plt.imshow(original_img)
        # plt.show()
        #### Da mettere in test

        dic_img_nodes_semantic = {path_name: give_semantic_to_nodes(dic_of_original_image_and_associated_words,
                                                                    dic_of_squared_image_and_associated_words,
                                                                    contours_of_squares,
                                                                    original_with_nothing_to_debug)}

        print('fine')
