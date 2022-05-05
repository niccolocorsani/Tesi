import os
import time
import matplotlib.pyplot as plt
import cv2
from termcolor import colored
import pytesseract
from business_logic.correct_csv import specify_physical_object, get_all_starts, get_all_exits
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ROOT_DIR + "/resources/google-auth.json"

plt.rcParams['figure.dpi'] = 1000  ## To increase matplotlib image definition


# def plot_image_with_background(img_path_from_root_normal, img_path_from_root_squared,number_of_contour):
#
#     original_img = cv2.imread(ROOT_DIR + img_path_from_root_normal)
#     img_square = cv2.imread(ROOT_DIR + img_path_from_root_squared)
#     heigth_original_img, width_original_img, channels = original_img.shape
#     img_resized_square = cv2.resize(img_square, (width_original_img, heigth_original_img))
#     contours_of_squares1 = get_inner_contour(img_resized_square)
#     contours_of_squares1 = sorted(contours_of_squares1, key=cv2.contourArea,
#                                   reverse=True)
#     contours_of_squares = remove_contour_under_area(contours_of_squares1, 400)
#     cv2.drawContours(original_img, contours_of_squares[number_of_contour], 0,
#                      (0, 0, 255), 5)

def my_text_detection_tessereact(img, contours):
    im2 = img.copy()
    node_words_dic = {}
    im3 = im2.copy()
    node = 0
    for cnt in contours:
        node = node + 1
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(
            cropped)  ## Se composto da più parole ritorna sempre una stringa, ma con \n\n

        values = text.split('\n')
        good_values = []
        for value in values:
            if len(value) > 0:
                good_values.append(value)

        # cv2.drawContours(im3, [cnt], 0,
        #                  (0, 0, 255), 5)
        # plt.imshow(im3)
        # plt.show()

        node_words_dic[node] = good_values

    return node_words_dic


def detect_text(path_of_img):
    """
    :rtype: return a dictionary where the keys are the text scanned and the values are the vertex of each detected text
    """
    ##  https://cloud.google.com/vision/docs/ocr
    from google.cloud import vision  # pip install --upgrade google-cloud-vision (se ci sono problemi di import)
    import io

    ##  raffinare algoritmo in modo che se due parole sono distanti meno di n pixel e si trovano sulla stessa y allora vanno nello stesso contorno

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
    str = 'NODE'
    first_node = True

    for text in texts:
        if i == 0:
            i = i + 1
            continue  ## TODO da rivedere

        if text.description == 'NODE' and first_node == False:
            vertices = ([(vertex.x, vertex.y)
                         for vertex in text.bounding_poly.vertices])
            text_vertex_dic[str.lower()] = vertices
            str = 'NODE'

        if text.description != 'NODE':
            str = str + text.description

        first_node = False

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
        cv2.rectangle(immmm, top_left, bottom_right, (255, 0, 0), 4)
    # plt.imshow(immmm)
    # plt.show()
    ####TO debug

    return text_vertex_dic


def merge_two_dictionaries(the_more_accurate_dic,
                           the_one_from_wich_take_value_only_if_the_value_of_same_key_is_empty_on_the_other_dic):
    merged_dic = {}
    for img_name in the_more_accurate_dic:
        merged_dic[img_name] = {}
        for node in the_more_accurate_dic[img_name]:
            if len(the_more_accurate_dic[img_name][node]) == 0:
                merged_dic[img_name][node] = \
                    the_one_from_wich_take_value_only_if_the_value_of_same_key_is_empty_on_the_other_dic[
                        img_name][node]
            else:
                merged_dic[img_name][node] = the_more_accurate_dic[img_name][node]

    return merged_dic


def find_center_of_one_contour(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]


def find_center_of_rectangle(top_left, bottom_right):
    ## vedere se è possibile mettere precisione maggiore con float....

    x = (top_left[0] + bottom_right[0]) / 2
    y = (top_left[1] + bottom_right[1]) / 2

    return int(x), int(y)


def give_semantic_to_nodes(dic_added_info,
                           contours_of_squares, my_img_to_debug):
    # {img_name: {node_name: list_of_words}...{node_name: list_of_words}}
    dic_node_elements = {}
    my_img_to_debug_squares = my_img_to_debug.copy()

    node = 0

    for img_name in dic_added_info.copy().keys():
        print(colored(img_name, 'red'))
        dic_of_word = dic_added_info[img_name].copy()
        for contour in contours_of_squares:
            node = node + 1

            cv2.putText(my_img_to_debug_squares, str(node), find_center_of_one_contour(contour),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.drawContours(my_img_to_debug_squares, [contour], 0,
                             (0, 0, 255), 5)

            text_of_contour = ''
            if node not in dic_node_elements:  # check if key does not exist
                dic_node_elements[node] = []
            for word in dic_of_word.copy().keys():
                top_left = dic_of_word[word][3]
                bottom_right = dic_of_word[word][1]
                center_of_rectangle = find_center_of_rectangle(top_left, bottom_right)
                distance = cv2.pointPolygonTest(contour, center_of_rectangle, True)

                if distance > -6:
                    dic_node_elements[node].append(word.lower())
                    text_of_contour = text_of_contour + '-' + word

                #### To generate image to check at the end
                if word == list(dic_of_word.keys())[-1]:
                    x, y = find_center_of_one_contour(contour)
                    cv2.putText(my_img_to_debug_squares, text_of_contour, [x - 20, y],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        # plt.imshow(my_img_to_debug_squares)
        # plt.title("Squared Image check if detected text is correct")
        plt.show()
        cv2.imwrite(ROOT_DIR + '/check_if_are_the_same_of_manualy_drawn/' + 'suqared_' + img_name,
                    my_img_to_debug_squares)
        #### To generate image to check at the end

    return dic_node_elements


def generate_lines(dic_img_nodes_semantic, contours, img, original_img):
    ## Se per caso non rispetta il pattern da eccezione
    nodes_csv = []
    edges_csv = []
    measures_csv = []
    added_nodes = []
    added_ends = []

    img_path = ''

    for img_name in dic_img_nodes_semantic.keys():
        img_path = img_name
        flusso_num = 0

        for node in dic_img_nodes_semantic[img_name]:

            try:

                for string_pattern in dic_img_nodes_semantic[img_name][node]:

                    string_pattern = string_pattern.lower()
                    node_type = ''

                    if len(dic_img_nodes_semantic[img_name][node]) != 1: raise Exception(
                        "not only one element in string_pattern")

                    node_name = ''
                    end = ''
                    measure_of = ''
                    type = 'Not-specified'  ## Se non è specificato ha questo valore
                    bad_ends = []
                    bad_measures = []
                    elements = string_pattern.split(',')

                    if '?' in string_pattern:
                        raise Exception("Node has issue ????")
                    if ':' not in string_pattern:
                        raise Exception("Node doesn'contain :")
                    if 'measu' not in string_pattern:
                        if 'star' in elements[0]:  ## Nodo star non può avere end
                            node_type = 'start'
                            node_name = elements[0]
                        elif 'exit' in elements[0]:  ## Nodo exit deve avere un end per forza
                            node_type = 'exit'
                            elements = string_pattern.split(',')
                            node_name = elements[0]
                            end = elements[1]
                        else:  ## Nodo normale con più end
                            if len(elements) == 1:
                                raise Exception("Should be a start maybe!!")
                            if '+' in elements[
                                1]:  ## TODO qui con un nodo tipo NODE:cl2 va in exception dicendo elemnts[1] out of bound
                                node_type = 'normal_multiple_end'
                                node_name = elements[0]
                                bad_ends = elements[1].split('+')
                                if len(bad_ends) == 0:  raise Exception("No ends in second element (line 235)")
                            else:  ## Nodo normale con un end
                                node_type = 'normal_one_end'
                                node_name = elements[0]
                                end = elements[1]

                    if 'measu' in string_pattern:  ## Nodo measure
                        if '+' in elements[1]:
                            node_type = 'measure_multiple'
                            node_name = elements[0]
                            bad_measures = elements[1].split('+')
                            if len(elements) > 2:
                                type = elements[2]
                        else:
                            node_type = 'measure_one'
                            node_name = elements[0]
                            measure_of = elements[1]
                            if len(elements) > 2:
                                type = elements[2]  ## può non essere specificato

                ##clean and invariant exception


                if node_name == 'hci': node_name = 'hcl'
                node_name= node_name.replace('[', '').replace("]", "")
                node_name = node_name.replace('node:', '').replace(" ", "")
                node_name = node_name.replace('/', '-')
                end = end.replace('endof:', '').replace(" ", "")
                type = type.replace('type:', '').replace('/', '-').replace(" ", "")
                measure_of = measure_of.replace('measureof:', '').replace('/', '-').replace(" ", "")
                ends = []
                measures = []
                for end in bad_ends:
                    if 'mea' in end: raise Exception("measu in end")
                    ends.append(end.replace('endof:', '').replace(" ", ""))
                for measure in bad_measures:
                    if '.' in measure: raise Exception(". AND NOT ,")
                    measures.append(measure.replace('measureof:', '').replace('/', '-').replace(" ", ""))
                if node_name == '': raise Exception("Node name not present")
                if 'end' in node_name:  raise Exception("End in node name")
                if 'mea' in end: raise Exception("measu in end")
                if 'mea' in node_name: raise Exception("measu in node_name")
                if '.' in measure_of: raise Exception(". AND NOT ,")
                if '!' in node_name:  raise Exception("bad_node name")


                ##clean and invariant exception

                ## TODO mettere che se -----node_name + ';Class;Valve;'----- già presente in in nodes_csv solleva eccezione
                ## Veloce si fa in una riga di codice
                ## TODO mettere che se -----end + 'Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;'----- già presente in in edges_csv solleva eccezione
                ## Veloce si fa in una riga di codice


                if node_type == 'normal_one_end':
                    if 'xv' in node_name:  ## sotto nodo del nodo normale: valvola
                        nodes_csv.append(node_name + ';Class;Valve;')  # Nodo normale valvola
                        edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                        flusso_num = flusso_num + 1

                    else:
                        nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
                        edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                        flusso_num = flusso_num + 1

                if node_type == 'start':
                    nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola

                if node_type == 'exit':
                    nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
                    edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                    edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                    edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)

                    flusso_num = flusso_num + 1
                if node_type == 'normal_multiple_end':
                    if 'xv' in node_name:  ## sotto nodo del nodo normale: valvola
                        nodes_csv.append(node_name + ';Class;Valve;')  # Nodo normale valvola
                    else:
                        nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
                    for end in ends:
                        edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                        flusso_num = flusso_num + 1
                if node_type == 'measure_one':
                    if '%' in type:
                        type = 'percentage'
                    measures_csv.append(node_name + ';Class;Sensor;')
                    measures_csv.append(type + ';Class;UnitOfMeasure;')

                    measures_csv.append(node_name + ';ObjectProperty;canMeasureIn;' + type)
                    measures_csv.append(node_name + ';ObjectProperty;isContainedIn;' + measure_of)

                if node_type == 'measure_multiple':
                    if '%' in type:
                        type = 'percentage'
                    measures_csv.append(node_name + ';Class;Sensor;')
                    measures_csv.append(type + ';Class;UnitOfMeasure;')

                    measures_csv.append(node_name + ';ObjectProperty;canMeasureIn;' + type)
                    for measure in measures:
                        measures_csv.append(node_name + ';ObjectProperty;isContainedIn;' + measure)



            except Exception as ex:
                print(ex)
                cv2.putText(img, str(node), find_center_of_one_contour(contours[int(node) - 1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2)

                print('in node ' + str(node))

                cv2.putText(original_img, str(node), find_center_of_one_contour(contours[int(node) - 1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2)

                continue

    subtraction = [x for x in added_ends if x not in added_nodes]

    ##TODO CHECK IF THERE ARE TWO NODES WITH SAME NAME

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(label=img_path)
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.title(label=img_path)
    plt.show()

    return nodes_csv, edges_csv, measures_csv


def remove_contour_under_area(contours, area):
    good_contours = []
    for contour in contours:
        val = cv2.contourArea(contour)
        if val > area:
            good_contours.append(contour)
    return good_contours


def get_inner_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == '__main__':

    path_name_original_images = os.listdir(ROOT_DIR + '/squares_image/')

    if os.path.exists(ROOT_DIR + '/all_csv/' + 'all-starts.txt'):
        os.remove(ROOT_DIR + '/all_csv/' + 'all-starts.txt')
    else:
        print("The file does not exist")

    if os.path.exists(ROOT_DIR + '/all_csv/' + 'all-exits.txt'):
        os.remove(ROOT_DIR + '/all_csv/' + 'all-exits.txt')
    else:
        print("The file does not exist")



    ## google, se non sono identici sostituire con txt....

    for path_name in path_name_original_images:

        img_square = cv2.imread(ROOT_DIR + '/squares_image/' + path_name)
        contours_of_squares1 = get_inner_contour(img_square)
        contours_of_squares1 = sorted(contours_of_squares1, key=cv2.contourArea,
                                      reverse=True)
        contours_of_squares = remove_contour_under_area(contours_of_squares1, 400)
        imgname_node_words_dic_google = {path_name: detect_text(ROOT_DIR + '/squares_image/' + path_name)}

        dic_img_nodes_semantic_google = {path_name: give_semantic_to_nodes(
            imgname_node_words_dic_google,
            contours_of_squares,
            img_square)}

        dic_img_nodes_semantic_opencv = {path_name: my_text_detection_tessereact(img_square, contours_of_squares)}

        dic_img_nodes_semantic = merge_two_dictionaries(dic_img_nodes_semantic_google, dic_img_nodes_semantic_opencv)

        original_img = cv2.imread(ROOT_DIR + '/drive_gray/' + path_name)

        heigth_square_img, width_square_img, channels = img_square.shape

        img_resized_original = cv2.resize(original_img, (width_square_img, heigth_square_img))

        nodes_csv, edges_csv, measures_csv = generate_lines(dic_img_nodes_semantic, contours_of_squares, img_square,
                                                            img_resized_original)


        specify_physical_object(nodes_csv)

        all_starts = []
        starts = get_all_starts(nodes_csv)
        all_starts.extend(starts)

        all_exits = []
        exits = get_all_exits(nodes_csv)
        all_exits.extend(exits)


        f2 = open(ROOT_DIR + '/all_csv/' + 'all-starts.txt', 'a')
        f2.write('\n')
        f2.write(path_name)
        f2.write('---')
        f2.write('\n')
        f2.write('---')
        f2.write('\n')
        for element in all_starts:
            f2.write(element)
            f2.write('\n')
            f2.flush()
        f2.close()


        f3 = open(ROOT_DIR + '/all_csv/' + 'all-exits.txt', 'a')
        f3.write(path_name)
        f3.write('\n')
        f3.write('---')
        f3.write('\n')
        f3.write('---')
        f3.write('\n')

        for element in all_exits:
            f3.write(element)
            f3.write('\n')
            f3.flush()
        f3.close()

        try:

            f = open(ROOT_DIR + '/all_csv/' + path_name.replace('.png', '.txt'), 'w')

            f1 = open('/Users/nicc/all_csv/' + path_name.replace('.png', '.csv'), 'w')

            for element in nodes_csv:
                f.write(element)
                f.write('\n')
                f1.write(element)
                f1.write('\n')
                f1.flush()
                f.flush()
            for element in edges_csv:
                f.write(element)
                f.write('\n')
                f1.write(element)
                f1.write('\n')
                f1.flush()
                f.flush()
            for element in measures_csv:
                f.write(element)
                f.write('\n')
                f1.write(element)
                f1.write('\n')
                f1.flush()
                f.flush()

            f.close()
            f1.close()
        except Exception as ex:
            print(ex)

        print('fine')
        ## da fare qui dic_img_nodes_semantic co tesseract e mergiare con dic_img_nodes_semantic






