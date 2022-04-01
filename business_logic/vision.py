import os
import time
import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_lines2(path):
    import cv2

    # Load image, convert to grayscale, Otsu's threshold
    image = cv2.imread(path)
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (36, 255, 12), 2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (36, 255, 12), 2)

    cv2.imshow('result', result)
    cv2.waitKey()


def save_image_to_edged_black_on_white_background_and_return_vertex_of_white(path_image_input, path_image_output):
    img = cv2.imread(path_image_input,
                     0)  # Con il flag settato a 0 mi restituisce una matrice con collonne = width e righe = heigth
    edges = cv2.Canny(img, 100, 255)  # --- image containing edges ---
    inverted_image = (255 - edges)  # --- inverte image ---
    cv2.imwrite(path_image_output, inverted_image)
    print('image saved')
    indices = np.where(edges != [0])

    # ritorna 2 array: il primo contiene
    # gli indici delle x dove è verificata la condizione,
    # il secondo contiene gli indici delle y dove è verificata la condizione
    # b = np.where(a < 4)
    # print(b)
    # print("Elements which are <4")
    # print(a[b])

    coordinates = zip(indices[0], indices[1])
    # # X e Y sono liste con tutti i valori di x e tutti i valori di y
    # X, Y = map(list, zip(*coordinates))
    # plt.scatter(X, Y)
    # plt.show()
    return tuple(coordinates)


def to_gray_scale(path_image_input, path_image_output):
    image = Image.open(path_image_input)
    image = image.convert("L")
    image.save(path_image_output)


def find_position_by_rgb_color_cv2(image_path, red, green, blue):
    image = cv2.imread(image_path)
    x, y, z = image.shape
    for i in range(x):
        for j in range(y):
            if image[i, j, 0] == blue & image[i, j, 1] == green & image[i, j, 2] == red:
                print("Found color at ", i, j)

    print('fine')


def find_position_by_rgb_color_pil(image_path, red, green, blue):
    color = (red, green, blue)
    im = Image.open(image_path)
    rgb_im = im.convert('RGB')
    for x in range(rgb_im.width):
        for y in range(rgb_im.height):
            r, g, b = rgb_im.getpixel((x, y))
            print(r, g, b)
            if (r, g, b) == color:
                print(f"Found {color} at {x},{y}!")

    print('fine')


def analyze_image_and_show_with_for_loop(image_path):
    values = []
    im = Image.open(image_path)
    rgb_im = im.convert('RGB')
    for x in range(rgb_im.width):
        values.append([])
        for y in range(rgb_im.height):
            r, g, b = rgb_im.getpixel((x, y))
            values[x].append([r, g, b])

    matplotlib.pyplot.imshow(values)
    matplotlib.pyplot.show()

    print('fine')


def get_end_points(image_path):
    # https://stackoverflow.com/questions/35847990/detect-holes-ends-and-beginnings-of-a-line-using-opencv
    def getLandmarks(corners):
        holes = []
        for i in range(0, len(corners)):
            x1, y1 = corners[i].ravel()
            holes.append((int(x1), int(y1)))

        return holes
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)
    holes = getLandmarks(corners)
    print(holes)
    ### Immagine bianca dove mettere cerchi ##
    height, width, channels = img.shape
    white_img = np.zeros([height, width, channels], dtype=np.uint8)
    white_img.fill(255)  # or img[:] = 255
    cv2.imwrite(ROOT_DIR + '/white_image.png', white_img)
    white_image_with_circle = cv2.imread(ROOT_DIR + '/white_image.png')
    ### Immagine bianca dove mettere cerchi ##
    ### Metto i cerchi ##
    for corner in holes:
        cv2.circle(white_image_with_circle, (corner), 20, (0, 0, 0), -1)

    plt.imshow(white_image_with_circle)
    plt.show()
    time.sleep(5)
    ### Metto i cerchi ##
    #### Contorni con solo cerchi
    gray = cv2.cvtColor(white_image_with_circle, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #### Contorni con solo cerchi
    #### Contorni con solo linea
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours_lines, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #### Contorni con solo linea
    #### Nuova immagine dove saranno posizionati i cerchi a ogni iterazione per visualizzarli bene (anche se non serve)
    new_img = np.zeros([height, width, channels], dtype=np.uint8)
    new_img.fill(255)  # or img[:] = 255
    cv2.imwrite(ROOT_DIR + '/new.png', new_img)
    new_image_withe_to_show_in_for_loop_evolving_each_iteration = cv2.imread(ROOT_DIR + '/new.png')
    #### Nuova immagine dove saranno posizionati i cerchi a ogni iterazione  per visualizzarli bene (anche se non serve)

    img_copy = new_image_withe_to_show_in_for_loop_evolving_each_iteration.copy();
    end_points_contours = []

    i = 0
    positions = []

    for contour in contours:
        intersection = get_number_of_intersection(img_copy, contour, contours_lines[1])
        cv2.drawContours(new_image_withe_to_show_in_for_loop_evolving_each_iteration, [contour], 0, (0, 0, 255), 2)
        cv2.drawContours(new_image_withe_to_show_in_for_loop_evolving_each_iteration, [contours_lines[1]], 0,
                         (0, 0, 255), 2)
        plt.imshow(new_image_withe_to_show_in_for_loop_evolving_each_iteration)
        plt.show()
        if (intersection == 4):
            end_points_contours.append(contour)
            positions.append(holes[i])

        i = i + 1

    return end_points_contours, new_image_withe_to_show_in_for_loop_evolving_each_iteration


def find_center_of_contour(contours):

    centers = []

    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append([cx, cy])

    return centers


def get_holes(image_path):
    # https://stackoverflow.com/questions/35847990/detect-holes-ends-and-beginnings-of-a-line-using-opencv
    def getLandmarks(corners):
        holes = []
        for i in range(0, len(corners)):
            for j in range(i + 1, len(corners)):
                x1, y1 = corners[i].ravel()
                x2, y2 = corners[j].ravel()
                if abs(x1 - x2) <= 30 and abs(y1 - y2) <= 30:
                    holes.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        return holes

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)
    holes = getLandmarks(corners)
    print(holes)
    for corner in holes:
        cv2.circle(img, (corner), 7, (255, 255, 0), -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def rgb_to_hex(r, g, b):
    return ('#{:X}{:X}{:X}').format(r, g, b)


def draw_text(path, text):
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(ROOT_DIR + '/resources/arial.ttf', 40)
    draw.text((0, 0), text, (255, 255, 255), font=font)


def detect_shape(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

        if len(approx) == 1:
            cv2.putText(img, 'angle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 4:
            cv2.putText(img, 'Quadrilateral', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(img, 'Pentagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(img, 'Hexagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            print(i)
            print('detected circle')
            cv2.putText(img, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # displaying the image after drawing contours
    cv2.imshow('shapes', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def replace_rgb_values(image):
    width, height, channels = image.shape
    for x in range(width):
        for y in range(height):
            r, g, b = image[x, y]
            if 255 > r > 120:
                image[x, y] = [0, 0, 0]

            if 119 > r > 1:
                image[x, y] = [255, 255, 255]
    print('fine')

    return image


def get_number_of_intersection(original_image, contour1, contour2):
    # https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect
    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]
    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])
    # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)  # colore nero primo contorno
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)  # colore nero secondo contorno
    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)
    # Check if there was a '1' in the intersection

    count = np.count_nonzero(intersection)
    return count


if __name__ == '__main__':
    # detect_shape(ROOT_DIR + '/ttttt.png')

    contours, img = get_end_points(ROOT_DIR + '/oooo-1.png')

    height, width, channels = img.shape

    new_img = np.zeros([height, width, channels], dtype=np.uint8)
    new_img.fill(255)


    positions = find_center_of_contour(contours)

    print(len(contours))
    print(len(positions))

    check_positions_img = new_img.copy()

    for position in positions:
        cv2.circle(check_positions_img, (position), 20, (0, 0, 0), -1)
        plt.imshow(check_positions_img)
        plt.title('fina plot')
        time.sleep(1)
        plt.show()

    check_contours_img = new_img.copy()
    for contour in contours:
        cv2.drawContours(check_contours_img, [contour], 0, (0, 0, 255), 2)
        plt.imshow(check_contours_img)
        plt.title('fina plot')
        time.sleep(1)
        plt.show()
