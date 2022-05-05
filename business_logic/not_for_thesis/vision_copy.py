import os
import time
import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
import imutils

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_holes2(image_path):
    import cv2
    import numpy as np

    def getLandmarks(corners):
        holes = []
        for i in range(0, len(corners)):
            for j in range(i + 1, len(corners)):
                x1, y1 = corners[i].ravel()
                x2, y2 = corners[j].ravel()
                if 12 <= abs(x1 - x2) <= 40 and 30 <= abs(y1 - y2) <= 12:
                    holes.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        return holes

    # lodes in img

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)

    holes = getLandmarks(corners)

    for corner in holes:
        cv2.circle(img, (corner), 7, (255, 255, 0), -1)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def get_end_points2(img):
    # https://stackoverflow.com/questions/35847990/detect-holes-ends-and-beginnings-of-a-line-using-opencv
    def getLandmarks(corners):
        holes = []
        for i in range(0, len(corners)):
            x1, y1 = corners[i].ravel()
            holes.append((int(x1), int(y1)))

        return holes

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
        cv2.circle(white_image_with_circle, (corner), 4, (0, 0, 0), -1)
    plt.imshow(white_image_with_circle)
    plt.show()
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

    for contour_of_line in contours_lines:

        for contour in contours:

            intersection = get_number_of_intersection(img_copy, contour, contour_of_line)
            # print(intersection)
            if (intersection == 2):
                cv2.drawContours(new_image_withe_to_show_in_for_loop_evolving_each_iteration, [contour_of_line], 0,
                                 (0, 0, 255), 1)
                cv2.drawContours(new_image_withe_to_show_in_for_loop_evolving_each_iteration, [contour], 0, (0, 255, 0),
                                 2)
                center = find_center_of_contours([contour])[0]
                cv2.putText(new_image_withe_to_show_in_for_loop_evolving_each_iteration, str(intersection),
                            center, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

                # center_line = find_center_of_contour([contour_of_line])[0]
                # cv2.putText(new_image_withe_to_show_in_for_loop_evolving_each_iteration, str(i) + 'img',
                #            center_line, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

                end_points_contours.append(contour)

    plt.imshow(new_image_withe_to_show_in_for_loop_evolving_each_iteration)
    plt.show()

    return end_points_contours, contours_lines, new_image_withe_to_show_in_for_loop_evolving_each_iteration


def get_end_points(img):
    # https://stackoverflow.com/questions/35847990/detect-holes-ends-and-beginnings-of-a-line-using-opencv
    def getLandmarks(corners):
        holes = []
        for i in range(0, len(corners)):
            x1, y1 = corners[i].ravel()
            holes.append((int(x1), int(y1)))

        return holes

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
        cv2.circle(white_image_with_circle, (corner), 4, (0, 0, 0), -1)
    plt.imshow(white_image_with_circle)
    plt.show()
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

    for contour in contours:

        for contour_of_line in contours_lines:

            intersection = get_number_of_intersection(img_copy, contour, contour_of_line)

            # print(intersection)

            if (intersection == 2):
                cv2.drawContours(new_image_withe_to_show_in_for_loop_evolving_each_iteration, [contour_of_line], 0,
                                 (0, 0, 255), 1)
                cv2.drawContours(new_image_withe_to_show_in_for_loop_evolving_each_iteration, [contour], 0, (0, 255, 0),
                                 2)
                center = find_center_of_contours([contour])[0]
                cv2.putText(new_image_withe_to_show_in_for_loop_evolving_each_iteration, str(intersection),
                            center, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

                # center_line = find_center_of_contour([contour_of_line])[0]
                # cv2.putText(new_image_withe_to_show_in_for_loop_evolving_each_iteration, str(i) + 'img',
                #            center_line, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

                end_points_contours.append(contour)

    plt.imshow(new_image_withe_to_show_in_for_loop_evolving_each_iteration)
    plt.show()

    return end_points_contours, contours_lines, new_image_withe_to_show_in_for_loop_evolving_each_iteration


def find_center_of_one_contour(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]


def find_center_of_contours(contours):
    centers = []

    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append([cx, cy])

    return centers


def get_holes(img):
    # https://stackoverflow.com/questions/35847990/detect-holes-ends-and-beginnings-of-a-line-using-opencv
    def getLandmarks(corners):
        holes = []
        for i in range(0, len(corners)):
            for j in range(i, len(corners)):
                x1, y1 = corners[i].ravel()
                x2, y2 = corners[j].ravel()
                if abs(x1 - x2) <= 30 and abs(y1 - y2) <= 30:
                    holes.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        return holes

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)
    holes = getLandmarks(corners)
    print(holes)
    for corner in holes:
        cv2.circle(img, (corner), 7, (255, 255, 0), -1)
    return holes, get_contours_of_img(img), img


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

    square_contours = []

    i = 0

    for contour in contours:

        if i == 0:
            i = 1
            continue
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

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

            square_contours.append(contour)

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

    return square_contours


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


def return_img_to_size_of_img_input(img_input_path):
    img = cv2.imread(img_input_path)
    plt.imshow(img)
    plt.show()
    heigth, width, channels = img.shape

    resized = cv2.resize(img, (width, heigth))
    plt.imshow(resized)
    plt.show()

    # w = int((width * 2))
    # h = int((heigth * 2))
    # resized = cv2.resize(img, (w,h))

    return


def get_contours_of_img(img):
    #### Contorni con solo cerchi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #### Contorni con solo cerchi
    return contours


def get_contour_areas(contours):
    all_areas = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas[area] = cnt
    return all_areas


# Work even with list of tuples
def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]


if __name__ == '__main__':
    img_square = cv2.imread(ROOT_DIR + '/squares.png')

    img_of_lines = cv2.imread(ROOT_DIR + '/senza_immagine.png')

    heigth_lines, width_lines, channels = img_of_lines.shape

    resized_square = cv2.resize(img_square, (width_lines, heigth_lines))

    contours_of_squares = get_contours_of_img(resized_square)
    contours_end_points, contours_lines, img = get_end_points2(img_of_lines)
    points = find_center_of_contours(contours_end_points)

    # points, contours_lines, img = get_holes(img_of_lines)

    cleaned_square_points = []

######Capire perchè non filtra i cerchi all'itnerno dei quadratiiii
#####

##TODO Attenzione al contorno dell'immagine totale-
i = 0
sono_quadrati = np.ones([heigth_lines, width_lines, channels], dtype=np.uint8)
sono_quadrati.fill(255)  # or img[:] = 255

sorted_contours = sorted(contours_of_squares, key=cv2.contourArea, reverse=True)

for contour_of_square in sorted_contours:

    if i == 0:
        i = i + 1
        continue
    cv2.drawContours(sono_quadrati, [contour_of_square], 0,
                     (0, 0, 255), 1)

    for point in points:
        cv2.circle(sono_quadrati, (point), 7, (50, 255, 0), -1)
        distance = cv2.pointPolygonTest(contour_of_square, point, False)
        if distance > 0:
            cleaned_square_points.append(point)

cleaned_square_points = removeDuplicates(cleaned_square_points)
print('cleaned_points_dimension')
print(len(cleaned_square_points))
print('not_cleaned_points_dimension')
print(len(points))
#########


time.sleep(5)

white_img = np.ones([heigth_lines, width_lines, channels], dtype=np.uint8)
white_img.fill(255)  # or img[:] = 255

#### disegna le linee
n = 0
for contour_of_line in contours_lines:
    n = n + 1
    cv2.drawContours(white_img, [contour_of_line], 0,
                     (0, 0, 0), 1)
    center = find_center_of_one_contour(contour_of_line)
    cv2.putText(white_img, str(n),
                center, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
plt.imshow(white_img)
plt.show()
print('number of contours:')
print(n)
time.sleep(10)

#### disegna i cerchi
for point in cleaned_square_points:
    cv2.circle(white_img, (point), 7, (255, 255, 0), -1)

plt.imshow(white_img)
plt.show()

time.sleep(1)

#### disegna i quadrati
for contour_of_square in contours_of_squares:
    cv2.drawContours(white_img, [contour_of_square], 0,
                     (255, 0, 0), 1)
plt.imshow(white_img)
plt.show()

time.sleep(1)

print('fine')

# get nearest end point per ricreare il nodo
