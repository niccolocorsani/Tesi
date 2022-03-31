import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Not usefull
def find_lines(path):
    ### MAKING TEMPLATE WITHOUT HOUGH

    # Read the image and make a copy then transform it to gray colorspace,
    # threshold the image and search for contours.
    img = cv2.imread(path)
    res = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Iterate through contours and draw a slightly bigger white rectangle
    # over the contours that are not big enough (the text) on the copy of the image.
    for i in contours:
        cnt = cv2.contourArea(i)
        if cnt < 500:
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(res, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 255, 255), -1)

    # Display the result. Note that the image is allready the template!
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional count the rows and columns of the table
    count = res.copy()
    gray = cv2.cvtColor(count, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    check = []
    for i in contours:
        cnt = cv2.contourArea(i)
        if 10000 > cnt > 10:
            cv2.drawContours(count, [i], 0, (255, 255, 0), 2)
            M = cv2.moments(i)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            check.append([cx, cy])

    check.sort(key=lambda xy: xy[1])
    columns = 1

    for i in range(0, len(check) - 1):
        if check[i + 1][1] + 5 >= check[i][1] >= check[i + 1][1] - 5:
            columns += 1
        else:
            break
    print(columns)

    check.sort(key=lambda tup: tup[0])
    rows = 1
    for i in range(0, len(check) - 1):
        if check[i + 1][0] + 5 >= check[i][0] >= check[i + 1][0] - 5:
            rows += 1
        else:
            break
    print('Columns: ', columns)
    print('Roiws : ', rows)

    cv2.imshow('res', count)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ### LINES WITH HOUGHLINES()

    # Convert the resulting image from previous step (no text) to gray colorspace.
    res2 = img.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # You can either use threshold or Canny edge for HoughLines().
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform HoughLines tranform.
    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(res2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result.
    cv2.imshow('res', res)
    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ### LINES WITH HOUGHLINESP()

    # Convert the resulting image from first step (no text) to gray colorspace.
    res3 = img.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection and dilate the edges for better result.
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Perform HoughLinesP tranform.
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 50, minLineLength, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(res3, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result.
    cv2.imwrite('h_res1.png', res)
    cv2.imwrite('h_res2.png', res2)
    cv2.imwrite('h_res3.png', res3)

    cv2.imshow('res', res)
    cv2.imshow('res2', res2)
    cv2.imshow('res3', res3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


##TODO meglio cambiare strada, forse uno shape recognition è megliooooo da li se trova angoli vuol dire che non è un end-point
## TODO o altrimenti guardare tipo di fare un rettangolo li vicino al punto e vedere se vi è un eventuale intersezione con questo rettangolo
def get_lines_end_points(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)
    holes = []

    for i in range(0, len(corners)):
        x1, y1 = corners[i].ravel()
        x = int((x1))
        y = int((y1))

        # TODO va ridotto tutto a 1 pixel per farlo funzionare
        for i in range(100):
            val = img[x][y]
            val_dex = img[x + i][y]
            val_up_dex = img[x + i][y + i]
            val_up = img[x][y + i]
            val_up_left = img[x - i][y + i]
            val_left = img[x - i][y]
            val_down_left = img[x - i][y - i]
            val_down = img[x][y - i]
            val_down_right = img[x + i][y - i]
            print(i)

            if (any(ele != 255 for ele in val) or any(ele != 255 for ele in val_dex) or any(
                    ele != 255 for ele in val_up_dex) or any(ele != 255 for ele in val_up) or any(
                ele != 255 for ele in val_up_left) or any(ele != 255 for ele in val_left) or any(
                ele != 255 for ele in val_down_left) or any(ele != 255 for ele in val_down) or any(
                ele != 255 for ele in val_down_right)):
                print('oo')

        break
        val = img[x][y]
        val_dex = img[x + 1][y]
        val_up_dex = img[x + 1][y + 1]
        val_up = img[x][y + 1]
        val_up_left = img[x - 1][y + 1]
        val_left = img[x - 1][y]
        val_down_left = img[x - 1][y - 1]
        val_down = img[x][y - 1]
        val_down_right = img[x + 1][y - 1]

        # TODO mettere vincoli in maniera tale che valuti solo gli edge e non gli angoli.....
        # Di base l'idea è che va ridotto a 1 pixel la larghezza delle linee ecc..  Poi si nota che gli end point hanno un solo figlio, gli angoli invece ne hanno 2
        circle_img = cv2.circle(img, (x, y), 7, (255, 255, 0), -1)
        matplotlib.pyplot.imshow(circle_img)
        matplotlib.pyplot.show()

        print(img[x][y])
        holes.append((x, y))

    print(holes)

    for corner in holes:
        cv2.circle(img, (corner), 7, (255, 255, 0), -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
def get_line_testinggg(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)
    holes = []

    x1, y1 = corners[0].ravel()
    x = int((x1))
    y = int((y1))

    circle_img = cv2.circle(img, (x, y), 7, (255, 255, 0), -1)
    matplotlib.pyplot.imshow(circle_img)
    matplotlib.pyplot.show()

    for corner in holes:
        cv2.circle(img, (x, y), 7, (255, 255, 0), -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
def correct_image_to_really_all_white_and_black(image_path):
    image = cv2.imread(image_path)
    # image_inverted = (255 - image)
    imagem = replace_rgb_values(image)

    width, height, channels = image.shape
    for x in range(width):
        for y in range(height):
            r, g, b = image[x, y]
            if r != 255 and r != 0:
                print(r)
                print(image[x, y])

    cv2.imwrite(ROOT_DIR + '/inverted_image.png', imagem)
##TODO meglio cambiare strada, forse uno shape recognition è megliooooo da li se trova angoli vuol dire che non è un end-point


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
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

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


if __name__ == '__main__':
    # plt.imshow([[[255, 0, 0],
    #              [0, 0, 0]]])
    #
    # plt.show()

    # correct_image_to_really_all_white_and_black(ROOT_DIR + '/disegno.png')
    # get_line_testinggg(ROOT_DIR + '/inverted_image.png')
    # get_holes(ROOT_DIR + '/oo.png')
    # find_position_by_rgb_color_pil(ROOT_DIR + '/result.png', 10, 50, 243)
    detect_shape(ROOT_DIR + '/disegno.png')
