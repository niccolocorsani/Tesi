import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageFilter

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


def get_pixel_color(x, y):
    pass


def to_gray_scale(path_image_input, path_image_output):
    image = Image.open(path_image_input)
    image = image.convert("L")
    image.save(path_image_output)


if __name__ == '__main__':

    # coordinates = save_image_to_edged_black_on_white_background_and_return_vertex_of_white(
    #     ROOT_DIR + '/bianco_nero_con_giallo_e_blu.jpg'
    #     , ROOT_DIR + '/output_files/black_and_white_image.png')

    image = cv2.imread(ROOT_DIR + '/bianco_nero_con_giallo_e_blu.jpg')

    width, height, channel = image.shape




    # 1, 49, 248
 ##TODO finire funzione in grado di trovare l'rgb associato a ciascun pixel
    for i in range(0, width):
        for j in range(0, height):
            for x in range(0, 3):
                sys.stdout.write(str(image[i][j][x])+ ' ')

            #    if image[i][j] == [248, 49]: print('o')

    print('fine')
