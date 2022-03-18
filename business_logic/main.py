import pandas as pd
import os
from termcolor import colored
from PIL import ImageFont, ImageDraw
import xlsxwriter
from business_logic.vision import *
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../resources/google-auth.json"


def draw_rettangle(xy, path, x_1, y_1, x_2, y_2):
    # creating a image object
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    text = 'Some textttttt'
    font = ImageFont.truetype('../resources/arial.ttf', 40)

    i = 0
    '''
    coordinates = [(x1, y1), (x2, y2)]
        (x1, y1)
            *--------------
            |             |
            |             |
            |             |
            |             |
            |             |
            |             |
            --------------*
                          (x2, y2)
    '''

    # ['(1,55)', '(76,53)', '(77,84)', '(2,86)'], il secondo è x2, y2 il quarto è x1,y1

    try:
        x1y1 = xy[0].replace('(', '').replace(')', '')
        x1 = int(x1y1.split(',')[0])
        y1 = int(x1y1.split(',')[1])
        x2y2 = xy[2].replace('(', '').replace(')', '')
        x2 = int(x2y2.split(',')[0])
        y2 = int(x2y2.split(',')[1])

        draw.rectangle([(x1, y1), (x2, y2)], outline="red")






    except:
        print("eccezione")
        i = i + 1

    try:
        draw.rectangle([(x_1, y_1), (x_2, y_2)], outline="red")
    except:
        i = i + 1


    return image


def detect_text(path):
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
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])
        text_vertex_dic[text.description] = vertices

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return text_vertex_dic


# The aim of this function is to compute all corrispondence between text (inside excel shetts) and images text
# The input parameters are:
# 1) the folder path, for which all the images inside are analyzed
# 2) not implemented yet, should be the path of the excel file ecc....
# This function depends on detect_text(path)


########## Cloruro Ferrico ferroso

# Già posso scrivere i nomi dei nodi e le eventuali unità di misura
def write_on_excel(nome, nodo_ua, ua_data_type, nome_strumento, dv_path, funzione, unita_di_misura, volume_m3, min_max,
                   prodotto, densita_kg_m3, destinazione, sorgente):
    workbook = xlsxwriter.Workbook('../output_files/info_altair.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A2', 'njnbj..')
    worksheet.write('B2', 'Geeks')
    worksheet.write('C2', 'For')
    worksheet.write('D2', 'Geeks')
    workbook.close()


def compute_corrispondence_from_image_google(folder_path):

    df = pd.read_excel('../input_files/altair.xlsx'.read(), sheet_name="NaOH KOH")

    listNaOH_KOH = df.values.tolist()
    df = pd.read_excel('../input_files/altair.xlsx'.read(), sheet_name="HCl")
    listHCL = df.values.tolist()
    df = pd.read_excel('../input_files/altair.xlsx'.read(), sheet_name="cloroparaffine (CPS)")
    list_cloro_paraffine = df.values.tolist()
    df = pd.read_excel('../input_files/altair.xlsx'.read(), sheet_name="ipoclorito di sodio")
    list_ipoclorito_di_sodio = df.values.tolist()
    df = pd.read_excel('../input_files/altair.xlsx'.read(), sheet_name="Cloruro Ferrico-Ferroso Pot.le")
    list_cloruro_ferrico_ferroso = df.values.tolist()
    df = pd.read_excel('../input_files/altair.xlsx'.read(), sheet_name="Cloruro Ferrico std")
    list_cloruro_ferrico_std = df.values.tolist()
    path_name = os.listdir("../pagine")

    textImage = ""

    # Read images
    for path_names in path_name:
        print(colored(path_names, 'red'))
        text_vertex_dic = detect_text(folder_path + "/" + path_names)
        all_words_of_image = text_vertex_dic.keys()
        print(colored(all_words_of_image, 'green'))

        # Compare ocr images text with excel text

        for word in all_words_of_image:
            if (len(word) < 4): continue
            clean_word = word.replace("\n", "").replace("|", "").replace(",", "").replace(".", "").replace("$", "S")

            image = draw_rettangle(text_vertex_dic.get(word), "../pagine/" + path_names, None, None, None, None)

            for word_list in listNaOH_KOH:
                if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                    print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet NaOH_KOH')

                if str(word_list[3]) in str(clean_word) and str(word_list[3]) != 'nan':
                    print('found correspondence of ' + str(word_list[3]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet NaOH_KOH: Nome Strumento')
            ########## NaOH_KOH

            ########## HCl
            for word_list in listHCL:
                if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                    print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet hcl')

                if str(word_list[4]) in str(clean_word) and str(word_list[4]) != 'nan':
                    print('found correspondence of ' + str(word_list[4]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet HCL: Nome Strumento')

            ########## HCl

            ########## CloroParaffine

            for word_list in list_cloro_paraffine:
                if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                    print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet cloroparaffine')

                if str(word_list[2]) in str(clean_word) and str(word_list[2]) != 'nan':
                    print('found correspondence of ' + str(word_list[2]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet cloroparaffine: Nome Strumento')

            ########## CloroParaffine

            ########## Ipoclorito di Sodio

            for word_list in list_ipoclorito_di_sodio:
                if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                    print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet ipoclorito_di_sodio')

            ########## Ipoclorito di Sodio

            ########## Cloruro Ferrico std

            for word_list in list_cloruro_ferrico_std:
                if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                    print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet list_cloruro_ferrico_std')

                if str(word_list[2]) in str(clean_word) and str(word_list[2]) != 'nan':
                    print('found correspondence of ' + str(word_list[2]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet list_cloruro_ferrico_std: Nome Strumento')

            ########## Cloruro Ferrico std

            ########## Cloruro Ferrico ferroso

            for word_list in list_cloruro_ferrico_ferroso:

                if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                    print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet list_cloruro_ferrico_ferroso')

                if str(word_list[2]) in str(clean_word) and str(word_list[2]) != 'nan':
                    print('found correspondence of ' + str(word_list[2]) + ' with ' + str(
                        clean_word) + ' in file: ' + path_names + ' and excel sheet list_cloruro_ferrico_ferroso: Nome Strumento')





        # image.save(path_name.replace('.jpg','') + '_modified.jpg')

        ########## Cloruro Ferrico ferroso



if __name__ == '__main__':
    # write_on_excel(None, None, None, None, None, None, None, None, None, None, None, None, None)
    compute_corrispondence_from_image_google('../pagine')
