import pandas as pd
from termcolor import colored
import xlsxwriter
from business_logic.vision import *
from business_logic.draw_things import DrawThings
import logging
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ROOT_DIR + "/resources/google-auth.json"


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


# Già posso scrivere i nomi dei nodi e le eventuali unità di misura
def write_on_excel(nome, nodo_ua, ua_data_type, nome_strumento, dv_path, funzione, unita_di_misura, volume_m3, min_max,
                   prodotto, densita_kg_m3, destinazione, sorgente):
    workbook = xlsxwriter.Workbook(ROOT_DIR + '/output_files/info_altair.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A2', 'njnbj..')
    worksheet.write('B2', 'Geeks')
    worksheet.write('C2', 'For')
    worksheet.write('D2', 'Geeks')
    workbook.close()


# The aim of this function is to compute all corrispondence between text (inside excel shetts) and images text
# The input parameters are:
# 1) the folder path, for which all the images inside are analyzed
# 2) not implemented yet, should be the path of the excel file ecc....
# This function depends on detect_text(path)
########## Cloruro Ferrico ferroso
def compute_corrispondence_from_image_google_and_save_files(image_folder_path):
    draw_things = DrawThings()

    df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx', sheet_name="NaOH KOH")
    listNaOH_KOH = df.values.tolist()
    df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx',
                       sheet_name="HCl")  # Necessario qui mettere ROOT_DIR perchè su alcuni sistemi va in bambola mettendo solo ../ che in realtà sarebbe giusto
    listHCL = df.values.tolist()
    df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx', sheet_name="cloroparaffine (CPS)")
    list_cloro_paraffine = df.values.tolist()
    df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx', sheet_name="ipoclorito di sodio")
    list_ipoclorito_di_sodio = df.values.tolist()
    df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx', sheet_name="Cloruro Ferrico-Ferroso Pot.le")
    list_cloruro_ferrico_ferroso = df.values.tolist()
    df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx', sheet_name="Cloruro Ferrico std")
    list_cloruro_ferrico_std = df.values.tolist()
    path_name = os.listdir(ROOT_DIR + "/pagine")

    for path_names in path_name:
        print(colored(path_names, 'red'))
        text_vertex_dic = detect_text(image_folder_path + "/" + path_names)
        all_words_of_image = text_vertex_dic.keys()
      #  print(colored(all_words_of_image, 'green'))
        image = Image.open(ROOT_DIR + "/pagine/" + path_names)

        # Compare ocr images text with excel text

        for word in all_words_of_image:
            if len(word) < 3: continue
            clean_word = word.replace("\n", "").replace("|", "").replace(",", "").replace(".", "").replace("$", "S")

            image = draw_things.draw_rectangle(text_vertex_dic.get(word), image)

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

        ##TODO salva solo l'ultimo rettangolo
        ## Per risolverlo cambiare la funzione, che non è che a ogni iterazione fa il draw di un solo rettangolo, ma creare una nuova
        ## funzione dove gli viene passato una lista con tutti i rettangoli e li salva tutti insieme

        path_new_image = ROOT_DIR + '/modified_images/' + path_names.replace('.jpg', '') + '_modified.jpg'
        try:

            image.save(path_new_image)

        except:
            print('Can t save file because need RGB')
            rgb_im = image.convert('RGB')
            rgb_im.save(path_new_image)


if __name__ == '__main__':
    compute_corrispondence_from_image_google_and_save_files(ROOT_DIR + '/pagine')
