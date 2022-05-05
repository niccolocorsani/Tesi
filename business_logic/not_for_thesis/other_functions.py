import pandas as pd
from termcolor import colored
from not_for_thesis.vision import *
from not_for_thesis.draw_things import DrawThings
from business_logic.main import detect_text
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ROOT_DIR + "/resources/google-auth.json"


# Both paths are realative to the root paths
# Folder path is the path of the images, excel_path is the path of the excel
def compute_corrispondence_from_image_google_with_less_lines_of_code(folder_path, excel_path):
    draw_things = DrawThings()
    xl = pd.ExcelFile(ROOT_DIR + excel_path)
    sheet_names = xl.sheet_names
    path_name = os.listdir(ROOT_DIR + folder_path)
    for sheet_name in sheet_names:
        df = pd.read_excel(ROOT_DIR + '/input_files/altair.xlsx', sheet_name)
        list_of_value = df.values.tolist()
        for path_names in path_name:
            print(colored(path_names, 'red'))
            text_vertex_dic = detect_text(folder_path + "/" + path_names)
            all_words_of_image = text_vertex_dic.keys()
            # print(colored(all_words_of_image, 'green'))
            image = Image.open(ROOT_DIR + folder_path + path_names)
            for word in all_words_of_image:
                if len(word) < 3: continue
                clean_word = word.replace("\n", "").replace("|", "").replace(",", "").replace(".", "").replace("$", "S")
                image = draw_things.draw_rectangle(text_vertex_dic.get(word), image)
                for word_list in list_of_value:
                    if str(word_list[0]) in str(clean_word) and str(word_list[0]) != 'nan':
                        print('found correspondence of ' + str(word_list[0]) + ' with ' + str(
                            clean_word) + ' in file: ' + path_names + ' and excel sheet: ' + sheet_name)
                    if str(word_list[3]) in str(clean_word) and str(word_list[3]) != 'nan':
                        print('found correspondence of ' + str(word_list[3]) + ' with ' + str(
                            clean_word) + ' in file: ' + path_names + ' and excel sheet: ' + sheet_name + ': Nome Strumento')
                    if str(word_list[2]) in str(clean_word) and str(word_list[2]) != 'nan':
                        print('found correspondence of ' + str(word_list[2]) + ' with ' + str(
                            clean_word) + ' in file: ' + path_names + ' and excel sheet: ' + sheet_name + ': Nome Strumento')
                    if str(word_list[4]) in str(clean_word) and str(word_list[4]) != 'nan':
                        print('found correspondence of ' + str(word_list[4]) + ' with ' + str(
                            clean_word) + ' in file: ' + path_names + ' and excel sheet: ' + sheet_name + ': Nome Strumento')

        path_new_image = ROOT_DIR + '/modified_images/' + path_names.replace('.jpg', '') + '_modified.jpg'
        try:
            image.save(path_new_image)

        except:
            rgb_im = image.convert('RGB')
            rgb_im.save(path_new_image)





if __name__ == '__main__':

    path_name_original_images = os.listdir(ROOT_DIR + '/drive/')




