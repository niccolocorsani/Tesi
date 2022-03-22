import logging
import os
from PIL import ImageFont, ImageDraw
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DrawThings:

    def __init__(self):
        pass

    def draw_rectangle(self,xy , image):
        draw = ImageDraw.Draw(image)
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

        if xy is not None:
            try:
                x1y1 = xy[0].replace('(', '').replace(')', '')
                x1 = int(x1y1.split(',')[0])
                y1 = int(x1y1.split(',')[1])
                x2y2 = xy[2].replace('(', '').replace(')', '')
                x2 = int(x2y2.split(',')[0])
                y2 = int(x2y2.split(',')[1])
                draw.rectangle([(x1, y1), (x2, y2)], outline="red")

            except:
                logging.exception("Exception: ")



        return image


    def draw_text(self,path, text):
        image = Image.open(path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(ROOT_DIR + '/resources/arial.ttf', 40)
        draw.text((0, 0), text, (255, 255, 255), font=font)


if __name__ == '__main__':
    pass