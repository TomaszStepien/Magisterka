"""takes pictures from input directory, lowers their resolution
and saves in output directory
"""

import os
from PIL import Image

root_dir = "C:\\magisterka"
input_dir = "good_pics"
output_dir = "bad_pics"
scale = 10

os.chdir(root_dir)

for picture_name in os.listdir(input_dir):
    im = Image.open(input_dir + "\\" + picture_name)
    size = im.size
    im = im.resize((size[0] // scale, size[1] // scale),
                   Image.ANTIALIAS)
    im.save(output_dir + "\\" + picture_name)
    im.close()
