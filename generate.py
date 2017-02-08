#!/usr/bin/python

import Image, ImageDraw, ImageFont
import numpy as np

FILE_IMAGES = 'gen_images.npy'
FILE_LABELS = 'gen_labels.npy'

def generate_chars(size):
    size = (size,)*2
    font = '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf'
    font = ImageFont.truetype(font, size=size[0]/2)

    gen_images = []
    gen_labels = []
    for NUMBER in range(10):
        im = Image.new('L', size)
        draw = ImageDraw.Draw(im)
        text = chr(ord('0') + NUMBER)
        draw.text((size[0] / 3, size[1] / 4), text, fill=255, font=font)  # Puts it roughly in the middle

        for MOD in range(1 << 8):
            quadData = [
                0, 0,             # NW
                0, size[1],       # SW
                size[0], size[1], # SE
                size[0], 0,       # NE
            ]

            # Modify the corners
            for i in range(8):
                if MOD & (1 << i):
                    quadData[i] += (size[i % 2] / 4) * (-1 if quadData[i] else 1)

            im_mod = im.transform(size, Image.QUAD, quadData)#, resample=Image.LINEAR)
            # im_mod = im_mod.rotate(17.5)#, resample=Image.LINEAR)

            #if MOD == 250:
            #   im_mod.show()

            nparray = np.asarray(im_mod, dtype='float32')
            nparray = nparray.reshape((1, size[0], size[1], 1))
            gen_images.append(nparray)
            label = np.zeros((1, 10), dtype='float32')
            label[0, NUMBER] = 1
            gen_labels.append(label)

    gen_images = np.concatenate(gen_images, 0)
    gen_labels = np.concatenate(gen_labels, 0)

    np.save(FILE_IMAGES, gen_images)
    np.save(FILE_LABELS, gen_labels)

generate_chars(28)
