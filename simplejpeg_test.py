import glob
import os.path as pt
import io

import numpy as np
from PIL import Image
import simplejpeg
import time



def mean_absolute_difference(a, b):
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()


def main():

    np.random.seed(9)
    num_images = 100
    im = np.random.randint(0, 255, (num_images, 679, 657), dtype=np.uint8)
    # for subsampling, code in (('422', 1), ('420', 2), ('440', 1), ('411', 2)):
    for i in range(100):
        # encode with simplejpeg, decode with Pillow
        prev = (time.time())
        
        # write a timing function to time the encoding and decoding
        
        encoded = simplejpeg.encode_jpeg(im[i, :,:], 85, colorsubsampling='422')
        post = (time.time())
        print(prev - post)
        # bio = io.BytesIO(encoded)
        # decoded = np.array(Image.open(bio))
        # assert 0 < mean_absolute_difference(im, decoded) < 50, subsampling
        # encode with Pillow, decode with simplejpeg
        # bio = io.BytesIO()
        # pil_im = Image.fromarray(im, 'RGB')
        # # pil_im.save(bio, format='JPEG', quality=85, subsampling=code)
        # decoded = simplejpeg.decode_jpeg(bio.getbuffer())
        # assert 0 < mean_absolute_difference(im, decoded) < 50, subsampling

if "__main__" == __name__:
    main()
