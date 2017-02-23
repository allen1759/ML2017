import sys
import numpy as np
from PIL import Image

im1 = np.array(Image.open(sys.argv[1]))
im2 = np.array(Image.open(sys.argv[2]))

if np.array_equal(im1, im2):
    print 'Correct'
else:
    print 'Wrong'
