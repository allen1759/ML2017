import sys
import numpy as np
from PIL import Image

im1 = np.array(Image.open(sys.argv[1]))
im2 = np.array(Image.open(sys.argv[2]))

for row1, row2 in zip(im1, im2):
    for e1, e2 in zip(row1, row2):
        if np.array_equal(e1, e2):
            e2.fill(0)

result = Image.fromarray(im2)
result.save('ans_two.png')
