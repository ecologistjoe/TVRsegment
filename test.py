
from PIL import Image
import numpy as np
from time import time
import segmentation


A = np.asarray(Image.open('landscape.png'))
A = A.astype('double') / 255;


#t0 = time()
#B = SB.TVR(A, 20, 0.01, 'aniso')
#print "Python", time()-t0


t0 = time()
C = segmentation.TVR(A, 4, 0.001, 'aniso', solver="jac")

#C = segmentation.segment(A, 10, 10, conn=4)


print "Cython", time()-t0


im = Image.fromarray((255*C).astype('uint8'));
im.save('landscape_TVR.png')


