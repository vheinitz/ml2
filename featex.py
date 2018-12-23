import numpy as np
import cv2



class FeatEx:

    def __init__(self):
        self.chain = []

    def append(self, step):
        self.chain.append(step)

    def process(self, ctx ):
        res = []
        for s in self.chain:
            res += s.process(ctx )
        return res



class Moments():

    def __init__(self):
        pass

    def process(self, ctx, itemname='result'):
        img = ctx[itemname]
        M = cv2.moments(img, binaryImage=True)
        D = []
        for k, v in M.iteritems():
            D.append(v)
        return D  #np.asarray(D)

class Pixels():

    def __init__(self):
        pass

    def process(self, ctx, itemname='result'):
        img = ctx[itemname]
        M = np.reshape(img,(1,32*32))
        D = M.tolist()
        return D[0]  #np.asarray(D)

"""


def moments( img, binImg=False ):
    M = cv2.moments(img, binImg)
    D = []
    for k,v in M.iteritems():
        D.append(v)
    return np.asarray(D)

def huMoments( img, binImg=False ):
    M = cv2.HuMoments(img, binImg)
    D = []
    for k,v in M.iteritems():
        D.append(v)
    return np.asarray(D)

def hog( img ):
    cell_size = (4, 4)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 1  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    h = hog.compute(img)
    return h


def corners( img ):
    fimg = np.float32(img)
    dst = cv2.cornerHarris(fimg, 2, 3, 0.04)
    #cv2.imshow("corners", dst)
    #cv2.waitKey(0)
    return fimg

"""