import numpy as np
import cv2
import os
import copy
import math
from matplotlib import pyplot as plt

class ProcChain:

    def __init__(self):
        self.chain = []
        self.debug = False
        self.debugData = []
        self.ctx = {}

    def context(self):
        return self.ctx

    def append(self, step):
        self.chain.append(step)

    def enableDebug(self, en = True):
        self.debug = True

    def process(self, img):
        step = 1
        tmp = copy.deepcopy(img)
        if self.debug:
            self.debugData.append(("START", tmp))

        for s in self.chain:
            tmp = s.process(tmp, self.ctx )
            if self.debug:
                stepDbg = s.debugData(step) #
                if stepDbg:
                    for e in stepDbg:
                        self.debugData.append((e[0], s.ctx[e[1]]))
                else:
                    caption = "%s:%d" %(s.sname(),step)
                    self.debugData.append( (caption,tmp) )

                n = int(math.sqrt( len(self.debugData) ) + 1)
                for i in xrange(len(self.debugData)):
                    plt.subplot(n, n, i + 1), plt.imshow(self.debugData[i][1], 'gray')
                    plt.title(self.debugData[i][0])
                    plt.xticks([]), plt.yticks([])

            step += 1

        if self.debug and plt:
            plt.show()

        return tmp



class ImgProcStep:
    pass

    def process(self, img, ctx):
        pass

    def sname(self):
        return ''

    def debugData(self, num):
        return None

class ImgProcToGray(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = img
        if len(img.shape)>=3 and img.shape[2] >=3:
            M = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return M

    def sname(self):
        return 'ToGray'


class ImgProcBlur(ImgProcStep):

    def __init__(self, t=0, **kwargs):
        self.t=t
        self.s = 5
        if kwargs:
            self.s = kwargs['size']

    def process(self, img, ctx):
        M=None
        if self.t == 0:
            M = cv2.blur(img, (self.s, self.s))
        elif self.t == 1:
            M = cv2.bilateralFilter(img, 9, 75, 75)
        elif self.t == 2:
            M = cv2.medianBlur(img, 5)
        elif self.t == 3:
            M = cv2.GaussianBlur(img, (5, 5), 0)

        return M

    def sname(self):
        return 'Blur'


class ImgProcSplit(ImgProcStep):
    def __init__(self, r,g,b):
        self.r = r
        self.g = g
        self.b = b

    def process(self, img, ctx):
        self.ctx[self.b], self.ctx[self.g] ,self.ctx[self.r] = cv2.split(img)
        return img

    def sname(self):
        return 'Split'

    def debugData(self, n):
        return [("%s R:%d" % (self.sname(),n),self.r),("%s G:%d" % (self.sname(),n),self.g),("%s B:%d" % (self.sname(),n),self.b)]

class ImgProcRoi(ImgProcStep):
    def __init__(self, x1,x2,y1,y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def process(self, img, ctx):
        return img[self.y1:self.y2,self.x1:self.x2]

    def sname(self):
        return 'ROI'

class ImgProcChRed(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.split(img)[0]

    def sname(self):
        return 'Red'

class ImgProcChGreen(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.split(img)[1]

    def sname(self):
        return 'Green'

class ImgProcChBlue(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.split(img)[2]

    def sname(self):
        return 'Blue'


class ImgProcUse(ImgProcStep):

    def __init__(self, ctx, name):
        self.name=name
        self.ctx = ctx

    def process(self, img, ctx):
        return self.ctx[self.name]

    def sname(self):
        return 'Use'


class ImgProcStore(ImgProcStep):

    def __init__(self, name):
        self.name = name

    def process(self, img, ctx):
        ctx[self.name] = img
        return img

    def sname(self):
        return 'Store'

class ImgProcTh(ImgProcStep):

    def __init__(self, th):
        self.th = th

    def process(self, img, ctx):
        return cv2.threshold(img, self.th,255,cv2.THRESH_BINARY)[1]


class ImgProcThInv(ImgProcStep):

    def __init__(self, th):
        self.th = th

    def process(self, img, ctx):
        return cv2.threshold(img, self.th,255,cv2.THRESH_BINARY_INV)[1]

class ImgProcThOtsu(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.threshold(img, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    def sname(self):
        return 'ThOtsu'


class ImgProcThAdpt(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    def sname(self):
        return 'ThAdpt'
class ImgProcResize(ImgProcStep):

    def __init__(self,w,h):
        self.w = w
        self.h = h

    def process(self, img, ctx):
        M = cv2.resize(img, (self.w, self.h))
        return M

    def sname(self):
        return 'Resize'

class ImgProcNorm(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return M

    def sname(self):
        return 'Norm'

class ImgProcBin(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ret, M = cv2.threshold(M, .5, 1, cv2.THRESH_BINARY)
        return M

    def sname(self):
        return 'Bin'

class ImgProcPyrDn(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = cv2.pyrDown( img )
        return M

    def sname(self):
        return 'PyrDn'

class ImgProcBilFilter(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.bilateralFilter(img, 9, 75, 75)

    def sname(self):
        return 'BilFilter'
