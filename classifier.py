# -*- coding: utf-8 -*-
"""
    Classifier module
    -----------------

    -Model management
    -Scaling
    -Persistence
    -Training
    -Testing

    :copyright: (c) 2018 by Aleksej Kusnezov
    :license: BSD, see LICENSE for more details.
"""

import os, sys
import cv2
import numpy as np
import uuid
from sklearn.externals import joblib
import sklearn.svm as svm


class Classifier:

    def __init__(self):
        self.basedir = ''
        self.clf = None
        self.pc = None
        self.fe = None
        self.resultmap = {}

    def addItem(self, img, classname):
        clpath = os.path.join( self.basedir, classname)
        try:
            os.makedirs(clpath)
        except Exception:
            pass  #unless class dir exists -> create

        unique_filename = str(uuid.uuid4())+".png"
        fpath = os.path.join(clpath, unique_filename)

        cv2.imwrite(fpath, img)


    def setBaseDir(self, dirname):

        if os.path.isdir(dirname):         #check if already exists yes? -> set as basedir
            self.basedir = dirname
            return True

        try:                               #create ok? -> set as basedir
            os.mkdir(dirname)
        except Exception:
            return False

        self.basedir = dirname
        return True

    def roi(self, fname='roi.txt'):
        roifile = os.path.join(self.basedir, fname)
        try:
            data = open(roifile, 'r').read().split(" ")
            return (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
        except Exception, ex:
            return (0, 0, 640, 480)

    def set_roi(self, x0, y0, x1, y1, fname='roi.txt'):
        roifile = os.path.join(self.basedir, fname)
        try:
            open(roifile, 'w').write("%d %d %d %d" % (x0, y0, x1, y1) )
        except Exception, ex:
            pass

    def learn(self, pc, fe):
        self.pc = pc
        self.fe = fe
        modelfile = os.path.join(self.basedir, 'er.model')
        if False and os.path.exists(modelfile):
            self.clf = joblib.load( os.path.join(self.basedir, 'er.model'))
        else:
            try:
                tmp = os.listdir(os.path.join(self.basedir, 'train'))
                clss = [ c for c in tmp if os.path.isdir(os.path.join(self.basedir,'train', c)) ]

                ft=[]
                cv=[]
                cid=0
                for c in clss:
                    cid += 1
                    self.resultmap[cid] = c
                    cd = os.path.join(self.basedir,'train', c)
                    its = os.listdir(cd)
                    for i in its:
                        try:
                            img = cv2.imread( os.path.join(cd,i) )
                            out = pc.process( img )
                            ft.append(fe.process(pc.context()))
                            cv.append(cid)
                            #cv2.imshow("img_pc", out)
                            #cv2.waitKey(1)
                        except Exception, ex:
                            print ('clsf: %s' % str(ex))
                            pass

                self.clf = svm.SVC(gamma=0.001, C=100.)
                self.clf.fit(ft, cv)

                joblib.dump(self.clf, os.path.join(self.basedir, 'er.model'))
            except Exception, ex:
                pass

        pass

    def verify(self):
        res ={}
        err = {}
        try:
            tmp = os.listdir(os.path.join(self.basedir, 'test'))
            clss = [c for c in tmp if os.path.isdir(os.path.join(self.basedir, 'test', c))]
            for c in clss:
                if c not in res:
                    res[c] = 0
                if c not in err:
                    err[c] = {}

                cd = os.path.join(self.basedir, 'test', c)
                its = os.listdir(cd)
                for i in its:
                    try:
                        img = cv2.imread(os.path.join(cd, i))
                        out = self.test(img)

                        #print "%s: %s - %s; %s" %( str(c == out),c,out,i )

                        if c == out:
                            res[c] += 1
                        else:
                            if out not in err[c]:
                                err[c][out]=0
                            err[c][out] += 1



                    except Exception, ex:
                        print ('clsf: %s' % str(ex))
                        pass

        except Exception, ex:
            pass

        return res, err


    def test(self, img):

        self.pc.process(img)
        fv = self.fe.process( self.pc.context())
        ft = np.asarray(fv).reshape(1, -1)
        res = self.clf.predict(ft)

        clname = self.resultmap[res[0]]

        return clname




if __name__ == '__main__':
    import procchain
    import featex
    import datagen

    t = Classifier()

    ret = t.setBaseDir("c:/tmp/anamit");

    pc = procchain.ProcChain()
    fe = featex.FeatEx()

    #fe.append(featex.Moments())
    fe.append(featex.Pixels())

    pc.append(procchain.ImgProcToGray())
    pc.append(procchain.ImgProcNorm())
    pc.append(procchain.ImgProcResize(32, 32))
    pc.append(procchain.ImgProcStore("result"))
    #pc.debug=True

    t.learn(pc,fe)
    res, err = t.verify()

    for c in res:
        print "OK %s %d/%d  " %( c, res[c], str(err[c]))


    pass

