import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.cm as cm
from MainApp2 import Ui_MainWindow
from PyQt5.QtWidgets import QApplication , QMainWindow
from PyQt5.QtCore import pyqtSlot
from scipy.special import jv

class MainApp(QMainWindow , Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.fig1()
    def fig1(self):
        lambda0=self.horizontalSliderlambda.value()*10**(-9)
        a=self.horizontalSlidera.value()*10**(-6)
        d=self.horizontalSliderd.value()*10**(-4)
        f=self.horizontalSliderf.value()
        r=self.horizontalSliderr.value()*10**(-3)
        Xmax = r; Xmin = -r
        Ymax = r; Ymin = -r
        N = 1000
        X = np.linspace(Xmin,Xmax,N)
        Y = np.linspace(Ymin,Ymax,N)
        Xx,Yy = np.meshgrid(X,Y)
        A = 2*math.pi*a/(lambda0*f)
        r1 = ((Xx-d)**2+Yy**2)**0.5
        r2 = ((Xx+d)**2+Yy**2)**0.5
        I = (jv(1,A*r1)/(A*r1))**2+(jv(1,A*r2)/(A*r2))**2

        mpl=self.widget.canvas
        mpl.ax.clear()
        mpl.figure.suptitle('Fraunhofer Diffraction of circular aperture',fontsize=25, fontweight='bold')
        mpl.ax.imshow(I, cmap ="gray",origin='lower', interpolation ='bilinear',vmin=0,vmax=0.01)
        mpl.ax.set_xticks(np.linspace(0,N,5)); mpl.ax.set_xticklabels(np.linspace(Xmin,Xmax,5))
        mpl.ax.set_yticks(np.linspace(0,N,5)); mpl.ax.set_yticklabels(np.linspace(Xmin,Xmax,5))
        mpl.ax.set_title(r"$\lambda = %.3e \ m, \ a = %.2e \ m, \ d= %.2e \ m, \ f = %.1f \ m,\ r= %.2e \ m$"% (lambda0,a,d,f,r),fontsize=15)
        mpl.draw()
    @pyqtSlot("double")
    def on_doubleSpinBoxlambda_valueChanged(self, value):
        self.horizontalSliderlambda.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxa_valueChanged(self, value):
        self.horizontalSlidera.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxd_valueChanged(self, value):
        self.horizontalSliderd.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxf_valueChanged(self, value):
        self.horizontalSliderf.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxr_valueChanged(self, value):
        self.horizontalSliderr.setValue(value)
    @pyqtSlot("int")
    def on_horizontalSliderlambda_valueChanged(self, value):
        self.doubleSpinBoxlambda.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSlidera_valueChanged(self, value):
        self.doubleSpinBoxa.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSliderd_valueChanged(self, value):
        self.doubleSpinBoxd.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSliderf_valueChanged(self, value):
        self.doubleSpinBoxf.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSliderr_valueChanged(self, value):
        self.doubleSpinBoxr.setValue(value)
        self.fig1()
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MyApplication = MainApp()
    MyApplication.show() 
    sys.exit(app.exec_())