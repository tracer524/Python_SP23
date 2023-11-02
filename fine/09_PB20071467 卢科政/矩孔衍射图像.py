import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.cm as cm
from MainApp import Ui_MainWindow
from PyQt5.QtWidgets import QApplication , QMainWindow
from PyQt5.QtCore import pyqtSlot

class MainApp(QMainWindow , Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.fig1()
    def fig1(self):
        lambda_0=self.horizontalSliderlambda.value()*10**(-9)
        b=self.horizontalSliderb.value()*10**(-5)
        h=self.horizontalSliderh.value()*10**(-5)
        a= 0.1
        f=self.horizontalSliderf.value()
        k = 2*math.pi/lambda_0
        XMax = a/2; YMax = a/2
        XMin = -a/2; YMin = -a/2
        N = 1000
        X = np.linspace(XMin,XMax,N)
        Y = np.linspace(YMin,YMax,N)
        B = (k*b*X)/(2*f)
        H = (k*h*Y)/(2*f)
        Bx,Hy = np.meshgrid(B,H)
        I = (np.sinc(Bx/math.pi)*np.sinc(Hy/math.pi))**2
        mpl=self.widget.canvas
        mpl.ax.clear()
        mpl.figure.suptitle('Fraunhofer Diffraction of rectangular aperture',fontsize=25, fontweight='bold')
        mpl.ax.imshow(I, cmap ="gray",origin='lower', interpolation ='bilinear',vmin=I.min(),vmax=0.01*I.max())
        mpl.ax.set_xticks([0, N/2, N]); mpl.ax.set_xticklabels([XMin,0,XMax])
        mpl.ax.set_yticks([0, N/2, N]); mpl.ax.set_yticklabels([YMin,0,YMax])
        mpl.ax.set_title(r"$\lambda = %.3e \ m, \ b = %.2e \ m, \ h= %.2e \ m, \ f_2 = %.1f \ m$"% (lambda_0,b,h,f),fontsize=15)
        mpl.draw()
    @pyqtSlot("double")
    def on_doubleSpinBoxlambda_valueChanged(self, value):
        self.horizontalSliderlambda.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxb_valueChanged(self, value):
        self.horizontalSliderb.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxh_valueChanged(self, value):
        self.horizontalSliderh.setValue(value)
    @pyqtSlot("double")
    def on_doubleSpinBoxf_valueChanged(self, value):
        self.horizontalSliderf.setValue(value)
    @pyqtSlot("int")
    def on_horizontalSliderlambda_valueChanged(self, value):
        self.doubleSpinBoxlambda.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSliderb_valueChanged(self, value):
        self.doubleSpinBoxb.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSliderh_valueChanged(self, value):
        self.doubleSpinBoxh.setValue(value)
        self.fig1()
    @pyqtSlot("int")
    def on_horizontalSliderf_valueChanged(self, value):
        self.doubleSpinBoxf.setValue(value)
        self.fig1()
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MyApplication = MainApp()
    MyApplication.show() 
    sys.exit(app.exec_())
