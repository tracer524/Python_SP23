import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.cm as cm
from MainApp3 import Ui_MainWindow
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
        r=10
        Xmax = r; Xmin = -r
        N = 1000
        X = np.linspace(Xmin,Xmax,N)
        A = 2*math.pi*a/(lambda0*f)
        d1 = d*A
        r1 = (X-d1)
        r2 = (X+d1)
        I1 = (2*jv(1,r1)/(r1))**2
        I2 = (2*jv(1,r2)/(r2))**2
        I3 = I1 + I2
        mpl=self.widget.canvas
        mpl.ax.clear()
        mpl.figure.suptitle('Fraunhofer Diffraction of circular aperture',fontsize=25, fontweight='bold')
        mpl.ax.plot(X,I1,label='light intensity 1')
        mpl.ax.plot(X,I2,label='light intensity 2')
        mpl.ax.plot(X,I3,label='SUM light intensity')
        mpl.ax.set_title(r"$\lambda = %.3e \ m, \ a = %.2e \ m, \ d= %.2e \ m, \ f = %.1f \ m$"% (lambda0,a,d,f),fontsize=15)
        mpl.ax.set_xlabel('Ar')
        mpl.ax.set_ylabel('I/I0')
        mpl.ax.legend()
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
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MyApplication = MainApp()
    MyApplication.show() 
    sys.exit(app.exec_())