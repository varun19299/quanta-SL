import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, qApp
from ui_ss2_1b_cont_stream_SL import Ui_ss2_1b_cont_stream_SL
from PyQt5 import QtCore, QtGui, uic
import pyqtgraph as pq

class App_ss2_1b_cont_stream_SL(QWidget):    
    ####################################################################    
    ########################### Constructor ############################
    def __init__(self, parent=None):
        super(App_ss2_1b_cont_stream_SL,self).__init__()
        self.ui = Ui_ss2_1b_cont_stream_SL()
        self.ui.setupUi(self)

########################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = App_ss2_1b_cont_stream_SL()
    myapp.show()
    
    sys.exit(app.exec_())   
