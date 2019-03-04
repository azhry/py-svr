# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.svm import SVR
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd, numpy as np, matplotlib.pyplot as plt

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    See: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

class Ui_MainWindow(object):
    
    train = None
    test = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(10, 0, 311, 551))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.graphicsView = QtWidgets.QLabel(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(330, 0, 461, 291))
        self.graphicsView.setObjectName("graphicsView")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(330, 300, 451, 251))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.groupBox)
        self.tableWidget_2.setGeometry(QtCore.QRect(10, 10, 141, 151))
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(1)
        self.tableWidget_2.setRowCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, item)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(170, 100, 261, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        # self.dateEdit = QtWidgets.QDateEdit(self.groupBox_2)
        # self.dateEdit.setGeometry(QtCore.QRect(140, 60, 110, 22))
        # self.dateEdit.setObjectName("dateEdit")
        # self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_2)
        # self.pushButton_2.setGeometry(QtCore.QRect(170, 90, 75, 23))
        # self.pushButton_2.setObjectName("pushButton_2")
        # self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        # self.label_2.setGeometry(QtCore.QRect(20, 30, 221, 16))
        # self.label_2.setObjectName("label_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(170, 10, 261, 80))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton.setGeometry(QtCore.QRect(180, 50, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(10, 20, 231, 21))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.menuFile.addAction(self.actionOpen_File)
        self.menubar.addAction(self.menuFile.menuAction())

        self.actionOpen_File.triggered.connect(lambda: self.import_excel(MainWindow))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Data"))
        item = self.tableWidget_2.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Score"))
        item = self.tableWidget_2.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "MSE"))
        item = self.tableWidget_2.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "MAE"))
        item = self.tableWidget_2.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "RMSE"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Value"))
        # self.groupBox_2.setTitle(_translate("MainWindow", "Make Prediction"))
        # self.pushButton_2.setText(_translate("MainWindow", "Predict"))
        # self.pushButton_2.clicked.connect(self.make_prediction)
        # self.label_2.setText(_translate("MainWindow", "Choose the end date you want to predict"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Test Model"))
        self.pushButton.setText(_translate("MainWindow", "Do Test"))
        self.pushButton.clicked.connect(lambda: self.do_test(MainWindow))
        self.label.setText(_translate("MainWindow", "Evaluates SVR by calculating score and error"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))

    def import_excel(self, MainWindow):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(MainWindow, "Select Excel File", "", "Excel Files(*.xls *.xlsx)", options=options)
        if (fileName):
            self.train = pd.read_excel(fileName, sheet_name='Training Data')
            self.test = pd.read_excel(fileName, sheet_name='Testing Data')
            self.tableWidget.setRowCount(len(self.train) + len(self.test) + 1)
            for i, (time, data) in enumerate(zip(self.train['Time'], self.train['Data'])):
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 0, item)
                item = self.tableWidget.item(i, 0)
                item.setText(str(time))

                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 1, item)
                item = self.tableWidget.item(i, 1)
                item.setText(str(data))

            train_len = len(self.train)
            for i, (time, data) in enumerate(zip(self.test['Time'], self.test['Data'])):
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i + train_len, 0, item)
                item = self.tableWidget.item(i + train_len, 0)
                item.setText(str(time))

                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i + train_len, 1, item)
                item = self.tableWidget.item(i + train_len, 1)
                item.setText(str(data))

            self.data = pd.concat([self.train, self.test])
            plt.plot(self.data['Time'].values, self.data['Data'].values)
            plt.savefig('Plot.png')

            pic = QtGui.QPixmap('Plot.png')
            pic = pic.scaled(511, 341)
            self.graphicsView.setPixmap(pic)


    def do_test(self, MainWindow):
        if self.train is not None and self.test is not None:
            steps = 6

            df = series_to_supervised(list(self.train['Data'].values), steps)
            x = df.iloc[:, [a for a in range(steps - 2)]].values
            y = df.iloc[:, [steps - 1]].values.ravel()

            df = series_to_supervised(list(self.test['Data'].values), steps)
            xtest = df.iloc[:, [a for a in range(steps - 2)]].values
            ytest = df.iloc[:, [steps - 1]].values.ravel()

            regressor = SVR(kernel='linear', epsilon=1.0)
            regressor.fit(x, y)
            ypred = regressor.predict(xtest)
            score = regressor.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(0, 0, item)
            item = self.tableWidget_2.item(0, 0)
            item.setText(str(score))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(1, 0, item)
            item = self.tableWidget_2.item(1, 0)
            item.setText(str(mse))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(2, 0, item)
            item = self.tableWidget_2.item(2, 0)
            item.setText(str(mae))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(3, 0, item)
            item = self.tableWidget_2.item(3, 0)
            item.setText(str(rmse))

            f, ax = plt.subplots()
            actual = ax.plot(self.data['Time'].values, self.data['Data'].values, color='blue', label='Actual')
            ttest = self.test[:-steps]
            predicted = ax.plot(ttest['Time'], ypred, color='red', label='Predicted')
            ax.legend()
            plt.savefig('Plot.png')

            pic = QtGui.QPixmap('Plot.png')
            pic = pic.scaled(511, 341)
            self.graphicsView.setPixmap(pic)
        else:
            MainWindow.msg = QtWidgets.QMessageBox()
            MainWindow.msg.setIcon(QtWidgets.QMessageBox.Warning)
            MainWindow.msg.setWindowTitle("Warning")
            MainWindow.msg.setText("Anda harus membuka file terlebih dahulu")
            MainWindow.msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            MainWindow.msg.show()

    def make_prediction(self):
        print('Make prediction')

if __name__ == "__main__":
   import sys
   app = QtWidgets.QApplication(sys.argv)
   MainWindow = QtWidgets.QMainWindow()
   ui = Ui_MainWindow()
   ui.setupUi(MainWindow)
   MainWindow.show()
   sys.exit(app.exec_())