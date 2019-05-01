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
from datetime import datetime
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     """
#     See: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

#     Frame a time series as a supervised learning dataset.
#     Arguments:
#         data: Sequence of observations as a list or NumPy array.
#         n_in: Number of lag observations as input (X).
#         n_out: Number of observations as output (y).
#         dropnan: Boolean whether or not to drop rows with NaN values.
#     Returns:
#         Pandas DataFrame of series framed for supervised learning.
#     """
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
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
        MainWindow.resize(1100, 600)
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
        self.graphicsView.setGeometry(QtCore.QRect(330, 0, 761, 291))
        self.graphicsView.setObjectName("graphicsView")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(330, 300, 751, 251))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.groupBox)
        self.tableWidget_2.setGeometry(QtCore.QRect(10, 10, 441, 171))
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(6)
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
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(5, item)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(470, 100, 261, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.dateEdit = QtWidgets.QDateEdit(QtCore.QDate(2019, 2, 1), self.groupBox_2)
        self.dateEdit.setGeometry(QtCore.QRect(140, 60, 110, 22))
        self.dateEdit.setObjectName("dateEdit")
        self.dateEdit.setDisplayFormat('MMMM yy')
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 90, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 221, 16))
        self.label_2.setObjectName("label_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(470, 10, 261, 80))
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
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "SDM"))
        item = self.tableWidget_2.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Score"))
        item = self.tableWidget_2.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "MSE"))
        item = self.tableWidget_2.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "MAE"))
        item = self.tableWidget_2.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "RMSE"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "SVR (linear)"))
        item = self.tableWidget_2.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "SVR (rbf)"))
        item = self.tableWidget_2.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Lasso"))
        item = self.tableWidget_2.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Elastic Net"))
        item = self.tableWidget_2.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Ridge"))
        item = self.tableWidget_2.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "MLP"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Make Prediction"))
        self.pushButton_2.setText(_translate("MainWindow", "Predict"))
        self.pushButton_2.clicked.connect(self.make_prediction)
        self.label_2.setText(_translate("MainWindow", "Choose the end date you want to predict"))
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
            self.tableWidget.setRowCount(len(self.train) + len(self.test))
            for i, (time, data, sdm) in enumerate(zip(self.train['Time'], self.train['Data'], self.train['Data2'])):
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 0, item)
                item = self.tableWidget.item(i, 0)
                item.setText(str(time))

                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 1, item)
                item = self.tableWidget.item(i, 1)
                item.setText(str(data))

                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 2, item)
                item = self.tableWidget.item(i, 2)
                item.setText(str(sdm))

            train_len = len(self.train)
            for i, (time, data, sdm) in enumerate(zip(self.test['Time'], self.test['Data'], self.test['Data2'])):
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i + train_len, 0, item)
                item = self.tableWidget.item(i + train_len, 0)
                item.setText(str(time))

                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i + train_len, 1, item)
                item = self.tableWidget.item(i + train_len, 1)
                item.setText(str(data))

                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i + train_len, 2, item)
                item = self.tableWidget.item(i + train_len, 2)
                item.setText(str(sdm))

            self.data = pd.concat([self.train, self.test])
            plt.plot(self.data['Time'].values, self.data['Data'].values)
            plt.savefig('Plot.png')

            pic = QtGui.QPixmap('Plot.png')
            pic = pic.scaled(811, 341)
            self.graphicsView.setPixmap(pic)


    def do_test(self, MainWindow):
        if self.train is not None and self.test is not None:
            steps = 6
            train = DataFrame()
            train['Data2'] = list(self.train['Data2'].values)
            train['Data'] = list(self.train['Data'].values)
            df = series_to_supervised(train.values, steps)
            x = df.iloc[:, [a for a in range(steps * 2 - 2)]].values
            y = df.iloc[:, [steps * 2 - 1]].values.ravel()

            test = DataFrame()
            test['Data2'] = list(self.test['Data2'].values)
            test['Data'] = list(self.test['Data'].values)
            df = series_to_supervised(test.values, steps)
            dft = series_to_supervised(list(self.test['Time'].values), steps)
            xtest = df.iloc[:, [a for a in range(steps * 2 - 2)]].values
            ytest = df.iloc[:, [steps * 2 - 1]].values.ravel()

            regressor = SVR(kernel='linear', epsilon=1.0)
            regressor.fit(x, y)
            ypred = regressor.predict(xtest)
            score = regressor.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            print("SVR Kernel Linear")
            print(f"Score: {score}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}\n")

            f, ax = plt.subplots()
            actual = ax.plot(self.data['Time'].values, self.data['Data'].values, color='blue', label='Actual')
            ttest = dft['var1(t)'].values
            predicted = ax.plot(ttest, ypred, color='red', label='Predicted')
            ax.legend()
            plt.savefig('Plot.png')

            pic = QtGui.QPixmap('Plot.png')
            pic = pic.scaled(811, 341)
            self.graphicsView.setPixmap(pic)

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

            regressor = SVR(kernel='rbf', epsilon=1.0)
            regressor.fit(x, y)
            ypred = regressor.predict(xtest)
            score = regressor.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            print("SVR Kernel RBF")
            print(f"Score: {score}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}\n")

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(0, 1, item)
            item = self.tableWidget_2.item(0, 1)
            item.setText(str(score))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(1, 1, item)
            item = self.tableWidget_2.item(1, 1)
            item.setText(str(mse))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(2, 1, item)
            item = self.tableWidget_2.item(2, 1)
            item.setText(str(mae))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(3, 1, item)
            item = self.tableWidget_2.item(3, 1)
            item.setText(str(rmse))

            from sklearn import linear_model
            reg = linear_model.Lasso(alpha=0.1)
            reg.fit(x, y)
            ypred = reg.predict(xtest)
            score = reg.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            print("Linear Model - Lasso")
            print(f"Score: {score}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}\n")

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(0, 2, item)
            item = self.tableWidget_2.item(0, 2)
            item.setText(str(score))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(1, 2, item)
            item = self.tableWidget_2.item(1, 2)
            item.setText(str(mse))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(2, 2, item)
            item = self.tableWidget_2.item(2, 2)
            item.setText(str(mae))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(3, 2, item)
            item = self.tableWidget_2.item(3, 2)
            item.setText(str(rmse))

            reg = linear_model.ElasticNet(alpha=0.1)
            reg.fit(x, y)
            ypred = reg.predict(xtest)
            score = reg.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            print("Linear Model - Elastic Net")
            print(f"Score: {score}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}\n")

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(0, 3, item)
            item = self.tableWidget_2.item(0, 3)
            item.setText(str(score))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(1, 3, item)
            item = self.tableWidget_2.item(1, 3)
            item.setText(str(mse))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(2, 3, item)
            item = self.tableWidget_2.item(2, 3)
            item.setText(str(mae))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(3, 3, item)
            item = self.tableWidget_2.item(3, 3)
            item.setText(str(rmse))

            reg = linear_model.Ridge(alpha=0.1)
            reg.fit(x, y)
            ypred = reg.predict(xtest)
            score = reg.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            print("Linear Model - Ridge")
            print(f"Score: {score}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}\n")

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(0, 4, item)
            item = self.tableWidget_2.item(0, 4)
            item.setText(str(score))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(1, 4, item)
            item = self.tableWidget_2.item(1, 4)
            item.setText(str(mse))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(2, 4, item)
            item = self.tableWidget_2.item(2, 4)
            item.setText(str(mae))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(3, 4, item)
            item = self.tableWidget_2.item(3, 4)
            item.setText(str(rmse))

            from sklearn.neural_network import MLPRegressor
            # from sklearn.preprocessing import StandardScaler

            # scaler = StandardScaler()
            # scaler.fit(x)
            # x_scaled = scaler.transform(x)
            # xtest_scaled = scaler.transform(xtest)

            hidden = 5
            reg = MLPRegressor(hidden_layer_sizes=(5, ), activation='logistic', solver='lbfgs', alpha=0.0001,random_state=0)
            reg.fit(x, y)
            ypred = reg.predict(xtest)
            score = reg.score(xtest, ytest)
            mse = mean_squared_error(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            rmse = sqrt(mse)
            print("Neural Network - Backpropagation")
            print(reg)
            print(f"Input: {x.shape[1]}")
            print(f"Hidden: {hidden}")
            print(f"Output: {reg.n_outputs_}")
            print(f"Score: {score}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}\n")

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(0, 5, item)
            item = self.tableWidget_2.item(0, 5)
            item.setText(str(score))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(1, 5, item)
            item = self.tableWidget_2.item(1, 5)
            item.setText(str(mse))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(2, 5, item)
            item = self.tableWidget_2.item(2, 5)
            item.setText(str(mae))

            item = QtWidgets.QTableWidgetItem()
            self.tableWidget_2.setItem(3, 5, item)
            item = self.tableWidget_2.item(3, 5)
            item.setText(str(rmse))

            
        else:
            MainWindow.msg = QtWidgets.QMessageBox()
            MainWindow.msg.setIcon(QtWidgets.QMessageBox.Warning)
            MainWindow.msg.setWindowTitle("Warning")
            MainWindow.msg.setText("Anda harus membuka file terlebih dahulu")
            MainWindow.msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            MainWindow.msg.show()

    def make_prediction(self):
        if self.train is not None and self.test is not None:
            steps = 6
            train = DataFrame()
            train['Data2'] = list(self.train['Data2'].values)
            train['Data'] = list(self.train['Data'].values)
            df = series_to_supervised(train.values, steps)
            x = df.iloc[:, [a for a in range(steps * 2 - 2)]].values
            y = df.iloc[:, [steps * 2 - 1]].values.ravel()

            start_year = 2019
            start_month = 1
            start_date = QtCore.QDate(start_year, start_month, 1);
            end_date = self.dateEdit.date()
            diff_year = end_date.year() - start_date.year()
            end_month = end_date.month() + 1
            if diff_year == 0:
                diff_month = end_month - start_month                
            else:
                diff_month = 12 - start_month + ((12 * diff_year) - (12 - end_month))


            # diff = start_date.daysTo(end_date)
            sdm_data = np.array([])
            forecast_time = np.array([])
            forecast_data = np.empty((0, steps * 2 - 2), int)
            old_forecast = x
            x_len = len(old_forecast)
            # diff_month = (diff // 30) + 1

            total_records = len(self.train) + len(self.test)

            if diff_month > 0:
                for i in range(diff_month):
                    sdm_data = np.append(sdm_data, old_forecast[i % x_len][-2])
                    forecast_data = np.append(forecast_data, np.array([old_forecast[i % x_len]]), axis = 0)
                    forecast_time = np.append(forecast_time, f'{start_year}-{start_month}')
                    if start_month == 12:
                        start_year = start_year + 1
                        start_month = 1
                    else:
                        start_month = start_month + 1

                regressor = SVR(kernel='linear', epsilon=1.0)
                regressor.fit(x, y)
                ypred = regressor.predict(forecast_data)

                f, ax = plt.subplots()
                actual = ax.plot(self.data['Time'].values, self.data['Data'].values, color='blue', label='Actual')
                predicted = ax.plot(forecast_time, ypred, color='red', label='Forecast')
                ax.legend()
                plt.savefig('Plot.png')

                pic = QtGui.QPixmap('Plot.png')
                pic = pic.scaled(511, 341)
                self.graphicsView.setPixmap(pic)

                self.tableWidget.setRowCount(len(self.train) + len(self.test) + len(ypred))
                for i, (time, predicted, sdm) in enumerate(zip(forecast_time, ypred, sdm_data)):
                    item = QtWidgets.QTableWidgetItem()
                    self.tableWidget.setItem(i + total_records, 0, item)
                    item = self.tableWidget.item(i + total_records, 0)
                    item.setText(str(time))

                    item = QtWidgets.QTableWidgetItem()
                    self.tableWidget.setItem(i + total_records, 1, item)
                    item = self.tableWidget.item(i + total_records, 1)
                    item.setText(str(int(predicted)))

                    item = QtWidgets.QTableWidgetItem()
                    self.tableWidget.setItem(i + total_records, 2, item)
                    item = self.tableWidget.item(i + total_records, 2)
                    item.setText(str(int(sdm)))  

if __name__ == "__main__":
   import sys
   app = QtWidgets.QApplication(sys.argv)
   MainWindow = QtWidgets.QMainWindow()
   ui = Ui_MainWindow()
   ui.setupUi(MainWindow)
   MainWindow.show()
   sys.exit(app.exec_())