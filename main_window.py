# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(816, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.new_btn = QtWidgets.QPushButton(self.centralwidget)
        self.new_btn.setGeometry(QtCore.QRect(70, 210, 81, 21))
        self.new_btn.setObjectName("new_btn")
        self.clause_edt = QtWidgets.QTextEdit(self.centralwidget)
        self.clause_edt.setGeometry(QtCore.QRect(40, 83, 161, 91))
        self.clause_edt.setObjectName("clause_edt")
        self.cause_edt = QtWidgets.QTextEdit(self.centralwidget)
        self.cause_edt.setGeometry(QtCore.QRect(260, 83, 161, 91))
        self.cause_edt.setObjectName("cause_edt")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 30, 101, 16))
        self.label.setObjectName("label")
        self.ecc_btn = QtWidgets.QPushButton(self.centralwidget)
        self.ecc_btn.setGeometry(QtCore.QRect(250, 210, 56, 17))
        self.ecc_btn.setObjectName("ecc_btn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 816, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.new_btn.clicked.connect(MainWindow.New)
        self.ecc_btn.clicked.connect(MainWindow.Ecc)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.new_btn.setText(_translate("MainWindow", "生成示例"))
        self.label.setText(_translate("MainWindow", "中文情感原因提取系统"))
        self.ecc_btn.setText(_translate("MainWindow", "提取原因"))
