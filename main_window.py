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
        MainWindow.setEnabled(True)
        MainWindow.resize(900, 600)
        MainWindow.setMaximumSize(QtCore.QSize(900, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(6, 7, 768, 965))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setMinimumSize(QtCore.QSize(300, 487))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(30)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(False)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.graphicsView = QtWidgets.QGraphicsView(self.layoutWidget)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setMinimumSize(QtCore.QSize(460, 487))
        self.graphicsView.setStyleSheet("border-image: url(:/newPrefix/logo.png);")
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout_3.addWidget(self.graphicsView)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.clause_edt = QtWidgets.QTextEdit(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.clause_edt.setFont(font)
        self.clause_edt.setObjectName("clause_edt")
        self.verticalLayout_2.addWidget(self.clause_edt)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.emotion_edt = QtWidgets.QTextEdit(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.emotion_edt.setFont(font)
        self.emotion_edt.setObjectName("emotion_edt")
        self.verticalLayout.addWidget(self.emotion_edt)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.cause_edt = QtWidgets.QTextEdit(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cause_edt.setFont(font)
        self.cause_edt.setObjectName("cause_edt")
        self.verticalLayout.addWidget(self.cause_edt)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.new_btn = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.new_btn.setFont(font)
        self.new_btn.setObjectName("new_btn")
        self.horizontalLayout_2.addWidget(self.new_btn)
        self.clear_btn = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.clear_btn.setFont(font)
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout_2.addWidget(self.clear_btn)
        self.ecc_btn = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.ecc_btn.setFont(font)
        self.ecc_btn.setObjectName("ecc_btn")
        self.horizontalLayout_2.addWidget(self.ecc_btn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.ecc_btn.clicked.connect(MainWindow.Ecc)
        self.clear_btn.clicked.connect(MainWindow.Clear)
        self.new_btn.clicked.connect(MainWindow.New)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "中文情绪原因发现系统"))
        self.label_2.setText(_translate("MainWindow", "原句"))
        self.clause_edt.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600; color:#4a4a4a;\">请输入，粘贴（中文）文本</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-weight:600; color:#4a4a4a;\"><br /></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "情感子句"))
        self.label_3.setText(_translate("MainWindow", "原因子句"))
        self.new_btn.setText(_translate("MainWindow", "生成示例"))
        self.clear_btn.setText(_translate("MainWindow", "重新输入"))
        self.ecc_btn.setText(_translate("MainWindow", "原因提取"))
import logo_rc
