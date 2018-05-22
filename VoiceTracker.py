#!/usr/bin/python3
#encoding: utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QIcon

# 回调函数
def chooseModel():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","")
    print(name)
    startWindow.hide()
    mainWindow.show()

def createModel():
    startWindow.hide()
    mainWindow.show()

def chooseTrainingSample():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","")
    print(name)

def chooseTestingSample():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","")
    print(name)

# 创建应用
app=QApplication(sys.argv)

# 创建起始窗体
startWindow=QWidget()
startWindow.setWindowTitle("VoiceTracker")
startWindow.resize(790,440)
startWindow.move(300,300)
palette=QPalette()
palette.setBrush(startWindow.backgroundRole(), QBrush(QPixmap('start.jpg')))
startWindow.setPalette(palette)

# 创建主窗体
mainWindow=QWidget()
mainWindow.setWindowTitle("VoiceTracker")
mainWindow.resize(790,440)
mainWindow.move(300,300)
palette=QPalette()
palette.setBrush(mainWindow.backgroundRole(), QBrush(QPixmap('operations.jpg')))
mainWindow.setPalette(palette)

# 创建并添加各个控件
## 新建模型按钮
importIcon=QIcon("cubes.ico")
importBtn=QPushButton("", startWindow)
importBtn.setStyleSheet("border:0px")
importBtn.resize(135,135)
importBtn.move(235,125)
importBtn.clicked.connect(createModel)
## 添加样本按钮
#sampleIcon=QIcon("addman.ico")
#sampleBtn=QPushButton("添加样本", startWindow)
#sampleBtn.resize(100,50)
#sampleBtn.move(20,90)
#sampleBtn.setStyleSheet("background:#81e0ea;border:2px groove gray;border-radius:10px;padding:2px")
#sampleBtn.clicked.connect(chooseTrainingSample)
## 识别样本按钮
recognIcon=QIcon("man.ico")
recognBtn=QPushButton("识别样本", startWindow)
recognBtn.resize(100,50)
recognBtn.move(20,160)
recognBtn.clicked.connect(chooseTestingSample)
# 标签列表
#labelList=QListWidget(startWindow)
#labelList.resize(260,360)
#labelList.move(260,20)
#labelList.setSortingEnabled(1)
#for i in ['张三','李四','王五']:
#    labelList.insertItem(0,i)

# 显示主窗体
startWindow.show()

# 等待应用退出
sys.exit(app.exec_())
