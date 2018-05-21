#!/usr/bin/python3
#encoding: utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QPushButton, QFileDialog

# 回调函数
def chooseModel():
    name=QFileDialog.getOpenFileName(mainWindow,"打开文件","")
    print(name)

def chooseTrainingSample():
    name=QFileDialog.getOpenFileName(mainWindow,"打开文件","")
    print(name)

def chooseTestingSample():
    name=QFileDialog.getOpenFileName(mainWindow,"打开文件","")
    print(name)

# 创建应用
app=QApplication(sys.argv)

# 创建主窗体
mainWindow=QWidget()
mainWindow.setWindowTitle("VoiceTracker")
mainWindow.resize(320,400)
mainWindow.move(300,300)

# 创建并添加各个控件
## 导入模型按钮
importBtn=QPushButton("导入模型", mainWindow)
importBtn.resize(100,50)
importBtn.move(200,20)
importBtn.clicked.connect(chooseModel)
## 添加样本按钮
sampleBtn=QPushButton("添加样本", mainWindow)
sampleBtn.resize(100,50)
sampleBtn.move(200,90)
sampleBtn.clicked.connect(chooseTrainingSample)
## 识别样本按钮
recognBtn=QPushButton("识别样本", mainWindow)
recognBtn.resize(100,50)
recognBtn.move(200,160)
recognBtn.clicked.connect(chooseTestingSample)
# 标签列表
labelList=QListWidget(mainWindow)
labelList.resize(160,360)
labelList.move(20,20)
labelList.setSortingEnabled(1)
for i in ['张三','李四','王五']:
    labelList.insertItem(0,i)

# 显示主窗体
mainWindow.show()

# 等待应用退出
sys.exit(app.exec_())
