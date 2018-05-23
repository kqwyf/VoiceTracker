#!/usr/bin/python3
#encoding: utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget, QTableWidgetItem, QGraphicsOpacityEffect, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QIcon

# 回调函数
def chooseModel():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","")
    print(name)
    if ''!=name[0]:
        startWindow.hide()
        mainWindow.show()

def showMain():
    startWindow.hide()
    modelWindow.hide()
    recordWindow.hide()
    mainWindow.show()

def showModel():
    mainWindow.hide()
    modelWindow.show()

def showRecord():
    mainWindow.hide()
    recordWindow.show()

def chooseTrainingSample():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","")
    print(name)

def chooseTestingSample():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","")
    print(name)

def showStart():
    modelWindow.hide()
    recordWindow.hide()
    startWindow.show()

recording=False
def startOrStopRecording():
    global recording
    if recording:
        recordingBtn.resize(60,60)
        recordingBtn.move(360,340)
        recordingBtn.setStyleSheet("border-radius:30px;background:red;")
        recording=False
    else:
        recordingBtn.resize(40,40)
        recordingBtn.move(370,350)
        recordingBtn.setStyleSheet("background:red;")
        recording=True

# 创建应用
app=QApplication(sys.argv)

# 创建各个窗体
## 创建起始窗体
startWindow=QWidget()
startWindow.setWindowTitle("VoiceTracker")
startWindow.resize(790,440)
startWindow.move(300,300)
palette=QPalette()
palette.setBrush(startWindow.backgroundRole(), QBrush(QPixmap('start.jpg')))
startWindow.setPalette(palette)
## 创建主窗体
mainWindow=QWidget()
mainWindow.setWindowTitle("VoiceTracker")
mainWindow.resize(790,440)
mainWindow.move(300,300)
palette=QPalette()
palette.setBrush(mainWindow.backgroundRole(), QBrush(QPixmap('operations.jpg')))
mainWindow.setPalette(palette)
mainWindow.closeEvent=lambda x:showStart()
## 创建语音库窗体
modelWindow=QWidget()
modelWindow.setWindowTitle("VoiceTracker")
modelWindow.resize(790,440)
modelWindow.move(300,300)
palette=QPalette()
palette.setBrush(modelWindow.backgroundRole(), QBrush(QPixmap('blank.jpg')))
modelWindow.setPalette(palette)
modelWindow.closeEvent=lambda x:showMain()
## 创建录音窗体
recordWindow=QWidget()
recordWindow.setWindowTitle("VoiceTracker")
recordWindow.resize(790,440)
recordWindow.move(300,300)
palette=QPalette()
palette.setBrush(recordWindow.backgroundRole(), QBrush(QPixmap('record.jpg')))
recordWindow.setPalette(palette)
recordWindow.closeEvent=lambda x:showMain()

# 创建透明效果
def getOpacityEffectObj():
    opacityEffect=QGraphicsOpacityEffect()
    opacityEffect.setOpacity(0.3)
    return opacityEffect

# 创建并添加各个控件
## 新建模型按钮
createBtn=QPushButton("", startWindow)
createBtn.setGraphicsEffect(getOpacityEffectObj())
createBtn.resize(135,135)
createBtn.move(235,125)
createBtn.clicked.connect(showMain)
## 导入模型按钮
importBtn=QPushButton("", startWindow)
importBtn.setGraphicsEffect(getOpacityEffectObj())
importBtn.resize(135,135)
importBtn.move(425,125)
importBtn.clicked.connect(chooseModel)
## 语音库按钮
databaseBtn=QPushButton("", mainWindow)
databaseBtn.setGraphicsEffect(getOpacityEffectObj())
databaseBtn.resize(135,135)
databaseBtn.move(48,125)
databaseBtn.clicked.connect(showModel)
## 添加文件按钮
sampleBtn=QPushButton("", mainWindow)
sampleBtn.setGraphicsEffect(getOpacityEffectObj())
sampleBtn.resize(135,135)
sampleBtn.move(235,125)
sampleBtn.clicked.connect(chooseTrainingSample)
## 现场录音按钮
recordBtn=QPushButton("", mainWindow)
recordBtn.setGraphicsEffect(getOpacityEffectObj())
recordBtn.resize(135,135)
recordBtn.move(425,125)
recordBtn.clicked.connect(showRecord)
## 识别样本按钮
recognBtn=QPushButton("", mainWindow)
recognBtn.setGraphicsEffect(getOpacityEffectObj())
recognBtn.resize(135,135)
recognBtn.move(615,125)
recognBtn.clicked.connect(chooseTestingSample)
## 开始/停止录音按钮
recordingBtn=QPushButton("",recordWindow)
recordingBtn.setStyleSheet("border-radius:30px;background:red;")
recordingBtn.resize(60,60)
recordingBtn.move(360,340)
recordingBtn.clicked.connect(startOrStopRecording)
# 标签列表
headers=['标签','性别','年龄','相似度']
labelList=QTableWidget(modelWindow)
labelList.resize(750,400)
labelList.move(20,20)
labelList.setRowCount(3)
labelList.setColumnCount(len(headers))
labelList.setStyleSheet("background:rgba(255,255,255,20%);border:white;color:white")
labelList.setHorizontalHeaderLabels(headers)
labelList.horizontalHeader().setStretchLastSection(True)
labelList.horizontalHeader().setStyleSheet("background:gray;")
labelList.verticalHeader().setStyleSheet("background:gray;")
names=['张三','李四','王五']
sexes=['男','女','男']
ages=['18','36','35']
rates=['0.78','0.22','0.26']
for i in range(len(names)):
    labelList.setItem(i,0,QTableWidgetItem(names[i]))
    labelList.setItem(i,1,QTableWidgetItem(sexes[i]))
    labelList.setItem(i,2,QTableWidgetItem(ages[i]))
    #labelList.setItem(i,3,QTableWidgetItem(rates[i]))

# 显示起始窗体
startWindow.show()

# 等待应用退出
sys.exit(app.exec_())
