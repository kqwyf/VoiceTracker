#!/usr/bin/python3
#encoding: utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget, QMessageBox, QLineEdit, QTableWidgetItem, QGraphicsOpacityEffect, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QIcon

# 回调函数
def chooseModel():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","","*.npy")
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
    name=QFileDialog.getOpenFileNames(startWindow,"打开文件","","*.wav")
    print(name)

def chooseTestingSample():
    name=QFileDialog.getOpenFileName(startWindow,"打开文件","","*.wav")
    print(name)

def showStart():
    modelWindow.hide()
    startWindow.show()

def showRecognWindow():
    mainWindow.hide()
    recognWindow.show()

def startRecognize():
    chooseTestingSample()

def startValidate():
    chooseTestingSample()

def startValidateProcess():
    QMessageBox.information(modelWindow,"识别结果","选定语音对于被测者张三具有98.3%的相似度。可以认为该语音属于张三（男，18岁）。")


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
palette.setBrush(mainWindow.backgroundRole(), QBrush(QPixmap('main.jpg')))
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
## 创建识别窗体
recognWindow=QWidget()
recognWindow.setWindowTitle("VoiceTracker")
recognWindow.resize(790,440)
recognWindow.move(300,300)
palette=QPalette()
palette.setBrush(recognWindow.backgroundRole(), QBrush(QPixmap('recognize.jpg')))
recognWindow.setPalette(palette)
recognWindow.closeEvent=lambda x:showMain()

# 创建透明效果
def getOpacityEffectObj(opacity):
    opacityEffect=QGraphicsOpacityEffect()
    opacityEffect.setOpacity(opacity)
    return opacityEffect

# 创建并添加各个控件
## 新建模型按钮
createBtn=QPushButton("", startWindow)
createBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
createBtn.resize(135,135)
createBtn.move(235,125)
createBtn.clicked.connect(showMain)
## 导入模型按钮
importBtn=QPushButton("", startWindow)
importBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
importBtn.resize(135,135)
importBtn.move(420,125)
importBtn.clicked.connect(chooseModel)
## 语音库按钮
databaseBtn=QPushButton("", mainWindow)
databaseBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
databaseBtn.resize(135,135)
databaseBtn.move(48,125)
databaseBtn.clicked.connect(showModel)
## 添加文件按钮
sampleBtn=QPushButton("", mainWindow)
sampleBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
sampleBtn.resize(135,135)
sampleBtn.move(235,125)
sampleBtn.clicked.connect(chooseTrainingSample)
## 现场录音按钮
recordBtn=QPushButton("", mainWindow)
recordBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
recordBtn.resize(135,135)
recordBtn.move(422,125)
recordBtn.clicked.connect(showRecord)
## 识别样本按钮
recognBtn=QPushButton("", mainWindow)
recognBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
recognBtn.resize(135,135)
recognBtn.move(608,125)
recognBtn.clicked.connect(showRecognWindow)
## 开始/停止录音按钮
recordingBtn=QPushButton("",recordWindow)
recordingBtn.setStyleSheet("border-radius:30px;background:red;")
recordingBtn.resize(60,60)
recordingBtn.move(360,340)
recordingBtn.clicked.connect(startOrStopRecording)
## 话者识别按钮
recognizeBtn=QPushButton("", recognWindow)
recognizeBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
recognizeBtn.resize(135,135)
recognizeBtn.move(235,125)
recognizeBtn.clicked.connect(startRecognize)
## 话者验证按钮
validateBtn=QPushButton("", recognWindow)
validateBtn.setGraphicsEffect(getOpacityEffectObj(0.3))
validateBtn.resize(135,135)
validateBtn.move(422,125)
validateBtn.clicked.connect(startValidate)
## 开始验证按钮
startValidateBtn=QPushButton("开始验证", modelWindow)
#startValidateBtn.setGraphicsEffect(getOpacityEffectObj(0.7))
startValidateBtn.setStyleSheet("background:gray;")
startValidateBtn.resize(135,24)
startValidateBtn.move(600,400)
startValidateBtn.clicked.connect(startValidateProcess)

## 模型搜索框
searchField=QLineEdit(modelWindow)
searchField.setStyleSheet("background:rgba(255,255,255,20%);color:rgba(255,255,255,50%);")
searchField.resize(200,24)
searchField.move(20,20)

# 标签列表
headers=['标签','性别','年龄','备注']
labelList=QTableWidget(modelWindow)
labelList.resize(750,340)
labelList.move(20,56)
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
