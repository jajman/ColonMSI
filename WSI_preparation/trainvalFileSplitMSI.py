import os, pickle, random, sys, shutil
from PIL import Image
import tensorflow as tf
import numpy as np
import time

def getFileName(fileName):
    lenFIleName = len(fileName)
    nameEnd = 0
    nameStart = 0
    isDot=False
    for i in reversed(range(lenFIleName)):
        if fileName[i] == '.':
            if isDot:
                continue
            else:
                nameEnd = i
                isDot=True
        if fileName[i] == '/':
            nameStart = i + 1
            break
    return fileName[nameStart:nameEnd]

dirName = '/media/jajman/NewVolume/TCGA_colon_MSI/'


with open('traintest/10FoldCOADREADmssTrain0.bin', 'rb') as f:
    curTrainFiles=pickle.load(f)
print(len(curTrainFiles))
'''for i in curTrainFiles:
    print(i[12:])'''
for i in range(len(curTrainFiles)):
    curTrainFiles[i]=curTrainFiles[i].replace('=',' ')
with open('traintest/10FoldCOADREADmssTest0.bin', 'rb') as f:
    curTestFiles=pickle.load(f)
print(len(curTestFiles))
'''for i in curTestFiles:
    print(i[12:])'''
for i in range(len(curTestFiles)):
    curTestFiles[i]=curTestFiles[i].replace('=',' ')


files = os.listdir(dirName)
files=sorted(files)
fileNum=len(files)
fileCount=0
trainNum=0
testNum=0
for i in files:
    fileCount += 1
    print(str(fileCount) + '/' + str(fileNum))
    files2 = os.listdir(dirName + i)
    for j in files2:
        if j.endswith('svs'):
            fileName =j[:12]
            codeName=j[12:14]
            if codeName=='-0':
                if fileName in curTrainFiles:
                    trainNum += 1
                    print('train move')
                    saveDirName = dirName + i + '/TUMOR/'
                    files3 = os.listdir(saveDirName)
                    for k in files3:
                        copyName = saveDirName + k
                        shutil.copy(copyName, '/home/jajman/train/normal/')

                if fileName in curTestFiles:
                    testNum += 1
                    print('test move')
                    saveDirName = dirName + i + '/TUMOR/'
                    files3 = os.listdir(saveDirName)
                    for k in files3:
                        copyName = saveDirName + k
                        shutil.copy(copyName, '/home/jajman/validation/normal/')

print(trainNum)
print(testNum)

print()
print()

with open('traintest/10FoldCOADREADmsihTrain0.bin', 'rb') as f:
    curTrainFiles=pickle.load(f)
print(len(curTrainFiles))
'''for i in curTrainFiles:
    print(i[12:])'''
for i in range(len(curTrainFiles)):
    curTrainFiles[i]=curTrainFiles[i].replace('=',' ')
with open('traintest/10FoldCOADREADmsihTest0.bin', 'rb') as f:
    curTestFiles=pickle.load(f)
print(len(curTestFiles))
'''for i in curTestFiles:
    print(i[12:])'''
for i in range(len(curTestFiles)):
    curTestFiles[i]=curTestFiles[i].replace('=',' ')

files = os.listdir(dirName)
files=sorted(files)
fileNum=len(files)
fileCount=0
trainNum=0
testNum=0
for i in files:
    fileCount += 1
    print(str(fileCount) + '/' + str(fileNum))
    files2 = os.listdir(dirName + i)
    for j in files2:
        if j.endswith('svs'):
            fileName =j[:12]
            codeName=j[12:14]
            if codeName=='-0':
                if fileName in curTrainFiles:
                    trainNum += 1
                    print('train move')
                    saveDirName = dirName + i + '/TUMOR/'
                    files3 = os.listdir(saveDirName)
                    for k in files3:
                        copyName = saveDirName + k
                        shutil.copy(copyName, '/home/jajman/train/tumor/')

                if fileName in curTestFiles:
                    testNum += 1
                    print('test move')
                    saveDirName = dirName + i + '/TUMOR/'
                    files3 = os.listdir(saveDirName)
                    for k in files3:
                        copyName = saveDirName + k
                        shutil.copy(copyName, '/home/jajman/validation/tumor/')

print(trainNum)
print(testNum)

