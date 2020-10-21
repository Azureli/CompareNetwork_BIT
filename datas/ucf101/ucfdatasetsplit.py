import random
file = open("E:\\计算机视觉实验室\\fsl\\ucf\\classInd.txt","r")
# 随机选取个70类别训练，10个类别validation，21个类别test
ClassList = []
ClassList2 = []
for line in file:
    ClassId, ClassName = line.split()
    ClassList.append(ClassName)

TrainList = random.sample(ClassList, 70)
for ClassName in ClassList:
    if ClassName not in TrainList:
        ClassList2.append(ClassName)
ValList = random.sample(ClassList2, 10)
# 从剩下的类别选取16类
TestList = []
for ClassName in ClassList2:
    if ClassName not in ValList:
        TestList.append(ClassName)
testlist = random.sample(TestList, 21)
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\trainclass.txt", "w") as f:
    for i in range(len(TrainList)):
        f.write(TrainList[i]+'\n')  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\testclass.txt", "w") as f:
    for i in range(len(TestList)):
        f.write(TestList[i]+'\n')  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\valclass.txt", "w") as f:
    for i in range(len(ValList)):
        f.write(ValList[i]+'\n')  # 自带文件关闭功能，不需要再写f.close()

TrainSplit1=[]
TestSplit1=[]
ValSplit1=[]

file = open("E:\\计算机视觉实验室\\fsl\\ucf\\trainlist01.txt","r")
for line in file.readlines():
    dir , classid = line.split()
    classname , videoname = dir.split('/')
    if classname in TrainList:
        TrainSplit1.append(line)
    if classname in TestList:
        TestSplit1.append(line)
    if classname in ValList:
        ValSplit1.append(line)

file = open("E:\\计算机视觉实验室\\fsl\\ucf\\testlist01.txt","r")
for line in file.readlines():
    classname , videoname = line.split('/')
    if classname in TrainList:
        TrainSplit1.append(line)
    if classname in TestList:
        TestSplit1.append(line)
    if classname in ValList:
        ValSplit1.append(line)

with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\trainsplit1.txt", "w") as f:
    for i in range(len(TrainSplit1)):
        f.write(TrainSplit1[i])  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\testsplit1.txt", "w") as f:
    for i in range(len(TestSplit1)):
        f.write(TestSplit1[i])  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\valsplit1.txt", "w") as f:
    for i in range(len(ValSplit1)):
        f.write(ValSplit1[i])  # 自带文件关闭功能，不需要再写f.close()

print("ucf fewshot dataset split1 is established !")



TrainSplit2=[]
TestSplit2=[]
ValSplit2=[]

file = open("E:\\计算机视觉实验室\\fsl\\ucf\\trainlist02.txt","r")
for line in file.readlines():
    dir , classid = line.split()
    classname , videoname = dir.split('/')
    if classname in TrainList:
        TrainSplit2.append(line)
    if classname in TestList:
        TestSplit2.append(line)
    if classname in ValList:
        ValSplit2.append(line)

file = open("E:\\计算机视觉实验室\\fsl\\ucf\\testlist02.txt","r")
for line in file.readlines():
    classname , videoname = line.split('/')
    if classname in TrainList:
        TrainSplit2.append(line)
    if classname in TestList:
        TestSplit2.append(line)
    if classname in ValList:
        ValSplit2.append(line)

with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\trainsplit2.txt", "w") as f:
    for i in range(len(TrainSplit2)):
        f.write(TrainSplit2[i])  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\testsplit2.txt", "w") as f:
    for i in range(len(TestSplit2)):
        f.write(TestSplit2[i])  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\valsplit2.txt", "w") as f:
    for i in range(len(ValSplit2)):
        f.write(ValSplit2[i])  # 自带文件关闭功能，不需要再写f.close()
print("ucf fewshot dataset split2 is established !")



TrainSplit3=[]
TestSplit3=[]
ValSplit3=[]

file = open("E:\\计算机视觉实验室\\fsl\\ucf\\trainlist03.txt","r")
for line in file.readlines():
    dir , classid = line.split()
    classname , videoname = dir.split('/')
    if classname in TrainList:
        TrainSplit3.append(line)
    if classname in TestList:
        TestSplit3.append(line)
    if classname in ValList:
        ValSplit3.append(line)

file = open("E:\\计算机视觉实验室\\fsl\\ucf\\testlist03.txt","r")
for line in file.readlines():
    classname , videoname = line.split('/')
    if classname in TrainList:
        TrainSplit3.append(line)
    if classname in TestList:
        TestSplit3.append(line)
    if classname in ValList:
        ValSplit3.append(line)

with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\trainsplit3.txt", "w") as f:
    for i in range(len(TrainSplit3)):
        f.write(TrainSplit3[i])  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\testsplit3.txt", "w") as f:
    for i in range(len(TestSplit3)):
        f.write(TestSplit3[i])  # 自带文件关闭功能，不需要再写f.close()
with open("E:\\计算机视觉实验室\\fsl\\ucf\\fewshot\\valsplit3.txt", "w") as f:
    for i in range(len(ValSplit3)):
        f.write(ValSplit3[i])  # 自带文件关闭功能，不需要再写f.close()
print("ucf fewshot dataset split3 is established !")