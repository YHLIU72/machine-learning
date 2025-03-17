import numpy as np

#创建测试数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]#嵌套列表
    labels = ["放贷","不放贷"]
    return dataSet, labels
#计算信息熵
def calcShannonEnt(dataSet):
    #len函数可以用于多种数据类型，包括字符串、列表、元组、字典、集合等。
    #len() 函数返回的是 嵌套列表的外层列表的长度，即外层列表中包含的元素个数。对于嵌套列表来说，len() 不会递归计算内层列表的长度。
    numEntries = len(dataSet)  # 计算数据集中的样本数量
    labelCounts = {}  # 创建一个空字典，用于存储每个类别的样本数量
    for featVec in dataSet:  # 遍历数据集中的每一条记录
        currentLabel = featVec[-1]  # 获取当前记录的类别标签
        if currentLabel not in labelCounts.keys():  # 如果当前类别标签不在字典中
            labelCounts[currentLabel] = 0  # 初始化该类别标签的计数为 0
        labelCounts[currentLabel] += 1  # 对当前类别标签的计数加 1
    shannonEnt = 0.0  # 初始化信息熵为 0.0
    for key in labelCounts:  # 遍历字典中的每个类别标签
        prob = float(labelCounts[key]) / numEntries  # 计算当前类别的概率
        shannonEnt -= prob * np.log2(prob)  # 计算信息熵并累加
    return shannonEnt  # 返回计算得到的信息熵
#划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建一个空列表，用于存储划分后的数据集
    for featVec in dataSet:  # 遍历数据集中的每一条记录
        if featVec[axis] == value:  # 如果当前记录在指定特征轴上的值等于目标值
            reducedFeatVec = featVec[:axis]  # 取特征轴之前的部分
            reducedFeatVec.extend(featVec[axis+1:])  # 拼接特征轴之后的部分
            retDataSet.append(reducedFeatVec)  # 将处理后的记录添加到新数据集中
    return retDataSet  # 返回划分后的数据集

#选择最好的数据集划分方式
"""" 
dataset:数据集,一个嵌套列表
        在函数中调用的数据需要满足一定的要求：
            第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；
            第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
    函数的参数是一个嵌套列表，其中每个子列表表示一个实例，最后一个元素是该实例的类别标签。
    函数的返回值是一个整数，表示最好的特征的索引。
返回值：    bestFeature:最好的特征"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 计算特征的数量（减去标签列）
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的信息熵
    bestInfoGain = 0.0  # 初始化最佳信息增益
    bestFeature = -1  # 初始化最佳特征的索引

    for i in range(numFeatures):  # 遍历每个特征
        featList = [example[i] for example in dataSet]  # 列表推导式，获取当前特征的所有值
        uniqueVals = set(featList)  # 创建集合，去重，得到当前特征的所有可能取值
        newEntropy = 0.0  # 初始化条件熵

        for value in uniqueVals:  # 遍历当前特征的每个取值
            subDataSet = splitDataSet(dataSet, i, value)  # 根据当前特征和取值划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算条件熵

        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if (infoGain > bestInfoGain):  # 如果当前信息增益更大
            bestInfoGain = infoGain  # 更新最佳信息增益
            bestFeature = i  # 更新最佳特征的索引

    return bestFeature  # 返回最佳特征的索引

#多数表决
def majorityCnt(classList):
    classCount = {}  # 创建一个空字典，用于存储每个类别的样本数量
    for vote in classList:  # 遍历类别标签列表
        if vote not in classCount.keys():  # 如果当前类别标签不在字典中
            classCount[vote] = 0  # 初始化该类别标签的计数为 0
        classCount[vote] += 1  # 对当前类别标签的计数加 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)  # 对字典按值降序排序
    return sortedClassCount[0][0]  # 返回计数最多的类别标签
#创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 提取数据集的类别标签列表
    if classList.count(classList[0]) == len(classList):  # 如果类别标签列表中的所有元素都相同
        return classList[0]  # 返回该元素作为叶子节点
    if len(dataSet[0]) == 1:  # 如果数据集中只有一个特征
        return majorityCnt(classList)  # 返回类别标签列表中计数最多的类别作为叶子节点
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最佳特征的索引
    bestFeatLabel = labels[bestFeat]  # 获取最佳特征的标签
    myTree = {bestFeatLabel: {}}  # 创建一个空字典，用于存储决策树
    del(labels[bestFeat])  # 从标签列表中删除最佳特征的标签
    featValues = [example[bestFeat] for example in dataSet]  # 提取最佳特征的所有取值
    uniqueVals = set(featValues)  # 创建集合，去重，得到最佳特征的所有可能取值
    for value in uniqueVals:  # 遍历最佳特征的每个取值
        subLabels = labels[:]  # 创建一个子标签列表的副本
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 递归构建决策树
    return myTree  # 返回构建好的决策树
#使用matplotlib绘制决策树
""""
annotate 是 matplotlib 库中的一个方法，用于在图中添加注释（如文本、箭头等）。以下是其常用输入参数和输出行为的详细解释：

常用输入参数
text：
    类型：字符串
    作用：注释的文本内容。
    示例："Feature A" 或 "Yes"。
xy：
    类型：元组 (x, y)
    作用：箭头指向的位置（即注释指向的目标点）。
    示例：(0.3, 0.7) 表示目标点位于图的左上方。
xytext：
    类型：元组 (x, y)
    作用：注释文本的位置。
    示例：(0.5, 0.5) 表示文本位于图的中心。
xycoords：
    类型：字符串
    作用：指定 xy 的坐标系。
    常用值：
    'data'：使用数据坐标（默认）。
    'axes fraction'：使用相对于坐标轴的比例（0 到 1）。
    'figure fraction'：使用相对于整个图的比例（0 到 1）。
    示例：xycoords='axes fraction'。
textcoords：
    类型：字符串
    作用：指定 xytext 的坐标系。
    常用值：与 xycoords 相同。
    示例：textcoords='axes fraction'。
arrowprops：
    类型：字典
    作用：设置箭头的样式。
    常用键值对：
    arrowstyle：箭头样式，如 "<-"（简单箭头）、"->"、"fancy" 等。
    color：箭头颜色。
    linewidth：箭头线宽。
    示例：arrowprops=dict(arrowstyle="<-", color="red")。
bbox：
    类型：字典
    作用：设置注释文本的边框样式。
    常用键值对：
    boxstyle：边框样式，如 "round"（圆角）、"sawtooth"（锯齿形）等。
    fc：填充颜色。
    ec：边框颜色。
    示例：bbox=dict(boxstyle="round", fc="yellow", ec="black")。
va 和 ha：
    类型：字符串
    作用：分别设置文本的垂直和水平对齐方式。
    常用值：
    va："center"（居中）、"top"（顶部）、"bottom"（底部）。
    ha："center"（居中）、"left"（左对齐）、"right"（右对齐）。
    示例：va="center", ha="center"。
输出行为:
    无返回值。
    作用：在图中添加注释，并绘制箭头（如果指定了 arrowprops）。

示例：


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], label="Line")

# 添加注释
ax.annotate("Point A", xy=(0.5, 0.5), xytext=(0.2, 0.8),
            xycoords='data', textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", color="blue"),
            bbox=dict(boxstyle="round", fc="yellow", ec="black"),
            va="center", ha="center")

plt.show()
"""

import matplotlib.pyplot as plt

# 定义节点的样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制决策树中的一个节点。

    参数:
    - nodeTxt: 节点显示的文本内容（字符串）。
    - centerPt: 节点的中心位置坐标（元组 (x, y)）。
    - parentPt: 父节点的位置坐标（元组 (x, y)）。
    - nodeType: 节点的样式（字典，包含方框形状、填充颜色等）。

    作用:
    - 在图中绘制一个节点，并连接到其父节点。
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# 绘制树的宽度
def getNumLeafs(myTree):
    """
    计算决策树的叶子节点数量。

    参数:
    - myTree: 决策树（字典结构）。

    返回值:
    - numLeafs: 叶子节点的数量（整数）。
    """
    numLeafs = 0  # 初始化叶子节点数量为 0
    firstStr = list(myTree.keys())[0]  # 获取决策树的第一个键（即当前节点的特征）
    secondDict = myTree[firstStr]  # 获取当前节点的子节点字典
    for key in secondDict.keys():  # 遍历子节点字典的键
        if type(secondDict[key]).__name__ == 'dict':  # 如果子节点仍然是字典（即非叶子节点）
            numLeafs += getNumLeafs(secondDict[key])  # 递归计算子树的叶子节点数量
        else:  # 如果子节点不是字典（即叶子节点）
            numLeafs += 1  # 叶子节点数量加 1
    return numLeafs  # 返回叶子节点数量

# 绘制树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 绘制决策树
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    """
    递归绘制决策树。

    参数:
    - myTree: 决策树（字典结构）。
    - parentPt: 父节点的位置坐标（元组 (x, y)）。
    - nodeTxt: 节点显示的文本内容（字符串）。

    作用:
    - 递归绘制决策树的每个节点，并连接到其父节点。
    """
    numLeafs = getNumLeafs(myTree)  # 获取当前子树的叶子节点数量
    depth = getTreeDepth(myTree)  # 获取当前子树的深度
    firstStr = list(myTree.keys())[0]  # 获取当前节点的特征名称
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 计算当前节点的中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 在父节点和当前节点之间绘制文本
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制当前节点
    secondDict = myTree[firstStr]  # 获取当前节点的子节点字典
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 更新 y 偏移量，用于绘制下一层节点
    for key in secondDict.keys():  # 遍历子节点字典的键
        if type(secondDict[key]).__name__ == 'dict':  # 如果子节点仍然是字典（即非叶子节点）
            plotTree(secondDict[key], cntrPt, str(key))  # 递归绘制子树
        else:  # 如果子节点不是字典（即叶子节点）
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW  # 更新 x 偏移量，用于绘制叶子节点
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)  # 绘制叶子节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  # 在叶子节点和父节点之间绘制文本
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD  # 恢复 y 偏移量，用于绘制同一层的其他节点

def createPlot(inTree):
    """
    创建并显示决策树的图形。

    参数:
    - inTree: 决策树（字典结构）。

    作用:
    - 初始化图形窗口，设置绘图区域，计算决策树的宽度和深度，并调用 plotTree 函数绘制决策树。
    """
    fig = plt.figure(1, facecolor='white')  # 创建一个新的图形窗口，背景颜色为白色
    fig.clf()  # 清除图形窗口中的内容
    axprops = dict(xticks=[], yticks=[])  # 设置坐标轴属性，不显示刻度
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 创建一个子图，不显示边框
    plotTree.totalW = float(getNumLeafs(inTree))  # 计算决策树的宽度（叶子节点数量）
    plotTree.totalD = float(getTreeDepth(inTree))  # 计算决策树的深度
    plotTree.xOff = -0.5 / plotTree.totalW  # 初始化 x 偏移量
    plotTree.yOff = 1.0  # 初始化 y 偏移量
    plotTree(inTree, (0.5, 1.0), '')  # 调用 plotTree 函数绘制决策树
    plt.show()  # 显示图形

#创建分类器
def classify(inputTree, featLabels, testVec):
    """
    使用决策树进行分类。

    参数:
    - inputTree: 已经生成的决策树（字典结构）。
    - featLabels: 决策树的特征标签列表。
    - testVec: 测试数据，即待分类的特征向量。

    返回值:
    - classLabel: 分类结果。
    """
    firstStr = list(inputTree.keys())[0]  # 获取决策树的第一个键（即根节点）
    secondDict = inputTree[firstStr]  # 获取根节点的子节点字典
    featIndex = featLabels.index(firstStr)  # 获取根节点特征在特征标签列表中的索引
    for key in secondDict.keys():  # 遍历子节点字典的键
        if testVec[featIndex] == key:  # 如果测试数据的特征值等于当前子节点的键
            if type(secondDict[key]).__name__ == 'dict':  # 如果子节点仍然是字典（即非叶子节点）
                classLabel = classify(secondDict[key], featLabels, testVec)  # 递归分类
            else:  # 如果子节点不是字典（即叶子节点）
                classLabel = secondDict[key]  # 分类结果为当前子节点的值
    return classLabel  # 返回分类结果
