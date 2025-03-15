#导入包
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
from os import listdir
#解决中文乱码问题
myfont = fm.FontProperties(fname='C:\Windows\Fonts\simhei.ttf',size=14)

#创建示例数据集和标签，用于测试
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#kNN分类算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #tile用于扩展数组，将inX扩展成与dataSet相同形状的矩阵，方便后续计算
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #axis=1表示按行求和
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，得到距离
    distances = sqDistances**0.5
    #对距离进行排序，返回的是索引
    sortedDistIndicies = distances.argsort()
    #创建一个字典，用于存储每个标签出现的次数
    classCount = {}
    #选出前k个距离最小的点
    for i in range(k):
        #获取标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #将标签加入字典，如果标签已经存在，则次数加1
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #对字典进行排序，返回的是一个列表，列表中的每个元素是一个元组，元组的第一个元素是标签，第二个元素是次数
    #sorted(iterable, key=None, reverse=False)
    #  iterable -- 可迭代对象。 key -- 主要是用来进行比较的元素，只有一个参数。 reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回出现次数最多的标签
    return sortedClassCount[0][0]

"""with open('example.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print("文件内容：")
    print(content)

# 打开文件并写入内容
with open('example.txt', 'w', encoding='utf-8') as file:
    file.write("这是新写入的内容。\n")
    file.write("第二行内容。\n")

# 打开文件并追加内容
with open('example.txt', 'a', encoding='utf-8') as file:
    file.write("这是追加的内容。\n")"""

#海伦约会项目
#约会网站数据处理
def file2matrix(filename):
    pth = '../data/'
    filename = pth + filename
    ## 打开文件并读取内容
    with open(filename) as fr:  # 使用 with 语句确保文件正确关闭
        arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 添加代码将文件内容转换为矩阵
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 移除字符串首尾的空格和换行符
        #str.strip([chars])chars:可选参数，指定要移除的字符，默认值为空白字符（包括空格、换行符 \n、制表符 \t 等）
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        #1表示不喜欢，2表示魅力一般，3表示极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

"""plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, alpha=None, ...)
主要参数说明
x, y:

必需参数，表示数据点的 x 和 y 坐标。
可以是列表、数组或标量。
s:

可选参数，表示点的大小。
可以是标量或与 x, y 长度相同的数组。
默认值为 None，使用默认大小。
c:

可选参数，表示点的颜色。
可以是颜色字符串（如 'r' 表示红色）、颜色列表或数值数组。
默认值为 None，使用默认颜色。
marker:

可选参数，表示点的形状。
常用值：'o'（圆形，默认）、's'（方形）、'^'（三角形）、'D'（菱形）等。
默认值为 'o'。
cmap:

可选参数，表示颜色映射。
当 c 为数值数组时，用于将数值映射到颜色。
常用值：'viridis'、'plasma'、'inferno' 等。
alpha:

可选参数，表示点的透明度。
取值范围为 0（完全透明）到 1（完全不透明）。
默认值为 None，使用完全不透明。
label:

可选参数，用于设置图例标签。
默认值为 None。
edgecolors:

可选参数，表示点的边缘颜色。
默认值为 None，使用与点相同的颜色"""

#数据可视化
def showData(datingDataMat,datingLabels):
    fig,axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(10,10))
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        elif i == 2:
            LabelsColors.append('orange')
        elif i == 3:
            LabelsColors.append('red')

    # 绘制散点图
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=.5,label=datingLabels)

    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=.5,label=datingLabels)

    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=.5,label=datingLabels)

    # 设置标题
    axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',fontproperties=myfont)
    axs[0][1].set_title('每年获得的飞行常客里程数与每周消费的冰淇淋公升数',fontproperties=myfont)
    axs[1][0].set_title('玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数',fontproperties=myfont)
    # 设置x轴标签
    axs[0][0].set_xlabel('每年获得的飞行常客里程数',fontproperties=myfont)
    axs[0][1].set_xlabel('每年获得的飞行常客里程数',fontproperties=myfont)
    axs[1][0].set_xlabel('玩视频游戏所消耗时间占比',fontproperties=myfont)
    # 设置y轴标签
    axs[0][0].set_ylabel('玩视频游戏所消耗时间占比',fontproperties=myfont)
    axs[0][1].set_ylabel('每周消费的冰淇淋公升数',fontproperties=myfont)
    axs[1][0].set_ylabel('每周消费的冰淇淋公升数',fontproperties=myfont)
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
 
    # 调整子图之间的间距
    plt.tight_layout()
    #显示图例
    plt.legend(['不喜欢','魅力一般','极具魅力'],loc='upper left',prop=myfont)
    # 显示图形
    plt.show()

#归一化特征值
def autoNorm(dataSet):
    #获取数据集列中的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #创建一个和dataSet一样大小的矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #tile函数将变量内容复制成输入矩阵同样大小的矩阵，（m，1）指定复制的行数和列数
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试约会网站预测函数
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

#约会网站预测函数
"""通过输入一个人的三个特征值，对其进行分类"""
def classifyperson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult - 1])

#手写数字识别系统
#将图像转换为向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
#手写数字识别系统测试
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('../data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('../data/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('../data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult,classNumStr))
        if(classifierResult!= classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

#调用scikit-learn库实现kNN算法
"""KNeighborsClassifier(n_neighbors=3) 是 scikit-learn 库中用于实现 k 近邻（k-Nearest Neighbors, kNN）算法的分类器。以下是它的详细参数和用法：

主要参数

n_neighbors（默认值为 5）：
表示 k 值，即选择多少个最近邻居。
较小的 k 值会使模型对噪声更敏感，较大的 k 值会使模型更平滑，但可能忽略局部特征。

weights（默认值为 'uniform'）：
表示邻居的权重计算方式。
'uniform'：所有邻居的权重相同。
'distance'：邻居的权重与距离成反比，距离越近的邻居权重越大。
也可以传入自定义的权重函数。

algorithm（默认值为 'auto'）：
表示用于计算最近邻居的算法。
'auto'：自动选择最佳算法。
'ball_tree'：使用 BallTree 数据结构。
'kd_tree'：使用 KDTree 数据结构。
'brute'：使用暴力搜索。

metric（默认值为 'minkowski'）：
表示距离度量方法。
'minkowski'：闵可夫斯基距离（默认 p=2，即欧氏距离）。
其他选项包括 'euclidean'（欧氏距离）、'manhattan'（曼哈顿距离）等。

p（默认值为 2）：
当 metric='minkowski' 时，表示闵可夫斯基距离的幂参数。
p=1：曼哈顿距离。
p=2：欧氏距离。

leaf_size（默认值为 30）：
当使用 'ball_tree' 或 'kd_tree' 时，表示叶子节点的大小。
影响树的构建和查询速度。

n_jobs（默认值为 None）：
表示并行计算时使用的 CPU 核心数。
None：使用单核。
-1：使用所有可用核心。

主要方法
fit(X, y)：
训练模型。
X：训练数据，形状为 (n_samples, n_features) 的数组。
y：标签，形状为 (n_samples,) 的数组。

predict(X)：
对测试数据进行分类。
返回预测的类别，形状为 (n_samples,) 的数组。

predict_proba(X)：
返回测试数据属于每个类别的概率。
形状为 (n_samples, n_classes) 的数组。

score(X, y)：
返回模型在测试数据上的准确率。"""

from sklearn.neighbors import KNeighborsClassifier
def sk_learn_kNN():
    #创建分类器
    knn = KNeighborsClassifier(n_neighbors=3,algorithm='auto')
    hwLabels = []
    trainingFileList = listdir('../data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('../data/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('../data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../data/testDigits/%s' % fileNameStr)
        knn.fit(trainingMat,hwLabels)
        classifierResult = knn.predict(vectorUnderTest)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult,classNumStr))
        if(classifierResult!= classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))



if __name__ == '__main__':

    # #测试classify0
    # group,labels = createDataSet()
    # print(group,labels)
    # test = [0,0]
    # test_class = classify0(test,group,labels,3)
    # print(test_class)

    # #测试file2matrix
    # datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    # print(datingDataMat[0:5,:],datingLabels[0:5])
    
    # #测试showData
    # showData(datingDataMat,datingLabels)

    # # datingClassTest()
    # classifyperson()

    sk_learn_kNN()

