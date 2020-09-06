#FINAL PROJECT BY SHREY SHARMA(ss4399) & ANKITA TALWAR(at754)

import sys
import time
import array
import copy
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid

start_time = time.time()

def extractCol(matrix, i):
    return [[row[i]] for row in matrix]

def mergeCol(a, b):
    return [x + y for x, y in zip(a, b)]

#Chi Square
def chiSqr(X, y, top_fea):
    rows = len(X)
    cols = len(X[0])
    T = []
    for j in range(0, cols):
        ct = [[1,1],[1,1],[1,1]]
        for i in range(0, rows):
            if y[i] == 0:
                if X[i][j] == 0:
                    ct[0][0] += 1
                elif X[i][j] == 1:
                    ct[1][0] += 1
                elif X[i][j] == 2:
                    ct[2][0] += 1
            elif y[i] == 1:
                if X[i][j] == 0:
                     ct[0][1] += 1
                elif X[i][j] == 1:
                    ct[1][1] += 1
                elif X[i][j] == 2:
                    ct[2][1] += 1
        col_totals = [ sum(x) for x in ct]
        row_totals = [ sum(x) for x in zip(*ct) ]
        total = sum(col_totals)
        exp_value = [[(row*col)/total for row in row_totals] for col in col_totals]
        sqr_value = [[((ct[i][j] - exp_value[i][j])**2)/exp_value[i][j] for j in range(0,len(exp_value[0]))] for i in range(0,len(exp_value))]
        x_2 = sum([sum(x) for x in zip(*sqr_value)])
        T.append(x_2)
    indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)
    indx = indices[:top_fea]
    return indx

def CreateDataSet(fea, dat):
    newData = extractCol(dat, fea[0])
    newLab = array.array("i")
    fea.remove(fea[0])
    length = len(fea)
    for i in range(0, length, 1):
        temp = extractCol(dat, fea[0])
        newData = mergeCol(newData, temp)
        fea.remove(fea[0])
    return newData


# Read data
datafile = sys.argv[1]
data = []

with open(datafile, "r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        data.append(l)

# Read labels
labelfile = sys.argv[2]
trainlabels = array.array("i")
with open(labelfile, "r") as infile:
    for line in infile:
        temp = line.split()
        trainlabels.append(int(temp[0]))


feat = 15
rows = len(data)
cols = len(data[0])
rowsl = len(trainlabels)

# print("Size of data: ",sys.getsizeof(data))
print(" %s seconds" % (time.time() - start_time))

# Dimensionality Reduction
neededFea = chiSqr(data, trainlabels, 15)

savedFea = copy.deepcopy(neededFea)

data1 = CreateDataSet(neededFea, data)

svm = svm.SVC(gamma=0.001)
logReg = linear_model.LogisticRegression()
NaiveB = GaussianNB()
NearestCen= NearestCentroid()

allAccuracies = array.array("f")
allFeatures = []

accuracy_svm = 0
accuracy_score = 0
accuracy_log = 0
accuracy_gnb = 0
accuracy_nc = 0

my_accuracy = 0
#5 -fold cross validation
iterations = 5
print("Cross validation iteration: ", end="")
for i in range(iterations):

    print(i)

    X_train, X_test, y_train, y_test = train_test_split(
        data1, trainlabels, test_size=0.2)

    newRows = len(X_train)
    newCols = len(X_train[0])
    newRowst = len(X_test)
    newColst = len(X_test[0])

    newRowsL = len(y_train)
    Features = chiSqr(X_train, y_train, feat)

    allFeatures.append(Features)
    argument = copy.deepcopy(Features)

    data_fea = CreateDataSet(argument, X_train)
    # print("New Data Made, rows= ",len(data_fea)," cols= ",len(data_fea[0]))

    svm.fit(data_fea, y_train)
    logReg.fit(data_fea, y_train)
    NaiveB.fit(data_fea, y_train)
    NearestCen.fit(data_fea, y_train)

    TestFeatures = chiSqr(X_test, y_test, feat)
    test_fea = CreateDataSet(TestFeatures, X_test)

    len_test_fea = len(test_fea)
    count_svm = 0
    count_log = 0
    count_nb = 0
    count_nc = 0
    count = 0
    for j in range(0, len_test_fea, 1):
        predLab_svm = int(svm.predict([test_fea[j]]))
        predLab_log = int(logReg.predict([test_fea[j]]))
        predLab_gnb = int(NaiveB.predict([test_fea[j]]))
        predLab_nc = int(NearestCen.predict([test_fea[j]]))
        h = predLab_svm + predLab_log + predLab_gnb + predLab_nc
        if (h >= 3):
            predict = 1
        elif (h <= 1):
            predict = 0
        else:
           predict = predLab_svm
        if (predict == y_test[j]):
            count += 1
        if (predLab_svm == y_test[j]):
            count_svm += 1
        if (predLab_log == y_test[j]):
            count_log += 1
        if (predLab_gnb == y_test[j]):
            count_nb+= 1
        if (predLab_nc == y_test[j]):
            count_nc += 1



    accuracy_svm += count_svm / len_test_fea
    accuracy_log += count_log / len_test_fea

    accuracy_gnb += count_nb / len_test_fea
    accuracy_nc += count_nc / len_test_fea

    my_accuracy += count / len_test_fea
    allAccuracies.append(count / len_test_fea)

print(" Done", end="")
print(" %s seconds" % (time.time() - start_time))

bestAc = max(allAccuracies)
bestInd = allAccuracies.index(bestAc)
bestFeatures = allFeatures[bestInd]

print("\nFeatures: ", feat)

originalFea = array.array("i")
for i in range(0, feat, 1):
    realIndex = savedFea[bestFeatures[i]]
    originalFea.append(realIndex)

print("The features are: ", originalFea)

#writing Final features to file
feature_out=open("FeatureLabels","w+")
feature_out.write(str(originalFea))

# Calculating Accuracy
argument1 = copy.deepcopy(originalFea)
AccData = CreateDataSet(argument1, data)

svm.fit(AccData, trainlabels)
logReg.fit(AccData, trainlabels)
NaiveB.fit(AccData, trainlabels)
NearestCen.fit(AccData, trainlabels)

svm_counter = 0
Counter = 0
k = len(AccData)
for i in range(0, k, 1):
    predLab_svm = int(svm.predict([AccData[i]]))
    predLab_log = int(logReg.predict([AccData[i]]))
    predLab_gnb = int(NaiveB.predict([AccData[i]]))
    predLab_nc = int(NearestCen.predict([AccData[i]]))
    h = predLab_svm + predLab_log + predLab_gnb + predLab_nc
    if (h >= 3):
        predict = 1
    elif (h <= 1):
        predict = 0
    else:
        predict = predLab_svm
    if (predict == trainlabels[i]):
        Counter += 1
    if (predLab_svm == trainlabels[i]):
        svm_counter += 1

FinalAcc = Counter / k
SVMAc = svm_counter / k
print("The Accuracy is: ", FinalAcc * 100)

# Reading Test data
testfile = sys.argv[3]
testdata =[]

with open(testfile, "r") as tfile:
    for line in tfile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        testdata.append(l)

# Reducing Dimensions to selected features
argument2 = copy.deepcopy(originalFea)
testdata1 = CreateDataSet(argument2, testdata)

# create a file
f1 = open("testLabels", "w+")

for i in range(0, len(testdata1), 1):
    lab1 = int(svm.predict([testdata1[i]]))
    lab2 = int(logReg.predict([testdata1[i]]))
    lab3 = int(NaiveB.predict([testdata1[i]]))
    lab4 = int(NearestCen.predict([testdata1[i]]))
    h = lab1 + lab2 + lab3 + lab4
    if (h >= 3):
        f1.write(str(1) + " " + str(i) + "\n")
    elif (h <= 1):
        f1.write(str(0) + " " + str(i) + "\n")
    else:
        f1.write(str(lab1) + " " + str(i) + "\n")

print(" %s seconds " % (time.time() - start_time))


