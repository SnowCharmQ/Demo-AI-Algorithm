from svm import *


def load_train_data(data_file):
    data, label = [], []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        label.append(float(lines[0]))
        index = 0
        tmp = []
        for i in range(1, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while int(li[0]) - 1 > index:
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data), np.mat(label).T


def load_test_data(test_file):
    data = []
    f = open(test_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        index = 0
        tmp = []
        for i in range(0, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while int(li[0]) - 1 > index:
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data)


def get_prediction(test_data, svm):
    m = np.shape(test_data)[0]
    prediction = []
    for i in range(m):
        predict = svm_predict(svm, test_data[i, :])
        prediction.append(str(int(np.sign(predict)[0, 0])))
    return prediction


def save_prediction(result_file, prediction):
    f = open(result_file, 'w')
    f.write("\n".join(prediction))
    f.close()
