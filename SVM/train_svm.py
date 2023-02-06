from utils import *

C = 0.6
toler = 0.001
max_iter = 500
dataset, labels = load_train_data("train_data")
svm_model = svm_training(dataset, labels, C, toler, max_iter)
accuracy = cal_accuracy(svm_model, dataset, labels)
print("The accuracy for training is %.3f%%" % (accuracy * 100))
save_svm_model(svm_model, "model.pkl")
