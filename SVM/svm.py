import numpy as np
import pickle as pickle


class SVM:
    def __init__(self, dataset, labels, C, toler, kernel_option):
        self.train_x = dataset
        self.train_y = labels
        self.C = C
        self.toler = toler
        self.n_samples = np.shape(dataset)[0]
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))
        self.kernel_opt = kernel_option
        self.kernel_mat = cal_kernel(self.train_x, self.kernel_opt)


def cal_kernel_value(train_x, train_x_i, kernel_option):
    kernel_type = kernel_option[0]
    m = np.shape(train_x)[0]

    kernel_value = np.mat(np.zeros((m, 1)))

    if kernel_type == 'rbf':
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(m):
            diff = train_x[i, :] - train_x_i
            kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        kernel_value = train_x * train_x_i.T
    return kernel_value


def cal_kernel(train_x, kernel_option):
    m = np.shape(train_x)[0]
    kernel_matrix = np.mat(np.zeros((m, m)))
    for i in range(m):
        kernel_matrix[:, i] = cal_kernel_value(train_x, train_x[i, :], kernel_option)
    return kernel_matrix


def cal_error(svm, alpha_k):
    output_k = float(np.multiply(svm.alphas, svm.train_y).T * svm.kernel_mat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


def update_error_tmp(svm, alpha_k):
    error = cal_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]


def select_second_sample_j(svm, alpha_i, error_i):
    svm.error_tmp[alpha_i] = [1, error_i]
    candidate_alpha_list = np.nonzero(svm.error_tmp[:, 0].A)[0]
    max_step = 0
    alpha_j = 0
    error_j = 0
    if len(candidate_alpha_list) > 1:
        for alpha_k in candidate_alpha_list:
            if alpha_k == alpha_i:
                continue
            error_k = cal_error(svm, alpha_k)
            if abs(error_k - error_i) > max_step:
                max_step = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.n_samples))
        error_j = cal_error(svm, alpha_j)
    return alpha_j, error_j


def choose_and_update(svm, alpha_i):
    error_i = cal_error(svm, alpha_i)
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or \
            (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):
        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            low = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            high = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            low = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            high = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if low == high:
            return 0
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] \
              - svm.kernel_mat[alpha_j, alpha_j]
        if eta >= 0:
            return 0
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta
        if svm.alphas[alpha_j] > high:
            svm.alphas[alpha_j] = high
        if svm.alphas[alpha_j] < low:
            svm.alphas[alpha_j] = low
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error_tmp(svm, alpha_j)
            return 0
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                               * (alpha_j_old - svm.alphas[alpha_j])
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
             * svm.kernel_mat[alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
             * svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
             * svm.kernel_mat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
             * svm.kernel_mat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0
        update_error_tmp(svm, alpha_j)
        update_error_tmp(svm, alpha_i)
        return 1
    else:
        return 0


def svm_training(train_x, train_y, C, toler, max_iter, kernel_option=('rbf', 0.431029)):
    svm = SVM(train_x, train_y, C, toler, kernel_option)
    entire_set = True
    alpha_pairs_changed = 0
    iteration = 0
    while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entire_set):
        print("Iter:", iteration)
        alpha_pairs_changed = 0
        if entire_set:
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:
            bound_samples = []
            for i in range(svm.n_samples):
                if 0 < svm.alphas[i, 0] < svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True

    return svm


def svm_predict(svm, test_sample_x):
    kernel_value = cal_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
    predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
    return predict


def cal_accuracy(svm, test_x, test_y):
    n_samples = np.shape(test_x)[0]
    correct = 0.0
    for i in range(n_samples):
        predict = svm_predict(svm, test_x[i, :])
        if np.sign(predict) == np.sign(test_y[i]):
            correct += 1
    accuracy = correct / n_samples
    return accuracy


def save_svm_model(svm_model, model_file):
    with open(model_file, 'wb') as f:
        pickle.dump(svm_model, f)


def load_svm_model(svm_model_file):
    with open(svm_model_file, 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model
