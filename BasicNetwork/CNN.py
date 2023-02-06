import numpy as np


def get_patch(input_array, i, j, filter_width, filter_height, stride):  # 获得待卷积的区域
    start_i = i * stride  # 通过乘上步长得到初始用于遍历的行值
    start_j = j * stride  # 通过乘上步长得到初始用于遍历的列值
    if input_array.ndim == 2:  # 输入的矩阵是二维的情况
        return input_array[
               start_i:(start_i + filter_height),
               start_j:(start_j + filter_width)]
    elif input_array.ndim == 3:  # 输入矩阵是三维的情况下完整保留深度层
        return input_array[:,
               start_i:(start_i + filter_height),
               start_j:(start_j + filter_width)]


def get_max_index(array):  # 在pooling时得到一个2D区域的最大值所在的索引
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


class ConvLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width,
                 filter_height, filter_number, zero_padding, stride, activator, learning_rate):
        self.delta_array = None
        self.input_width = int(input_width)  # 输入宽度
        self.input_height = int(input_height)  # 输入高度
        self.channel_number = int(channel_number)  # 输入的channel数
        self.filter_width = int(filter_width)  # 卷积核的宽度
        self.filter_height = int(filter_height)  # 卷积核的高度
        self.filter_number = int(filter_number)  # 卷积核的数目
        self.zero_padding = int(zero_padding)  # 零填充的大小
        self.stride = int(stride)  # 步长
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding, stride)  # 计算输出的宽度
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding, stride)  # 计算输出的高度
        dim = (self.filter_number, self.output_height, self.output_width)
        self.output_array = np.zeros(dim)  # 输出矩阵
        self.filters = []
        for i in range(filter_number):  # 添加filter到列表中
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))
        self.activator = activator  # 激活函数
        self.leaning_rate = learning_rate  # 学习率

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):  # 计算输出尺寸 = (W\H - 2 * F) / S + 1
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)  # 零填充输入矩阵
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())  # 卷积操作
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        """
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一项的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self, sensitivity_array, activator):  # 计算传递到上一层的sensitivity map
        # sensitivity_array: 本层的sensitivity map
        # activator: 上一层的激活函数
        expanded_array = self.expand_sensitivity_map(sensitivity_array)  # 处理卷积步长，对原始的sensitivity map进行扩展
        """
        full卷积，对sensitivity map进行zero padding
        虽然原始输入的zero padding单元也会获得残差
        但这个残差不需要继续向上传递一次就不计算了
        """
        expanded_width = expanded_array.shape[2]
        zp = int((self.input_width + self.filter_width - 1 - expanded_width) / 2)
        padded_array = padding(expanded_array, zp)
        self.delta_array = self.create_delta_array()  # 初始化delta_array用于保存传递到上一层的sensitivity map
        """
        对于具有多个filter的卷积层来说，最终传递到上一层的sensitivity map
        相当于所有的filter的sensitivity map之和
        """
        for f in range(self.filter_number):
            filter = self.filters[f]
            flipped_weights = np.array(map(lambda i: np.rot90(i, 2), filter.get_weights()))
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)  # 将计算结果于激活函数的偏导数做element-wise乘法操作
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):  # 将步长为S的sensitivity map还原为步长为1的sensitivity map
        depth = sensitivity_array.shape[0]  # 确定扩展后sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_height, expanded_width))  # 构建新的sensitivity map
        for i in range(self.output_height):  # 从原始的sensitivity map拷贝误差值
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    def bp_gradient(self, sensitivity_map):  # 计算梯度
        expanded_array = self.expand_sensitivity_map(sensitivity_map)
        for f in range(self.filter_number):  # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            filter.bias_grad = expanded_array[f].sum()

    def update(self):  # 根据梯度下降更新权重
        for filter in self.filters:
            filter.update(self.leaning_rate)


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(input_array, kernel_array, output_array, stride, bias):
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_patch(input_array, i, j,
                                            kernel_width, kernel_height, stride)
                                  * kernel_array).sum() + bias


def padding(input_array, zp):  # 零填充的操作
    if zp == 0:  # 不进行操作时直接返回原矩阵
        return input_array
    if input_array.ndim == 3:
        input_width = input_array.shape[2]
        input_height = input_array.shape[1]
        input_depth = input_array.shape[0]
        padded_array = np.zeros((input_depth, input_height + 2 * zp,
                                 input_width + 2 * zp))
        padded_array[:, zp:zp + input_height, zp:zp + input_width] \
            = input_array  # 输入为3D时需考虑输入的矩阵深度，直接copy即可
        return padded_array
    elif input_array.ndim == 2:
        input_width = input_array.shape[1]
        input_height = input_array.shape[0]
        padded_array = np.zeros((
            input_height + 2 * zp,
            input_width + 2 * zp
        ))
        padded_array[zp:zp + input_height,
        zp: zp + input_width] = input_array
        return padded_array


class Filter(object):  # 用于卷积操作的神经元
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))  # 初始化权重矩阵
        self.bias = 0  # 偏置
        self.weights_grad = np.zeros(self.weights.shape)  # 初始化权重梯度矩阵
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights: \n%s\n bias:\n%s' % (repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class MaxPoolingLayer:
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, stride):
        self.delta_array = None
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width - filter_width) / self.stride + 1
        self.output_height = (input_height - filter_height) / self.stride + 1
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_height))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width, self.filter_height,
                                  self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(input_array[d], i, j,
                                            self.filter_width, self.filter_height, self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k, j * self.stride + l] = \
                        sensitivity_array[d, i, j]
