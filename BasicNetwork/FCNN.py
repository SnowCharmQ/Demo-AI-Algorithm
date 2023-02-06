import random
from functools import reduce
import numpy as np


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index  # 节点所属的层的编号
        self.node_index = node_index  # 节点的编号
        self.downstream = []  # 用于反向传播算法
        self.upstream = []  # 作为上层用于反向传播算法
        self.output = 0  # 输出
        self.delta = 0  # 权值

    def set_output(self, output):
        self.output = output  # 设置节点的输出值

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)  # 添加一个到下游的链接

    def calc_output(self):  # 计算节点的输出
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = np.sigmoid(output)

    def calc_hidden_layer_delta(self):  # 当节点属于隐藏层时，计算delta
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstrem_node.delta * conn.weight,
                                  self.downstream, 0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):  # 当节点属于输出层时，计算delta
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):  # 打印节点的信息
        node_str = '%u-%u: output: %f delta: %f' % \
                   (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(Node):  # 实现一个输出恒为1的节点（计算偏置项时需要）
    def __init__(self, layer_index, node_index):
        super().__init__(layer_index, node_index)
        self.output = 1

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index  # 层编号
        self.nodes = []  # 将节点添加到该层中
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):  # 设置层的输出，当层时输入层时使用
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node  # 连接的上游节点
        self.downstream_node = downstream_node  # 连接的下游节点
        self.weight = random.uniform(-0.1, 0.1)  # 权重初始化为一个很小的随机数
        self.gradient = 0.0

    def calc_gradient(self):  # 计算梯度
        self.gradient = self.downstream_node.delta * self.upstream_node.ouput

    def get_gradient(self):  # 获取当前的梯度
        return self.gradient

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % \
               (self.upstream_node.layer_index,
                self.upstream_node.node_index,
                self.downstream_node.layer_index,
                self.downstream_node.node_index,
                self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):  # layers是二维数组，描述神经网络的每层节点数
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)  # 得到层的数量
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))  # 向layers列表中添加所需的各层对象
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)  # 创建各层之间的连接
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:  # 添加各层之间的连接
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.node_append_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        # labels: 数组，训练样本标签。每个元素是一个样本的标签
        # data_set: 二位数组，训练样本特征。每个元素是一个样本的特征
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):  # 用一个样本训练网络
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def update_weight(self, rate):  # 更新每个连接权重
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_delta(self, label):  # 计算每个节点的delta
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):  # 获得网络在一个样本下每个连接上的梯度
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()
