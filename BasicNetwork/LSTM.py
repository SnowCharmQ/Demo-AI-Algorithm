import numpy as np
from activators import SigmoidActivator, TanhActivator


class LSTM:
    def __init__(self, input_width, state_width, learning_rate):
        self.Wfh_grad = None
        self.Wfx_grad = None
        self.bf_grad = None
        self.Wih_grad = None
        self.Wix_grad = None
        self.bi_grad = None
        self.Woh_grad = None
        self.Wox_grad = None
        self.bo_grad = None
        self.Wch_grad = None
        self.Wcx_grad = None
        self.bc_grad = None
        self.delta_h_list = None
        self.delta_o_list = None
        self.delta_i_list = None
        self.delta_f_list = None
        self.delta_ct_list = None
        self.input_width = input_width
        self.state_width = state_width
        self.learning_rate = learning_rate
        self.gate_activator = SigmoidActivator()
        self.output_activator = TanhActivator()
        self.times = 0  # 初始化时间为0
        self.c_list = self.init_state_vec()  # 各个时刻的单元状态向量c
        self.h_list = self.init_state_vec()  # 各个时刻的输出向量h
        self.f_list = self.init_state_vec()  # 各个时刻的遗忘门f
        self.i_list = self.init_state_vec()  # 各个时刻的输入门i
        self.o_list = self.init_state_vec()  # 各个时刻的输出门o
        self.ct_list = self.init_state_vec()  # 各个时刻的即时状态c~
        self.Wfh, self.Wfx, self.bf = self.init_weight_mat()  # 遗忘门的权重矩阵Wfh，Wfx，偏置项bf
        self.Wih, self.Wix, self.bi = self.init_weight_mat()  # 输入门的权重矩阵Wih，Wix，偏置项bi
        self.Woh, self.Wox, self.bo = self.init_weight_mat()  # 输出门的权重矩阵Woh，Wox，偏置项bo
        self.Wch, self.Wcx, self.bc = self.init_weight_mat()  # 单元状态的权重矩阵Wch，Wcx，偏置项bc

    def init_state_vec(self):
        """
        初始化保存状态的向量
        """
        state_vec_list = [np.zeros((self.state_width, 1))]
        return state_vec_list

    def init_weight_mat(self):
        """
        初始化权重矩阵
        """
        Wh = np.random.uniform(-1e-4, 1e-4, (self.state_width, self.state_width))
        Wx = np.random.uniform(-1e-4, 1e-4, (self.state_width, self.state_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    def forward(self, x):
        self.times += 1
        fg = self.calc_gate(x, self.Wfx, self.Wfh, self.bf, self.gate_activator)
        ig = self.calc_gate(x, self.Wix, self.Wih, self.bi, self.gate_activator)
        og = self.calc_gate(x, self.Wox, self.Woh, self.bo, self.gate_activator)
        ct = self.calc_gate(x, self.Wcx, self.Wch, self.bc, self.output_activator)
        self.ct_list.append(ct)
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)

    def calc_gate(self, x, Wx, Wh, b, activator):
        h = self.h_list[self.times - 1]
        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)
        return gate

    def backward(self, x, delta_h, activator):
        self.calc_delta(delta_h, activator)
        self.calc_gradient(x)

    def calc_delta(self, delta_h, activator):
        self.delta_h_list = self.init_delta()  # 输出误差项
        self.delta_o_list = self.init_delta()  # 输出门误差项
        self.delta_i_list = self.init_delta()  # 输入门误差项
        self.delta_f_list = self.init_delta()  # 遗忘门误差项
        self.delta_ct_list = self.init_delta()  # 即时输出误差项
        self.delta_h_list[-1] = delta_h
        for k in range(self.times, 0, -1):
            self.calc_delta_k(k)  # 计算每个时刻的误差项

    def init_delta(self):
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros((self.state_width, 1)))
        return delta_list

    def calc_delta_k(self, k):
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_pre = self.c_list[k - 1]
        tanh_c = self.output_activator.forward(c)
        delta_k = self.delta_h_list[k]
        delta_o = (delta_k * tanh_c * self.gate_activator.backward(og))
        delta_f = (delta_k * og * (1 - tanh_c * tanh_c) * c_pre * self.gate_activator.backward(fg))
        delta_i = (delta_k * og * (1 - tanh_c * tanh_c) * ct * self.gate_activator.backward(ig))
        delta_ct = (delta_k * og * (1 - tanh_c * tanh_c) * ct * self.output_activator.backward(ct))
        delta_h_prev = (np.dot(delta_o.transpose(), self.Woh) + np.dot(delta_i.transpose(), self.Wih) +
                        np.dot(delta_f.transpose(), self.Wfh) + np.dot(delta_ct.transpose(), self.Wch)).transpose()
        self.delta_h_list[k - 1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def calc_gradient(self, x):
        # 初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
            self.init_weight_gradient_mat())
        # 初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad, self.Wix_grad, self.bi_grad = (
            self.init_weight_gradient_mat())
        # 初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad, self.Wox_grad, self.bo_grad = (
            self.init_weight_gradient_mat())
        # 初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad, self.Wcx_grad, self.bc_grad = (
            self.init_weight_gradient_mat())
        # 计算对上一次输出h的权重梯度
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (Wfh_grad, bf_grad,
             Wih_grad, bi_grad,
             Woh_grad, bo_grad,
             Wch_grad, bc_grad) = (
                self.calc_gradient_t(t))
            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wch_grad
            self.bc_grad += bc_grad
            print('-----%d-----' % t)
            print(Wfh_grad)
            print(self.Wfh_grad)
        # 计算对本次输入x的权重梯度
        xt = x.transpose()
        self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
        self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
        self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
        self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)

    def init_weight_gradient_mat(self):
        Wh_grad = np.zeros((self.state_width,
                            self.state_width))
        Wx_grad = np.zeros((self.state_width,
                            self.input_width))
        b_grad = np.zeros((self.state_width, 1))
        return Wh_grad, Wx_grad, b_grad

    def calc_gradient_t(self, t):
        """
        计算每个时刻t权重的梯度
        """
        h_prev = self.h_list[t - 1].transpose()
        Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
        bf_grad = self.delta_f_list[t]
        Wih_grad = np.dot(self.delta_i_list[t], h_prev)
        bi_grad = self.delta_f_list[t]
        Woh_grad = np.dot(self.delta_o_list[t], h_prev)
        bo_grad = self.delta_f_list[t]
        Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
        bc_grad = self.delta_ct_list[t]
        return Wfh_grad, bf_grad, Wih_grad, bi_grad, Woh_grad, bo_grad, Wch_grad, bc_grad
