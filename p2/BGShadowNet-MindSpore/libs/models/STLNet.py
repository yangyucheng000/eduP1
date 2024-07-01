import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class ConvBNReLU(nn.Cell):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                    has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__(auto_prefix=True)
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, pad_mode='pad', dilation=dilation, has_bias=False, group=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, pad_mode='pad', dilation=dilation, has_bias=False, group=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out, momentum=0.1)
        if self.has_relu:
            self.relu = nn.ReLU()       # inplace=True

    def construct(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x



class QCO_1d(nn.Cell):
    def __init__(self, level_num):
        super(QCO_1d, self).__init__(auto_prefix=True)
        self.conv1 = nn.SequentialCell(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU())       # inplace=True
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.SequentialCell(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'), nn.LeakyReLU())
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape
        x_ave = ops.adaptive_avg_pool2d(x, (1, 1))
        cos_sim = (ops.L2Normalize(axis=1,epsilon=1e-12)(x_ave) * ops.L2Normalize(axis=1,epsilon=1e-12)(x)).sum(1)          # TODO: waiting for check
        cos_sim = cos_sim.view(N, -1)
        cos_sim_min, _ = cos_sim.min(-1, return_indices=True)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(-1, return_indices=True)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = ops.arange(self.level_num).float()
        q_levels = q_levels.expand_as(np.ones((N, self.level_num)))
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        quant = 1 - ops.abs(q_levels - cos_sim)
        quant = quant * (quant > (1 - q_levels_inter))
        sta = quant.sum(1)
        sta = sta / (sta.sum(-1).unsqueeze(-1))
        sta = sta.unsqueeze(1)
        sta = ops.cat([q_levels, sta], axis=1)
        sta = self.f1(sta)
        sta = self.f2(sta)
        x_ave = x_ave.squeeze(-1).squeeze(-1)
        x_ave = x_ave.expand_as(np.ones((self.level_num, N, C))).permute(1, 2, 0)
        sta = ops.cat([sta, x_ave], axis=1)
        sta = self.out(sta)
        return sta, quant


class QCO_2d(nn.Cell):
    def __init__(self, scale, level_num):
        super(QCO_2d, self).__init__(auto_prefix=True)
        self.f1 = nn.SequentialCell(ConvBNReLU(3, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='2d'), nn.LeakyReLU())
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='2d')
        self.out = nn.SequentialCell(ConvBNReLU(256+128, 128, 1, 1, 0, has_bn=True, has_relu=True, mode='2d'),
                                     ConvBNReLU(128, 128, 1, 1, 0, has_bn=True, has_relu=False, mode='2d'))
        self.scale = scale
        self.level_num = level_num
    def construct(self, x):
        N1, C1, H1, W1 = x.shape
        if H1 // self.level_num != 0 or W1 // self.level_num != 0:
            x = ops.adaptive_avg_pool2d(x, ((int(H1/self.level_num)*self.level_num), int(W1/self.level_num)*self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = ops.adaptive_avg_pool2d(x, (self.scale, self.scale))
        x_ave_up = ops.adaptive_avg_pool2d(x_ave, (H, W))
        cos_sim = (ops.L2Normalize(axis=1)(x_ave_up) * ops.L2Normalize(axis=1)(x)).sum(1)
        cos_sim = cos_sim.unsqueeze(1)
        cos_sim = cos_sim.reshape(N, 1, self.scale, self.size_h, self.scale, self.size_w)
        cos_sim = cos_sim.permute(0, 1, 2, 4, 3, 5)
        cos_sim = cos_sim.reshape(N, 1, int(self.scale*self.scale), int(self.size_h*self.size_w))
        cos_sim = cos_sim.permute(0, 1, 3, 2)
        cos_sim = cos_sim.squeeze(1)
        cos_sim_min, _ = cos_sim.min(1, return_indices=True)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(1, return_indices=True)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = ops.arange(self.level_num).float()
        q_levels = q_levels.expand_as(np.ones((N, self.scale*self.scale, self.level_num)))
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        q_levels = q_levels.unsqueeze(1)
        quant = 1 - ops.abs(q_levels - cos_sim)
        quant = quant * (quant > (1 - q_levels_inter))
        quant = quant.view([N, self.size_h, self.size_w, self.scale*self.scale, self.level_num])
        quant = quant.permute(0, -2, -1, 1, 2)
        quant = quant.contiguous().view(N, -1, self.size_h, self.size_w)
        quant = ops.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant = quant.view(N, self.scale*self.scale, self.level_num, self.size_h+1, self.size_w+1)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3)
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2)
        quant = quant_left * quant_right
        sta = quant.sum(-1).sum(-1)
        sta = sta / (sta.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)
        sta = sta.unsqueeze(1)
        q_levels = q_levels.expand_as(np.ones((self.level_num, N, 1, self.scale*self.scale, self.level_num)))
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4)
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3)
        sta = ops.cat([q_levels_h, q_levels_w, sta], axis=1)
        sta = sta.view(N, 3, self.scale * self.scale, -1)
        sta = self.f1(sta)
        sta = self.f2(sta)
        x_ave = x_ave.view(N, C, -1)
        x_ave = x_ave.expand_as(np.ones((self.level_num*self.level_num, N, C, self.scale*self.scale)))
        x_ave = x_ave.permute(1, 2, 3, 0)
        sta = ops.cat([x_ave, sta], axis=1)
        sta = self.out(sta)
        sta = sta.mean(-1)
        sta = sta.view(N, sta.shape[1], self.scale, self.scale)
        return sta


class TEM(nn.Cell):
    def __init__(self, level_num):
        super(TEM, self).__init__(auto_prefix=True)
        self.level_num = level_num
        self.qco = QCO_1d(level_num)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')
    def construct(self, x):
        tem_dict = {}
        N, C, H, W = x.shape
        sta, quant = self.qco(x)
        tem_dict['qco_sta'] = sta
        tem_dict['qco_quant'] = quant
        k = self.k(sta)
        tem_dict['k'] = k
        q = self.q(sta)
        tem_dict['q'] = q
        v = self.v(sta)
        tem_dict['v'] = v
        k = k.permute(0, 2, 1)
        w = ops.bmm(k, q)
        w = ops.softmax(w, axis=-1)
        v = v.permute(0, 2, 1)
        f = ops.bmm(w, v)
        f = f.permute(0, 2, 1)
        f = self.out(f)
        tem_dict['f'] = f
        quant = quant.permute(0, 2, 1)
        out = ops.bmm(f, quant)
        out = out.view(N, 256, H, W)
        tem_dict['out'] = out
        return out, tem_dict



class PTFEM(nn.Cell):
    def __init__(self):
        super(PTFEM, self).__init__(auto_prefix=True)
        self.conv = ConvBNReLU(512, 256, 1, 1, 0, has_bn=False, has_relu=False)
        self.qco_1 = QCO_2d(1, 8)
        self.qco_2 = QCO_2d(2, 8)
        self.qco_3 = QCO_2d(4, 8)
        self.qco_6 = QCO_2d(8, 8)
        self.out = ConvBNReLU(256, 256, 1, 1, 0)
    def construct(self, x):
        H, W = x.shape[2:]
        x = self.conv(x)
        sta_1 = self.qco_1(x)
        sta_2 = self.qco_2(x)
        sta_3 = self.qco_3(x)
        sta_6 = self.qco_6(x)
        N, C = sta_1.shape[:2]
        sta_1 = sta_1.view(N, C, 1, 1)

        sta_2 = sta_2.view(N, C, 2, 2)
        sta_1 = ops.interpolate(sta_1, size=(H, W), mode='bilinear', align_corners=True)
        sta_2 = ops.interpolate(sta_2, size=(H, W), mode='bilinear', align_corners=True)
        x = ops.cat([sta_1, sta_2], axis=1)
        x = self.out(x)
        return x



class STL(nn.Cell):
    def __init__(self, in_channel):
        super().__init__(auto_prefix=True)
        self.conv_start = ConvBNReLU(in_channel, 256, 1, 1, 0)
        self.tem = TEM(128)
        self.ptfem = PTFEM()
        self.conv_end = ConvBNReLU(512, 192, 1, 1, 0)
    def construct(self, x):
        stl_dict = {}
        x = self.conv_start(x)
        stl_dict["conv_start"] = x
        x_tem, tem_dict = self.tem(x)
        stl_dict.update(tem_dict)
        stl_dict["tem"] = x_tem
        x = ops.cat([x_tem, x], axis=1)
        x = self.conv_end(x)
        stl_dict["conv_end"] = x
        return x, stl_dict
