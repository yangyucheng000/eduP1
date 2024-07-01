import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.initializer import Normal, Constant
import numpy as np
from models.embedder import get_embedder
from models.weight_norm import WeightNorm

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Cell):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Dense(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        lin.weight.set_data(ms.common.initializer.initializer(
                            ms.common.initializer.Normal(sigma=0.0001, mean=np.sqrt(np.pi) / np.sqrt(dims[l])),
                            lin.weight.shape, lin.weight.dtype))
                        lin.bias.set_data(ms.common.initializer.initializer(-bias, lin.bias.shape, lin.bias.dtype))
                    else:
                        lin.weight.set_data(ms.common.initializer.initializer(
                            ms.common.initializer.Normal(sigma=0.0001, mean=-np.sqrt(np.pi) / np.sqrt(dims[l])),
                            lin.weight.shape, lin.weight.dtype))
                        lin.bias.set_data(ms.common.initializer.initializer(bias, lin.bias.shape, lin.bias.dtype))
                elif multires > 0 and l == 0:
                    lin.bias.set_data(ms.common.initializer.initializer(0.0, lin.bias.shape, lin.bias.dtype))
                    lin.weight.set_data(ms.common.initializer.initializer(0.0, lin.weight.shape, lin.weight.dtype))
                    mean = ms.Tensor(0.0, ms.float32)
                    stddev = ms.Tensor(np.sqrt(2) / np.sqrt(out_dim), ms.float32)
                    shape = (lin.weight[:, :3].shape[0], lin.weight[:, :3].shape[1])
                    lin.weight[:, :3] = ms.ops.normal(shape, mean, stddev)
                elif multires > 0 and l in self.skip_in:
                    lin.bias.set_data(ms.common.initializer.initializer(0.0, lin.bias.shape, lin.bias.dtype))
                    mean = ms.Tensor(0.0, ms.float32)
                    stddev = ms.Tensor(np.sqrt(2) / np.sqrt(out_dim), ms.float32)
                    shape = (lin.weight[:, :-(dims[0] - 3)].shape[0], lin.weight[:, :-(dims[0] - 3)].shape[1])
                    lin.weight[:, :-(dims[0] - 3)] = ms.ops.normal(shape, mean, stddev)
                    lin.weight[:, -(dims[0] - 3):] = ms.ops.zeros_like(lin.weight[:, -(dims[0] - 3):])
                else:
                    lin.weight.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Normal(sigma=np.sqrt(2) / np.sqrt(out_dim), mean=0.0),
                        lin.weight.shape, lin.weight.dtype))
                    lin.bias.set_data(ms.common.initializer.initializer(0.0, lin.bias.shape, lin.bias.dtype))

            setattr(self, "lin" + str(l), lin)

    def batch_norm(self, x):
        mean = ops.mean(x, axis=1, keep_dims=True)
        variance = ops.var(x, axis=1, ddof=2, keepdims=True)
        x = (x - mean) / ops.sqrt(variance + 1e-5)
        x = 0.1 * x
        return x

    def construct(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = ops.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = ops.relu(x)

        output = ops.cat([x[:, :1] / self.scale], axis=-1)
        return output

    def sdf(self, x):
        result = self.construct(x)
        value = result[:, :1]
        return value

    def sdf_hidden_appearance(self, x):
        return self.construct(x)

    def gradient(self, x):
        grad_op = ops.GradOperation()
        gradient_function = grad_op(self.sdf)
        gradients = gradient_function(x)
        return gradients.unsqueeze(1)

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Cell):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.CellList(
            [nn.Dense(self.input_ch, W)] +
            [nn.Dense(W, W) if i not in self.skips else nn.Dense(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.CellList([nn.Dense(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Dense(W, W)
            self.alpha_linear = nn.Dense(W, 1)
            self.rgb_linear = nn.Dense(W // 2, 3)
        else:
            self.output_linear = nn.Dense(W, output_ch)

    def construct(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = ops.relu(h)
            if i in self.skips:
                h = ops.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = ops.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = ops.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

class SingleVarianceNetwork(nn.Cell):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = ms.Parameter(ms.Tensor(init_val, ms.float32), requires_grad=True)

    def construct(self, x):
        output = ops.ones([len(x), 1]) * ops.exp(self.variance * 10.0)
        return output


class SampleNetwork(nn.Cell):
    '''
    Represent the intersection (sample) point as differentiable function of the implicit geometry and camera parameters.
    See equation 3 in the paper for more details.
    '''

    def construct(self, surface_output, surface_sdf_values, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        # t -> t(theta)
        surface_ray_dirs_0 = surface_ray_dirs
        surface_points_dot = ops.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)
        surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / surface_points_dot

        # t(theta) -> x(theta,c,v)
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

        return surface_points_theta_c_v



