import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

def l2normalize(v, eps=1e-12):
    return v / (v.norm()+eps)


class SpectralNorm(nn.Cell):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__(auto_prefix=False)
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_orig")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.set_data(l2normalize(ops.mv(ops.t(w.view(height, -1)), u.data)))
            u.set_data(l2normalize(ops.mv(w.view(height, -1), v.data)))
        sigma = ops.tensor_dot(u, ops.mv(w.view(height, -1), v), axes=1)
        setattr(self.module, self.name, w/sigma.expand_as(w))
        # self.module.insert_param_to_cell(self.name, w/sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_orig")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        # w = self.module.get_parameters()[self.name]
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).shape[1]

        u = mindspore.Parameter(ops.randn((height)), requires_grad=False)
        v = mindspore.Parameter(ops.randn((width)), requires_grad=False)
        u.set_data(l2normalize(u.data))
        v.set_data(l2normalize(v.data))
        w_orig = mindspore.Parameter(w.data)

        # par_dict[key] = par_dict.pop(name)
        # cover ori parameter, and change the parameter name
        self.module.insert_param_to_cell(self.name + "_u", u)
        self.module.insert_param_to_cell(self.name + "_v", v)
        self.module.insert_param_to_cell(self.name + "_orig", w_orig)


    def construct(self, *args):
        self._update_u_v()
        return self.module.construct(*args)
