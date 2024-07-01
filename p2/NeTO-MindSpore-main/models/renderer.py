import mindspore as ms
import mindspore.ops as ops
import numpy as np
import logging
import mcubes
from icecream import ic


# from .diff import gradient

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = ops.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = ops.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = ops.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = ops.meshgrid(xs, ys, zs, indexing='ij')
                pts = ops.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=-1)
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).asnumpy()
                val = ops.stop_gradient(val)
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    # u = ops.stop_gradient(u)
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.asnumpy()
    b_max_np = ops.stop_gradient(b_max_np)
    b_min_np = bound_min.asnumpy()
    b_min_np = ops.stop_gradient(b_min_np)

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / ops.sum(weights, -1, keepdim=True)  # 概率
    cdf = ops.cumsum(pdf, -1)
    cdf = ops.cat([ops.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = ops.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.broadcast_to(tuple(list(cdf.shape[:-1]) + [n_samples]))
    else:
        u = ops.rand(list(cdf.shape[:-1]) + [n_samples])
    # Invert CDF
    # u = u.contiguous()
    inds = ops.searchsorted(cdf, u, right=True)
    # inds = ops.stop_gradient(inds)
    below = ops.maximum(ops.zeros_like(inds - 1), inds - 1)
    above = ops.minimum((cdf.shape[-1] - 1) * ops.ones_like(inds), inds)
    inds_g = ops.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = ops.gather_elements(cdf.unsqueeze(1).broadcast_to(tuple(matched_shape)), 2, inds_g)
    bins_g = ops.gather_elements(bins.unsqueeze(1).broadcast_to(tuple(matched_shape)), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = ops.where(denom < 1e-5, ops.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 # color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.sdf_network.requires_grad = True
        self.deviation_network = deviation_network
        self.deviation_network.requires_grad = True
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = ops.cat([dists, ms.Tensor([sample_dist]).broadcast_to(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = ops.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = ops.cat([pts / dis_to_center, 1.0 / dis_to_center], axis=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].broadcast_to(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - ops.exp(-ops.Softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * ops.cumprod(ops.cat([ops.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample_occulsion(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = ops.norm(pts, ord=2, dim=-1, keepdim=False)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        dist = (next_z_vals - prev_z_vals)

        e_sdf = ops.exp(-mid_sdf * inv_s)
        density = (e_sdf / ops.pow(1 + e_sdf, 2)) * inv_s * dist

        z_samples = sample_pdf(z_vals, density, n_importance, det=True)
        z_samples = ops.stop_gradient(z_samples)

        return z_samples

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = ops.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = ops.cat([ops.zeros([batch_size, 1]), cos_val[:, :-1]], axis=-1)
        cos_val = ops.stack([prev_cos_val, cos_val], axis=-1)
        cos_val, _ = ops.min(cos_val, axis=-1, keepdims=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = ops.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = ops.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * ops.cumprod(
            ops.cat([ops.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True)
        z_samples = ops.stop_gradient(z_samples)
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = ops.cat([z_vals, new_z_vals], axis=-1)
        z_vals, index = ops.sort(z_vals, axis=-1)

        if not last:
            new_sdf = self.sdf_network(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = ops.cat([sdf, new_sdf], axis=-1)
            xx = ops.arange(batch_size)[:, None].broadcast_to((batch_size, n_samples + n_importance)).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        #
        # # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = ops.cat([dists, ms.Tensor([sample_dist]).broadcast_to(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].broadcast_to(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]

        gradients = ms.grad(sdf_network)(pts)

        inv_s = deviation_network(ops.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.broadcast_to((batch_size * n_samples, 1))

        true_cos = (dirs * gradients).sum(-1, keepdims=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(ops.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     ops.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = ops.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = ops.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = ops.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float()
        inside_sphere = ops.stop_gradient(inside_sphere)
        relax_inside_sphere = (pts_norm < 1.2).float()
        relax_inside_sphere = ops.stop_gradient(relax_inside_sphere)

        weights = alpha * ops.cumprod(ops.cat([ops.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(axis=-1, keepdims=True)

        # Eikonal loss
        gradient_error = (ops.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                   dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        gradients = gradients.reshape(batch_size, n_samples, 3)
        gradients = ((gradients * weights[:, :gradients.shape[1], None]) * inside_sphere[..., None]).sum(axis=1)

        inter_point = pts.reshape(batch_size, n_samples, 3)
        sdf = sdf.reshape(batch_size, n_samples)
        inter_point = ((inter_point * weights[:, :inter_point.shape[1], None]) * inside_sphere[..., None]).sum(axis=1)

        return {
            'point': inter_point,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients,
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'alpha': alpha,
            'weight_sum': weights_sum
        }

    def check_occlusion(self, check_sdf):
        occlusion_sign = check_sdf.sign()
        occlusion_sign = ops.stop_gradient(occlusion_sign)
        occlusion_sign[occlusion_sign == 0] = -1
        occlusion_sign[:, 1:-1] = occlusion_sign[:, 1:-1] + occlusion_sign[:, :-2]
        occlusion_sign = (occlusion_sign == 0).sum(1)
        occlusion_mask = (occlusion_sign == 2)
        occlusion_mask = ops.stop_gradient(occlusion_mask)
        occlusion_sign = ops.stop_gradient(occlusion_sign)
        return occlusion_sign, occlusion_mask

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1,
               background_rgb=None, cos_anneal_ratio=0.0, path='straight'):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = ops.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = ops.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (ops.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = ops.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = ops.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = ops.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / ops.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            pts = ops.stop_gradient(pts)
            sdf = self.sdf_network(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
            sdf = ops.stop_gradient(sdf)
            if path != 'straight':
                check_z_vals = z_vals
                check_sdf = sdf
                for i in range(self.up_sample_steps // 2):
                    new_check_z_vals = self.up_sample_occulsion(rays_o,
                                                                rays_d,
                                                                check_z_vals,
                                                                check_sdf,
                                                                self.n_importance // self.up_sample_steps,
                                                                64 * 2 ** i)
                    check_z_vals, check_sdf = self.cat_z_vals(rays_o,
                                                              rays_d,
                                                              check_z_vals,
                                                              new_check_z_vals,
                                                              check_sdf,
                                                              last=False)
                occlusion_sign, occlusion_mask = self.check_occlusion(check_sdf)
            else:
                occlusion_sign = None
                occlusion_mask = None

            for i in range(self.up_sample_steps):
                new_z_vals = self.up_sample(rays_o,
                                            rays_d,
                                            z_vals,
                                            sdf,
                                            self.n_importance // self.up_sample_steps,
                                            64 * 2 ** i)
                z_vals, sdf = self.cat_z_vals(rays_o,
                                              rays_d,
                                              z_vals,
                                              new_z_vals,
                                              sdf,
                                              last=(i + 1 == self.up_sample_steps))
            z_vals = ops.stop_gradient(z_vals)
            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = ops.cat([z_vals, z_vals_outside], axis=-1)
            z_vals_feed, _ = ops.sort(z_vals_feed, axis=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    cos_anneal_ratio=cos_anneal_ratio)
        weights = ret_fine['weights']
        weights_sum = weights.sum(axis=-1, keepdims=True)
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(axis=-1, keep_dims=True)

        return {
            'inter_point': ret_fine['point'],
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': ops.max(weights, axis=-1, keepdims=True)[0],
            'gradients': ret_fine['gradients'],
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'alpha': ret_fine['alpha'],
            'mid_z_vals': ret_fine['mid_z_vals'],
            'sdf': ret_fine['sdf'],
            'occlusion_sign': occlusion_sign,
            'occlusion_mask': occlusion_mask
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network(pts))
