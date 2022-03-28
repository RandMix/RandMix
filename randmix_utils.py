import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.stats import beta
import cv2
from math import ceil
import gco

#FMix

def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)

    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power

    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space

    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, b, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha

    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha)
    else:
        lam = beta.rvs(alpha, b)

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.

    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, b, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda

    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, b, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """

    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1-mask)
    return x1+x2, index, lam


class FMixBase:
    r""" FMix augmentation

        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
    """

    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__()
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft
        self.index = None
        self.lam = None

    def __call__(self, x):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

# saliency mix
def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# co-mixup

criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()

def to_one_hot(inp, num_classes, device='cuda'):
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)

    return y_onehot


def distance(z, dist_type='l2'):
    '''Return distance matrix between vectors'''
    with torch.no_grad():
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        if dist_type[:2] == 'l2':
            A_dist = (diff**2).sum(-1)
            if dist_type == 'l2':
                A_dist = torch.sqrt(A_dist)
            elif dist_type == 'l22':
                pass
        elif dist_type == 'l1':
            A_dist = diff.abs().sum(-1)
        elif dist_type == 'linf':
            A_dist = diff.abs().max(-1)[0]
        else:
            return None
    return A_dist


def to_onehot(idx, n_input, device='cuda'):
    '''Return one-hot vector'''
    idx_onehot = torch.zeros((idx.shape[0], n_input), dtype=torch.float32, device=device)
    idx_onehot.scatter_(1, idx.unsqueeze(1), 1)
    return idx_onehot


def random_initialize(n_input, n_output, height, width):
    '''Initialization of labeling for Co-Mixup'''
    return np.random.randint(0, n_input, (n_output, width, height))


def obj_fn(cost_matrix, mask_onehot, beta, gamma):
    '''Calculate objective without thresholding'''
    n_output, height, width, n_input = mask_onehot.shape
    mask_idx_sum = mask_onehot.reshape(n_output, height * width, n_input).sum(1)

    loss = 0
    loss += torch.sum(cost_matrix.permute(1, 2, 0).unsqueeze(0) * mask_onehot)
    loss += beta / 2 * (((mask_onehot[:, :-1, :, :] - mask_onehot[:, 1:, :, :])**2).sum() +
                        ((mask_onehot[:, :, :-1, :] - mask_onehot[:, :, 1:, :])**2).sum())
    loss += gamma * (torch.sum(mask_idx_sum.sum(0)**2) - torch.sum(mask_idx_sum**2))

    return loss


def obj_fn_thres(cost_matrix, mask_onehot, beta, gamma, thres):
    '''Calculate objective with thresholding'''
    n_output, height, width, n_input = mask_onehot.shape
    mask_idx_sum = mask_onehot.reshape(n_output, height * width, n_input).sum(1)

    loss = 0
    loss += torch.sum(cost_matrix.permute(1, 2, 0).unsqueeze(0) * mask_onehot)
    loss += beta / 2 * (((mask_onehot[:, :-1, :, :] - mask_onehot[:, 1:, :, :])**2).sum() +
                        ((mask_onehot[:, :, :-1, :] - mask_onehot[:, :, 1:, :])**2).sum())

    penalty = mask_idx_sum.sum(0, keepdim=True) - mask_idx_sum
    modular_penalty = (penalty > thres).float() * penalty
    loss += gamma * torch.sum(modular_penalty * mask_idx_sum)

    return loss


def mix_input(mask_onehot, input_sp, target_reweighted, sc=None):
    ''' Mix inputs and one-hot labels based on labeling (mask_onehot)'''
    n_output, height, width, n_input = mask_onehot.shape
    _, n_class = target_reweighted.shape

    mask_onehot_im = F.interpolate(mask_onehot.permute(0, 3, 1, 2),
                                   size=input_sp.shape[-1],
                                   mode='nearest')
    output = torch.sum(mask_onehot_im.unsqueeze(2) * input_sp.unsqueeze(0), dim=1)

    if sc is None:
        mask_target = torch.matmul(mask_onehot, target_reweighted)
    else:
        weighted_mask = mask_onehot * sc.permute(1, 2, 0).unsqueeze(0)
        mask_target = torch.matmul(weighted_mask, target_reweighted)

    target = mask_target.reshape(n_output, height * width, n_class).sum(-2)
    target /= target.sum(-1, keepdim=True)

    return output, target


def resolve_label(assigned_label_total, device='cuda'):
    '''A post-processing for resolving identical outputs'''
    n_output, n_input = assigned_label_total.shape
    add_cost = torch.zeros_like(assigned_label_total)

    dist = torch.min(
        (assigned_label_total.unsqueeze(1) - assigned_label_total.unsqueeze(0)).abs().sum(-1),
        torch.tensor(1.0, device=device))
    coincide = torch.triu(1. - dist, diagonal=1)

    for i1, i2 in coincide.nonzero():
        nonzeros = assigned_label_total[i1].nonzero()
        if len(nonzeros) == 1:
            continue
        else:
            add_cost[i1][nonzeros[0]] = 1.
            add_cost[i2][nonzeros[1]] = 1.

    return add_cost


def graphcut_multi(cost, beta=1, algorithm='swap', n_label=0, add_idx=None):
    '''find optimal labeling using Graph-Cut algorithm'''
    height, width, n_input = cost.shape

    unary = np.ascontiguousarray(cost)
    pairwise = (np.ones(shape=(n_input, n_input), dtype=np.float32) -
                np.eye(n_input, dtype=np.float32))
    if n_label == 2:
        pairwise[-1, :-1][add_idx] = 0.25
        pairwise[:-1, -1][add_idx] = 0.25
    elif n_label == 3:
        pairwise[-3:, :-3][:, add_idx] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])
        pairwise[:-3, -3:][add_idx, :] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])

    cost_v = beta * np.ones(shape=[height - 1, width], dtype=np.float32)
    cost_h = beta * np.ones(shape=[height, width - 1], dtype=np.float32)

    mask_idx = gco.cut_grid_graph(unary, pairwise, cost_v, cost_h, algorithm=algorithm)
    return mask_idx


def graphcut_wrapper(cost_penalty, label_count, n_input, height, width, beta, device, iter_idx=0):
    '''Wrapper of graphcut_multi performing efficient extension to multi-label'''
    assigned_label = (label_count > 0)
    if iter_idx > 0:
        n_label = int(assigned_label.float().sum())
    else:
        n_label = 0

    if n_label == 2:
        cost_add = cost_penalty[:, :, assigned_label].mean(-1, keepdim=True) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=2,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 1,
                                    device=device).reshape(height, width, n_input + 1)

        idx_matrix = torch.zeros([1, 1, n_input], device=device)
        idx_matrix[:, :, assigned_label] = 0.5
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + mask_idx_onehot[:, :,
                                                                          n_input:] * idx_matrix
    elif n_label >= 3:
        soft_label = torch.tensor([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], device=device)

        _, indices = torch.topk(label_count, k=3)
        assigned_label = torch.zeros_like(assigned_label)
        assigned_label[indices] = True

        cost_add = torch.matmul(cost_penalty[:, :, assigned_label], soft_label) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=3,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 3,
                                    device=device).reshape(height, width, n_input + 3)

        idx_matrix = torch.zeros([3, n_input], device=device)
        idx_matrix[:, assigned_label] = soft_label
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + torch.matmul(
            mask_idx_onehot[:, :, n_input:], idx_matrix)
    else:
        unary = cost_penalty.cpu().numpy()
        mask_idx_np = graphcut_multi(unary, beta=beta, algorithm='swap')
        mask_onehot_i = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                  n_input,
                                  device=device).reshape(height, width, n_input)

    return mask_onehot_i


def get_onehot_matrix(cost_matrix,
                      A,
                      n_output,
                      idx=None,
                      beta=0.32,
                      gamma=1.,
                      eta=0.05,
                      mixup_alpha=2.0,
                      thres=0.84,
                      thres_type='hard',
                      set_resolve=True,
                      niter=3,
                      device='cuda'):
    '''Iterative submodular minimization algorithm with the modularization of supermodular term'''
    n_input, height, width = cost_matrix.shape
    thres = thres * height * width
    beta = beta / height / width
    gamma = gamma / height / width
    eta = eta / height / width

    add_cost = None

    # Add prior term
    lam = mixup_alpha * torch.ones(n_input, device=device)
    alpha = torch.distributions.dirichlet.Dirichlet(lam).sample().reshape(n_input, 1, 1)
    cost_matrix -= eta * torch.log(alpha + 1e-8)

    with torch.no_grad():
        # Init
        if idx is None:
            mask_idx = torch.tensor(random_initialize(n_input, n_output, height, width),
                                    device=device)
        else:
            mask_idx = idx

        mask_onehot = to_onehot(mask_idx.reshape(-1), n_input,
                                device=device).reshape([n_output, height, width, n_input])

        loss_prev = obj_fn(cost_matrix, mask_onehot, beta, gamma)
        penalty = to_onehot(mask_idx.reshape(-1), n_input, device=device).sum(0).reshape(-1, 1, 1)

        # Main loop
        for iter_idx in range(niter):
            for i in range(n_output):
                label_count = mask_onehot[i].reshape([height * width, n_input]).sum(0)
                penalty -= label_count.reshape(-1, 1, 1)
                if thres_type == 'hard':
                    modular_penalty = (2 * gamma * (
                        (A @ penalty.squeeze() > thres).float() * A @ penalty.squeeze())).reshape(
                            -1, 1, 1)
                elif thres_type == 'soft':
                    modular_penalty = (2 * gamma * ((A @ penalty.squeeze() > thres).float() *
                                                    (A @ penalty.squeeze() - thres))).reshape(
                                                        -1, 1, 1)
                else:
                    raise AssertionError("wrong threshold type!")

                if add_cost is not None:
                    cost_penalty = (cost_matrix + modular_penalty +
                                    gamma * add_cost[i].reshape(-1, 1, 1)).permute(1, 2, 0)
                else:
                    cost_penalty = (cost_matrix + modular_penalty).permute(1, 2, 0)

                mask_onehot[i] = graphcut_wrapper(cost_penalty, label_count, n_input, height, width,
                                                  beta, device, iter_idx)
                penalty += mask_onehot[i].reshape([height * width,
                                                   n_input]).sum(0).reshape(-1, 1, 1)

            if iter_idx == niter - 2 and set_resolve:
                assigned_label_total = (mask_onehot.reshape(n_output, -1, n_input).sum(1) >
                                        0).float()
                add_cost = resolve_label(assigned_label_total, device=device)

            loss = obj_fn(cost_matrix, mask_onehot, beta, gamma)
            if (loss_prev - loss).abs() / loss.abs() < 1e-6:
                break
            loss_prev = loss

    return mask_onehot

def mixup_process(out, target_reweighted, args=None, sc=None, A_dist=None):
    m_block_num = args.m_block_num
    m_part = args.m_part
    batch_size = out.shape[0]
    width = out.shape[-1]

    if A_dist is None:
        A_dist = torch.eye(batch_size, device=out.device)

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)

    block_size = width // m_block_num
    sc = F.avg_pool2d(sc, block_size)

    out_list = []
    target_list = []

    # Partition a batch
    for i in range(ceil(batch_size / m_part)):
        with torch.no_grad():
            sc_part = sc[i * m_part:(i + 1) * m_part]
            A_dist_part = A_dist[i * m_part:(i + 1) * m_part, i * m_part:(i + 1) * m_part]

            n_input = sc_part.shape[0]
            sc_norm = sc_part / sc_part.reshape(n_input, -1).sum(1).reshape(n_input, 1, 1)
            cost_matrix = -sc_norm

            A_base = torch.eye(n_input, device=out.device)
            A_dist_part = A_dist_part / torch.sum(A_dist_part) * n_input
            A = (1 - args.m_omega) * A_base + args.m_omega * A_dist_part

            # Return a batch(partitioned) of mixup labeling
            mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                            A,
                                            n_output=n_input,
                                            beta=args.m_beta,
                                            gamma=args.m_gamma,
                                            eta=args.m_eta,
                                            mixup_alpha=args.mixup_alpha,
                                            thres=args.m_thres,
                                            thres_type=args.m_thres_type,
                                            set_resolve=args.set_resolve,
                                            niter=args.m_niter,
                                            device='cuda')

        # Generate image and corrsponding soft target
        output_part, target_part = mix_input(mask_onehot, out[i * m_part:(i + 1) * m_part],
                                             target_reweighted[i * m_part:(i + 1) * m_part])

        out_list.append(output_part)
        target_list.append(target_part)

    with torch.no_grad():
        out = torch.cat(out_list, dim=0)
        target_reweighted = torch.cat(target_list, dim=0)

    return out.contiguous(), target_reweighted