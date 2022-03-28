# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import torchvision
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, feature_image, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), \
        torch.squeeze(torchvision.transforms.functional.affine(img=torch.unsqueeze(feature_image, 0), angle=0,
            translate=(0,0), scale=1.0, shear=(v*180,0)),0)


def ShearY(img, feature_image, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), \
        torch.squeeze(torchvision.transforms.functional.affine(img=torch.unsqueeze(feature_image, 0), angle=0,
            translate=(0,0), scale=1.0, shear=(0,v*180)),0)


def TranslateX(img, feature_image, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), \
        torch.squeeze(torchvision.transforms.functional.affine(img=torch.unsqueeze(feature_image, 0), angle=0,
            translate=(v,0), scale=1.0, shear=(0,0)),0)


def TranslateY(img, feature_image, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), \
        torch.squeeze(torchvision.transforms.functional.affine(img=torch.unsqueeze(feature_image, 0), angle=0,
            translate=(0,v), scale=1.0, shear=(0,0)),0)


def Rotate(img, feature_image, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v), \
        torch.squeeze(torchvision.transforms.functional.affine(img=torch.unsqueeze(feature_image, 0), angle=v,
            translate=(0,0), scale=1.0, shear=(0,0)),0)


def AutoContrast(img, feature_image, _):
    return PIL.ImageOps.autocontrast(img), feature_image


def Invert(img, feature_image, _):
    return PIL.ImageOps.invert(img), feature_image


def Equalize(img, feature_image, _):
    return PIL.ImageOps.equalize(img), feature_image


def Flip(img, feature_image, _):  # not from the paper
    return PIL.ImageOps.mirror(img), feature_image


def Solarize(img, feature_image, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v), feature_image


def Posterize(img, feature_image, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v), feature_image


def Contrast(img, feature_image, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v), feature_image


def Color(img, feature_image, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v), feature_image


def Brightness(img, feature_image, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v), feature_image


def Sharpness(img, feature_image, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v), feature_image


def Cutout(img, feature_image, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, feature_image, v)


def CutoutAbs(img, feature_image, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)

    feature_image[int(x0):int(x1),int(y0):int(y1)] = 0
    return img, feature_image


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, feature_image, v):
    return img, feature_image


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0),
        (ShearX, 0., 0.3),  # 0
        (ShearY, 0., 0.3),  # 1
        (TranslateX, 0., 0.33),  # 2
        (TranslateY, 0., 0.33),  # 3
        (Rotate, 0, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 110),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        # (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    # l = [
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        # (Invert, 0, 1),
        # (Rotate, 0, 30),
        # (Posterize, 0, 4),
        # (Solarize, 0, 256),
        # (SolarizeAdd, 0, 110),
        # (Color, 0.1, 1.9),
        # (Contrast, 0.1, 1.9),
        # (Brightness, 0.1, 1.9),
        # (Sharpness, 0.1, 1.9),
        # (ShearX, 0., 0.3),
        # (ShearY, 0., 0.3),
        # (CutoutAbs, 0, 40),
        # (TranslateXabs, 0., 100),
        # (TranslateYabs, 0., 100),
    # ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img, feature_image):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img, feature_image = op(img, feature_image, val)

        return img, feature_image