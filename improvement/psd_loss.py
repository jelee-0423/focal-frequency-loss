"""
@Description  : This script contains three methods for 1D AI distribution calculation, all work well so just select one as per your need.
All are based on gray-scale image. Please modify to RGB image base.
@Author       : Chi Liu
@Date         : 2022-03-25 17:53:20
@LastEditTime : 2022-04-07 20:51:12
"""
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

# from torchvision.transforms import GaussianBlur

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


#################### improvement #####################


class PSDLoss(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False,
                 batch_matrix=False):
        super(PSDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, img):
        # CxNxHxW, torch.fft.fft2默认只处理后两维，分别对每个样本的每个通道的HxW做fft，互不干扰
        # freq = torch.fft.fft2(y, norm='ortho')  # ffl中做了'1/sqrt(H*W)'的归一化，但是考虑到psd中要计算振幅，所以这里和AI中保持一致
        freq = torch.fft.fft2(img)  # torch版本要高于1.7.1，因为在低于等于1.7.1的版本中没找到fftshift函数，而这个变换是AI所需要的
        freq = torch.fft.fftshift(freq, (2, 3))  # 只变换后两维
        freq = torch.stack([freq.real, freq.imag], -1)  # 和ffl统一，方便理解
        return freq

    def psd1D_circle_mask(self, h, w, psd1D):
        """
        Generate a circle mask of AI/psd1D, circles of the same radius share the same AI value.
        Now only work for images of the same height and width, with the center in (h//2, w//2).
        numpy version.
        :param h: int, the height of target image/mask
        :param w: int, the width of target image/mask
        :param psd1D: ndarray, the 1D psd of image, len(psd1D) = int(h*np.sqrt(2)/2)-2
        :return: ndarray, circle mask with height h and width w
        """
        assert h == w, 'The height and width of the image are not equal!'

        # draw a circle
        y, x = np.indices((h, w))
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
        r = np.hypot(x - center[0], y - center[1])  # r[i][i] = (x[i][i]**2 + y[i][i]**2)**0.5
        r_int = r.astype(int)

        psd1D = np.insert(psd1D, 0, psd1D[0])  # 将中心点（半径为0的点）的psd值设为和半径为1的同样的值
        psd1D = np.append(psd1D, 0)  # 将四个角（径向积分时忽略的边角点）的psd值设为0

        mask = r_int.copy()
        for idx, psd in enumerate(psd1D):
            mask = np.where(mask == idx, psd, mask)  # 对应半径的圆周上的点设置为对应的psd值

        return mask

    def get_mask(self, freq):
        freq_np = freq.cpu().detach().numpy()  # NCHW2
        freq_np = freq_np[..., 0] + 1j * freq_np[..., 1]  # NCHW
        freq_np += 1e-8
        magnitude_spectrum = 20 * np.log(np.abs(freq_np))  # still NCHW

        n, c, h, w = freq_np.shape
        # len_psd = int(h*np.sqrt(2)/2) - 2
        mask_all = np.zeros(freq_np.shape)

        for i in range(n):
            for j in range(c):
                psd1D = azimuthalAverage(magnitude_spectrum[i][j])
                psd1D = (psd1D - np.min(psd1D)) / (np.max(psd1D) - np.min(psd1D))  # max-min normalization
                mask = self.psd1D_circle_mask(h, w, psd1D)
                mask_all[i][j] = mask

        return mask_all

    def loss_formulation(self, recon_freq, real_freq, norm='l2', weight='recon', matrix=None):
        # frequency distance using (squared) Euclidean distance
        tmp = recon_freq ** 2 - real_freq ** 2  # |Fr|**2 - |Ff|**2, 得到实数？
        # 1. tmp是一个(a, b)形式的复数a+bi
        # if norm == 'l1':
        #     tmp = tmp ** 2  # (a**2, b**2)
        #     freq_distance = torch.sqrt(tmp[..., 0] + tmp[..., 1])  # sqrt(a**2 + b**2)
        # else:  # 默认为l2-norm
        #     tmp = tmp ** 2  # (a**2, b**2)
        #     freq_distance = tmp[..., 0] + tmp[..., 1]  # (sqrt(a**2 + b**2))**2 => a**2 + b**2

        # 2. tmp是实数
        tmp = tmp[..., 0] + tmp[..., 1]  # equal to torch.sum(recon_freq**2, 4) - torch.sum(real_freq**2, 4)
        if norm == 'l1':
            freq_distance = torch.abs(tmp)
        else:
            freq_distance = tmp**2

        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, detached
            recon_mask_all = self.get_mask(recon_freq)

            if weight == 'recon_real':
                real_mask_all = self.get_mask(real_freq)
                weight_matrix = torch.abs(torch.from_numpy(real_mask_all - recon_mask_all))
            else:  # weight == 'recon'
                weight_matrix = torch.from_numpy(1 - recon_mask_all)

            weight_matrix[torch.isnan(weight_matrix)] = 0.0
            weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, norm='l2', weight='recon', matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, norm, weight, matrix) * self.loss_weight


###################### Method 1 ######################
# The function spectral_vector() outputs the 1D AI
######################################################


class PSDLoss1(nn.Module):
    f_cache = "spectralloss.{}.cache"

    def __init__(self,
                 rows,
                 cols,
                 eps=1E-8,
                 cache=False,
                 is_avg=False,
                 is_thre=False,
                 is_filter=False):
        super(PSDLoss1, self).__init__()
        self.img_size = rows
        self.is_avg = is_avg
        self.is_thre = is_thre
        self.is_filter = is_filter
        self.eps = eps
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(rows / 2)
        # number of cols after onesided fft
        cols_onesided = int(cols / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices(
            (rows, cols_onesided)) - np.array([[[shift_rows]], [[0]]])  # 一半的列，而行也需要一半[-shift_rows, shift_rows]
        r = np.sqrt(r[0, :, :] ** 2 + r[1, :, :] ** 2)  # 就是一半的r
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r, axes=0)  # axes=0表示只更改行，列不变，把最中间的零移到最上边
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = torch.from_numpy(r).expand(r_max + 1, -1, -1).to(
            torch.float)  # 复制了r_max+1个，shape为(r_max+1, r.shape[0], r.shape[1])
        radius_to_slice = torch.arange(r_max + 1).view(-1, 1, 1)
        # generate mask for each radius
        mask = torch.where(
            r == radius_to_slice,
            torch.tensor(1, dtype=torch.float),
            torch.tensor(0, dtype=torch.float),
        )  # https://www.cnblogs.com/massquantity/p/8908859.html 相等的话设为第二个参数(1)，不相等的话设为第三个参数(0)
        # how man entries for each radius?
        mask_n = torch.sum(mask, axis=(1, 2))  # 计算每一圈的点之和（圈与圈之间的半径差为1）
        mask = mask.unsqueeze(0)  # add batch dimension
        # normalization vector incl. batch dimension
        mask_n = (1 / mask_n.to(torch.float)).unsqueeze(0)  # 每一圈累加之后，需要除以累加点的数量，归一化
        self.criterion_l1 = torch.nn.L1Loss(reduction="sum")
        self.r_max = r_max
        self.vector_length = r_max + 1

        self.register_buffer("mask", mask)
        self.register_buffer("mask_n", mask_n)

        if cache and os.path.isfile(self.f_cache.format(self.img_size)):
            self._load_cache()
        else:
            self.is_fitted = False
            self.register_buffer("mean", None)

        if device is not None:
            self.to(device)
        self.device = device

    def _save_cache(self):
        torch.save(self.mean, self.f_cache.format(self.img_size))
        self.is_fitted = True

    def _load_cache(self):
        mean = torch.load(self.f_cache.format(self.img_size),
                          map_location=self.mask.device)
        self.register_buffer("mean", mean)
        self.is_fitted = True

    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################

    def fft(self, data):
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data = (0.299 * data[:, 0, :, :] + 0.587 * data[:, 1, :, :] +
                    0.114 * data[:, 2, :, :])

        fft = torch.rfft(data, signal_ndim=2, onesided=True)  # 和torch.fft.fft2的区别是不是只在于这里分开了实部虚部？
        # fft = torch.fft.rfft(data)
        # abs of complex
        fft_abs = torch.sum(fft ** 2, dim=3)
        fft_abs = fft_abs + self.eps
        fft_abs = 20 * torch.log(fft_abs)
        return fft_abs

    def spectral_vector(self, data):
        """get 1d psd AI profile. Assumes first dimension to be batch size."""
        fft = (self.fft(data).unsqueeze(1).expand(-1, self.vector_length, -1,
                                                  -1)
               )  # repeat img for each radius

        # apply mask and compute profile vector
        profile = (fft * self.mask).sum((2, 3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - profile.min(1)[0].view(-1, 1)
        profile = profile / profile.max(1)[0].view(-1, 1)

        return profile

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_filter:
            filter = GaussianBlur(3, 0.2)
            pred = pred - filter(pred)
            target = target - filter(target)

        if self.is_thre:
            frequency_thre = int(0.1 * self.vector_length)
        else:
            frequency_thre = 0

        pred_profiles = self.spectral_vector(pred)[:, frequency_thre:]
        target_profiles = self.spectral_vector(target)[:, frequency_thre:]

        if self.is_avg:
            target_profiles_avg = target_profiles.mean(dim=0)
            target_profiles = torch.zeros_like(target_profiles)
            for i in range(target_profiles.shape[0]):
                target_profiles[i, :] = target_profiles_avg
        pred_profiles = Variable(pred_profiles, requires_grad=False).to(device)
        target_profiles = Variable(target_profiles,
                                   requires_grad=True).to(device)

        # criterion = nn.BCELoss()
        criterion = nn.MSELoss()
        return criterion(pred_profiles, target_profiles)


###################### Method 2 ######################
# The function get_fft_feature() outputs the 1D AI
######################################################

class PSDLoss2(nn.Module):
    def __init__(self):
        super().__init__()

    def RGB2gray(self, rgb):
        if rgb.size(1) == 3:
            r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        elif rgb.size(1) == 1:
            return rgb[:, 0, :, :]

    def shift(self, x):
        out = torch.zeros(x.size())
        H, W = x.size(-2), x.size(-1)
        out[:, :int(H / 2), :int(W / 2)] = x[:, int(H / 2):, int(W / 2):]
        out[:, :int(H / 2), int(W / 2):] = x[:, int(H / 2):, :int(W / 2)]
        out[:, int(H / 2):, :int(W / 2)] = x[:, :int(H / 2), int(W / 2):]
        out[:, int(H / 2):, int(W / 2):] = x[:, :int(H / 2), :int(W / 2)]
        return out

    def azimuthalAverage(self, spe_ts):
        """
        Calculate the azimuthally averaged radial profile from a W*W spectrum spe_ts
        """
        x_index = torch.zeros_like(spe_ts)
        W = spe_ts.size(0)
        for i in range(W):
            x_index[i, :] = torch.arange(0, W)
        x_index = x_index.to(dtype=torch.int)
        y_index = x_index.transpose(0, 1)  # x_index, y_index = np.indices(spe_ts.shape)
        radius = torch.sqrt((x_index - 10 / 2) ** 2 + (y_index - 10 / 2) ** 2)  # 5作为圆心？
        radius = radius.to(dtype=torch.int)
        radius = torch.flatten(radius)
        radius_bin = torch.bincount(
            radius)  # Count the frequency of each value in an array of non-negative ints. 从最小值到最大值，间隔1为1个bin
        ten_bin = torch.bincount(radius, spe_ts.flatten())
        radial_prof = ten_bin / (radius_bin + 1e-10)
        return radial_prof

    def get_fft_feature(self, x_rgb):
        """get 1d psd AI profile

        Args:
            x_rgb (torch.Tensor): RGB image batch tensor with size N*W*W

        Returns:
            torch.Tensor: 1d psd profile
        """

        epsilon = 1e-8

        x_gray = self.RGB2gray(x_rgb)
        fft = torch.rfft(x_gray, 2, onesided=False)
        fft += epsilon
        magnitude_spectrum = torch.log((torch.sqrt(fft[:, :, :, 0] ** 2 +
                                                   fft[:, :, :, 1] ** 2)) +
                                       epsilon)
        magnitude_spectrum = self.shift(magnitude_spectrum)

        out = []
        for i in range(magnitude_spectrum.size(0)):
            out.append(
                self.azimuthalAverage(
                    magnitude_spectrum[i]).float().unsqueeze(0))
        out = torch.cat(out, dim=0)
        out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (
                torch.max(out, dim=1, keepdim=True)[0] -
                torch.min(out, dim=1, keepdim=True)[0])
        return out

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
        """
        pred_psd = self.get_fft_feature(pred)
        target_psd = self.get_fft_feature(target)
        mse = nn.MSELoss()
        return mse(pred_psd, target_psd).to(device)


###################### Method 3 ######################
# A single function azimuthalAverage() outputs the 1D AI
######################################################

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])  # r[i][i] = (x[i][i]**2 + y[i][i]**2)**0.5

    # Get sorted radii
    ind = np.argsort(
        r.flat)  # r.flat就是将r扁平化（第二行放在第一行后面，第三行再放后面等等），但是返回的不是扁平化的np.array，需要通过r.flat[i]来取值；np.argsort从小到大排序
    r_sorted = r.flat[ind]  # 从小到大排序后的扁平化后的r，shape为(image.shape[0]*image.shape[1], )
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)  # 不是四舍五入，只取整数位

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[
                         :-1]  # Assumes all radii represented  # r_int[1:]不带第一项，r_int[:-1]不带最后一项，两个相减表示用r_int的后一项减前一项
    rind = np.where(deltar)[
        0]  # location of changed radius  # np.where(deltar)输出deltar中非零元素的坐标，由于返回一个元组，当deltar为一维时，取第零个就好（后面也没别的了）
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted,
                     dtype=float)  # csim[i] = csim[i-1]+i_sorted[i], csim[0]=0; 运行时警告：复数部分丢失？(ComplexWarning: Casting complex values to real discards the imaginary part
    tbin = csim[rind[1:]] - csim[rind[:-1]]  # 用有变化的后一个减前一个，就得到一圈的总和

    radial_prof = tbin / nr

    return radial_prof
