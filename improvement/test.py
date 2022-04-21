from psd_loss import PSDLoss1, PSDLoss2, azimuthalAverage
from PIL import Image
from torchvision import transforms
import numpy as np


def RGB2gray(rgb):
    # input: hwc
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def psd1D_circle_mask(h, w, psd1D):
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

    psd1D = np.insert(psd1D, 0, 1)  # 将中心点（半径为0的点）的psd值设为1
    psd1D = np.append(psd1D, 0)  # 将四个角（径向积分时忽略的边角点）的psd值设为0

    mask = r_int.copy()
    for idx, psd in enumerate(psd1D):
        mask = np.where(mask == idx, psd, mask)  # 对应半径的圆周上的点设置为对应的psd值

    return mask


img = Image.open('./gakki.jpg')
img = transforms.ToTensor()(img).unsqueeze(0)

# PSDLoss Test
# psd = PSDLoss1(183, 183)
# loss = psd(img_L, img_L)
# print('Done.')

# AI Test
# PSDLoss1
# psd1 = PSDLoss1(182, 182)
# ai1 = psd1.spectral_vector(img)  # 129
#
# psd2 = PSDLoss2()
# ai2 = psd2.get_fft_feature(img)  # 249

img_np = img.squeeze(0).permute(1, 2, 0).numpy()[:182, :182]
img_np = RGB2gray(img_np)
fft = np.fft.fft2(img_np)
fshift = np.fft.fftshift(fft)
fshift += 1e-8
magnitude_spectrum = 20 * np.log(np.abs(fshift))  # np.abs计算复数的模，magnitude spectrum是振幅谱
psd1D = azimuthalAverage(magnitude_spectrum)  # 126 = int(182)*np.sqrt(2)/2
ai3 = (psd1D - np.min(psd1D)) / (np.max(psd1D) - np.min(psd1D))
circle_mask = psd1D_circle_mask(182, 182, ai3)

print('Done.')
