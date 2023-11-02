import sys
import math
import time
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.constants import pi  # pi = 3.14159265358979
from scipy.constants import c  # 光速 c = 299792458 m/s
from scipy.constants import G  # 引力常数 = 6.67408e-11 m3/Kg/s2
from scipy.constants import au  # 日地距离(1天文单位) = 149597870691 m
from PIL import Image

M_sun = 1.98840987e+30  # 太阳质量 (kg)


class Blackhole:

    def __init__(self, imgstr='milkyway.jpg') -> None:

        self.Rs = 8  # 史瓦西半径(单位au)

        self.M = self.Rs * c**2 * au / 2 / G / M_sun  # 黑洞质量(单位M_sun)
        self.D = 50
        self.axe_X = 360  # size over x
        self.axe_Y = self.axe_X // 2
        self.img_res = self.axe_X / 360  # =Pixels per degree along axis
        self.img_res_Y = self.axe_Y / 180  # =Pixels per degree along axis
        self.FOV_img = 360
        self.FOV_img_Y = self.FOV_img // 2
        self.img_debut = None
        self.zoom = 0

        self.img_matrix_x = None
        self.img_matrix_y = None
        self.img2 = None

        try:
            self.abs_path = os.path.abspath(os.path.dirname(sys.argv[0]))
            folder = os.path.join(self.abs_path, 'images')
            img_name = os.path.join(folder, imgstr)
            self.open(img_name, size=self.axe_X)

        except FileNotFoundError:
            print("milkyway image not found!")

    def compute(self, Rs, D):
        """
        计算黑洞造成的时空变形, 并且将其应用到图像上.
        """
        self.Rs = Rs
        self.D = D
        self.M = (self.Rs * c**2 * au) / (2 * G * M_sun)

        print("M = %.1e M_sun\t%.2e Kg" % (self.M, self.M * M_sun))
        print("Rs = %s ua\t%.2e m" % (self.Rs, self.Rs * au))
        print("D = %s ua\t%.2e m\n" % (self.D, self.D * au))

        vrai_debut = time.process_time()
        seen_angle, deviated_angle = self.trajectories()
        self.interpolation = self.interpolate(seen_angle, deviated_angle)
        img_matrix_x, img_matrix_y = self.create_matrices()

        self.img_matrix_x = img_matrix_x
        self.img_matrix_y = img_matrix_y

        self.img2 = self.img_pixels(self.img_debut)

        vrai_fin = time.process_time()
        print("\nglobal computing time: %.1f\n" % (vrai_fin - vrai_debut))

        self.img2.save(os.path.join(self.abs_path, "output.png"))

    def interpolate(self, x_pivot, f_pivot):
        """创建插值数据"""
        interpolation = interp1d(x_pivot, f_pivot, kind='cubic',
                                 bounds_error=False)
        return interpolation

    def trajectories(self):
        """计算光子轨迹"""
        alpha_min = self.search_alpha_min()
        alpha_finder = self.axe_X / 2

        seen_angle = np.array([])
        deviated_angle = np.array([])

        booli = False
        points = 40

        for i in range(6):
            for alpha in np.linspace(alpha_finder, alpha_min, num=points,
                                     endpoint=booli):
                r, phi = self.solver(alpha)

                if r[-1] > 1.1 * self.Rs:  # if not capture by black hole
                    seen_angle = np.append(seen_angle, 180 - alpha)
                    dev_angle = phi[-1] + math.asin(
                        self.D / r[-1] * math.sin(phi[-1]))
                    dev_angle = math.degrees(dev_angle)
                    deviated_angle = np.append(deviated_angle, dev_angle)

            alpha_finder = alpha_min + \
                (alpha_finder - alpha_min) / (points + 1)

            points = 10

            if i == 4:
                booli = True

        return seen_angle, deviated_angle

    def search_alpha_min(self):
        """返回光子被黑洞捕捉的最大倾角"""
        alpha_min = 0

        for alpha in range(0, 180, 4):
            r = self.solver(alpha)[0]
            if r[-1] > 1.1 * self.Rs:
                break

        if (alpha - 4) > 0:
            alpha_min = alpha - 4
        i = 1

        while alpha_min == 0 or round(alpha_min * self.img_res) != round(
            (alpha_min + i * 10) * self.img_res):

            for alpha in range(int(alpha_min / i), int(180 / i), 1):
                alpha = alpha * i
                r = self.solver(alpha)[0]

                if r[-1] > 1.1 * self.Rs:
                    break

            if (alpha - i) > 0:
                alpha_min = alpha - i

            i = i / 10

        i = 10 * i
        alpha_min += i
        return alpha_min

    def solver(self, alpha):
        """在球坐标系下求解微分方程、计算光子轨迹, 给出其与
            黑洞距离及其初始角速度。"""
        if alpha == 0:  
            return [0], [0]  

        if alpha == 180:
            return [self.D], [0] 

        y0 = [1 / self.D, 1 / (self.D * math.tan(math.radians(alpha)))]
        sol = solve_ivp(fun=self._diff_eq,
                        t_span=[0, 10 * pi],
                        y0=y0,
                        method='Radau',
                        events=[self._eventRs])

        phi = np.array(sol.t)
        r = np.abs(1 / sol.y[0, :])

        return r, phi

    def _diff_eq(self, phi, u):
        """差分方程"""
        v0 = u[1]
        v1 = 3 / 2 * self.Rs * u[0]**2 - u[0]
        return v0, v1

    def _eventRs(self, phi, u):
        """黑洞内部, 则停止计算"""
        with np.errstate(all='ignore'):
            return 1 / u[0] - self.Rs

    _eventRs.terminal = True

    def create_matrices(self):
        """创建像素矩阵以及其对应位置信息
        """
        x = np.arange(0, self.axe_X)
        y = np.arange(0, self.axe_Y)
        xv, yv = np.meshgrid(x, y)

        img_matrix_x, img_matrix_y = self.find_position(xv, yv)
        return img_matrix_x, img_matrix_y

    def find_position(self, xv, yv):
        """获取看到的像素位置并搜索偏离的像素位置"""
        phi = xv * self.FOV_img / 360 / self.img_res
        theta = yv * self.FOV_img_Y / 180 / self.img_res_Y
        phi2 = phi + (360 - self.FOV_img) / 2
        theta2 = theta + (180 - self.FOV_img_Y) / 2

        u, v, w = spheric2cart(
            np.radians(theta2),
            np.radians(phi2))

        with np.errstate(all='ignore'):
            beta = -np.arctan(w / v)
        
        matrix = rotation_matrix(beta)
        u2 = matrix[0, 0] * u
        v2 = matrix[1, 1] * v + matrix[1, 2] * w
        w2 = matrix[2, 1] * v + matrix[2, 2] * w
        _, seen_angle = cart2spheric(u2, v2, w2) 

        seen_angle = np.degrees(seen_angle)
        seen_angle = np.mod(seen_angle, 360)

        deviated_angle = np.zeros(seen_angle.shape)
        deviated_angle[seen_angle < 180] = self.interpolation(
            seen_angle[seen_angle < 180])
        deviated_angle[seen_angle >= 180] = 360 - self.interpolation(
            360 - seen_angle[seen_angle >= 180])

        theta = pi / 2  # *np.ones(deviated_angle.shape)
        phi = np.radians(deviated_angle)
        u3, v3, w3 = spheric2cart(theta, phi)

        matrix = rotation_matrix(-beta)
        u4 = matrix[0, 0] * u3
        v4 = matrix[1, 1] * v3 + matrix[1, 2] * w3
        w4 = matrix[2, 1] * v3 + matrix[2, 2] * w3

        theta, phi = cart2spheric(u4, v4, w4) 
        theta, phi = np.degrees(theta), np.degrees(phi)

        phi -= (360 - self.FOV_img) / 2
        theta -= (180 - self.FOV_img_Y) / 2

        with np.errstate(all='ignore'):  
            phi = np.mod(phi, 360)
            theta = np.mod(theta, 180)

        phi[phi == 360] = 0
        xv2 = phi * 360 / self.FOV_img * self.img_res
        yv2 = theta * 180 / self.FOV_img_Y * self.img_res_Y

        xv2[np.isnan(xv2)] = -1
        yv2[np.isnan(yv2)] = -1

        xv2 = np.array(xv2, dtype=int)
        yv2 = np.array(yv2, dtype=int)

        return xv2, yv2

    def open(self, img_name, size="default"):
        self.img_original = Image.open(img_name, mode='r')
        self.img_name = img_name

        if size == "default":
            size = self.img_original.size[0]

        self.img_debut = self.img_resize(size)
        return self.img_debut

    def img_resize(self, axe_X):
        self.img_debut = self.img_original.convert("RGB")
        size_X, size_Y = self.img_debut.size
        size_factor = axe_X / size_X
        axe_X = int(axe_X)
        axe_Y = int(size_factor * size_Y)

        if axe_X % 2 != 0:
            axe_X -= 1

        if axe_Y % 2 != 0:
            axe_Y -= 1

        self.img_debut = self.img_debut.resize((axe_X, axe_Y))
        self.FOV_img_Y = self.FOV_img * axe_Y / axe_X

        if self.FOV_img_Y > 180:
            raise StopIteration("Can't have a FOV>180 in the Y-axis")

        print("size %sx%s pixels\n" % (axe_X, axe_Y))
        self.img_res = axe_X / 360  # =Pixels per degree along axis
        self.img_res_Y = axe_Y / 180  # =Pixels per degree along axis

        self.axe_X, self.axe_Y = axe_X, axe_Y

        return self.img_debut

    def img_pixels(self, img_debut):
        pixels = np.array(img_debut)
        pixels2 = np.array(img_debut)

        xv, yv = self.img_matrix_x, self.img_matrix_y

        yv[yv >= self.axe_Y] = -2 
        xv[xv >= self.axe_X] = -2

        pixels2 = pixels[yv, xv] 
        pixels2[xv == -1] = [0, 0, 0]  # color the black hole in black
        pixels2[yv == -2] = [255, 192, 203]  # color pixels outside
        pixels2[xv == -2] = [255, 192, 203]

        img2 = Image.fromarray(pixels2.astype('uint8'), 'RGB')
        return img2


def rotation_matrix(beta):
    beta = np.array(beta)
    aa_bb, ab2neg = np.cos(beta), np.sin(beta)
    zero, one = np.zeros(beta.shape), np.ones(beta.shape)

    return np.array([[one, zero, zero], [zero, aa_bb, -ab2neg],
                     [zero, ab2neg, aa_bb]])


def img_offset_X(img, offset_X):
    offset_X = int(offset_X)
    (axe_X, axe_Y) = img.size

    while offset_X >= axe_X:
        offset_X -= axe_X

    if offset_X == 0:
        return img

    if offset_X < 0:
        offset_X = -offset_X
        img_right = img.crop((0, 0, axe_X - offset_X, axe_Y))
        img_left = img.crop((axe_X - offset_X, 0, axe_X, axe_Y))
        img.paste(img_right, (offset_X, 0))

    else:
        img_right = img.crop((0, 0, offset_X, axe_Y))
        img_left = img.crop((offset_X, 0, axe_X, axe_Y))
        img.paste(img_right, (axe_X - offset_X, 0))

    img.paste(img_left, (0, 0))

    return img


def spheric2cart(theta, phi):
    """将球面坐标转换为笛卡尔坐标。"""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cart2spheric(x, y, z):
    """笛卡尔坐标转球面"""
    with np.errstate(all='ignore'):
        theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi


if __name__ == "__main__":
    blackhole = Blackhole('milkyway.jpg')
    blackhole.compute(8, 50)
