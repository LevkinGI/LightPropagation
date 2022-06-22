import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.fft import fft, ifft, fftshift, fftfreq, rfft
from scipy.integrate import solve_ivp, RK23
from scipy.constants import c, pi

import numpy as np

import sys
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from Interface import MainWindow

from MyException import *

exp, sqrt, cos = np.exp, np.sqrt, np.cos

"""#Создаем волну"""


class Wave:
    def __init__(self, lam, a0, tau, running_cs=False):
        self.A0 = a0
        self.tau = tau
        self.lam = lam
        self.running_cs = running_cs
        self.n = self.n_lam(self.lam)
        self.phase_speed = self.phase_speed_lam(self.lam)
        self.k = 2 * pi / self.lam
        self.omega = self.k * self.phase_speed
        self.group_speed = self.group_speed_lam(self.lam)

    @staticmethod
    def n_lam(lam):
        # return 1
        lam1 = lam * 1e+6
        return sqrt(1 + 0.5675888 * lam1 ** 2 / (lam1 ** 2 - 0.050263605 ** 2) +
                    0.4710914 * lam1 ** 2 / (lam1 ** 2 - 0.1003909 ** 2) +
                    3.8484723 * lam1 ** 2 / (lam1 ** 2 - 34.649040 ** 2))

    def phase_speed_lam(self, lam):
        return c / self.n_lam(lam)

    def group_speed_lam(self, lam):
        return c / (self.n_lam(lam) - self.lam * derivative(self.n_lam, self.lam, 1e-9))

    def k2(self):
        return self.lam ** 3 / (2 * pi * c ** 2) * derivative(func=self.n_lam, x0=self.lam, dx=1e-12, n=2)

    def k_lam(self, lam):
        return self.k \
               + 1 / self.group_speed_lam(lam) * (lam - self.lam) \
               + 0.5 * self.k2() * (lam - self.lam) ** 2

    def A_z(self, z):
        return self.A0 * (1 + (z * abs(self.k2()) / self.tau ** 2) ** 2) ** -4 \
               * exp(-1j * 0.5 * np.arctan(self.k2() * z / self.tau ** 2))

    def A_z_t(self, z, t):

        L = self.tau**2 / abs(self.k2())
        V0 = sqrt(1 + (z / L) ** 2)
        fi = (z/L)**2 / (2*V0**2*self.k2()*z)*t**2 - 0.5 * np.arctan(self.k2() * z / self.tau ** 2)

        if self.running_cs:
            return V0**-0.5 * self.A0*exp(-t**2/(2*V0**2*tau**2) + 1j*fi)

            #return self.A_z(z) * exp(-t ** 2 / self.tau ** 2) * exp(-1j * self.omega * t)

        return self.A_z(z) * exp(-(t - z / self.group_speed) ** 2 / self.tau ** 2) \
               * exp(-1j * self.omega * (t - z / self.group_speed))

    def A_omega(self, t):
        omega = fftshift(fftfreq(len(t), (t.max() - t.min()) / len(t)) * 2 * pi)
        A_omega = abs(fftshift(ifft(self.A_z_t(0, t))))
        return omega, A_omega


A = Wave(lam=4e-6, a0=1e+15, tau=80e-15, running_cs=False)
t = np.linspace(-500e-13, 500e-13, 10 ** 7, endpoint=False)

# omega, A_omega = A.A_omega(t)

# plt.plot(t, A_t)
# #plt.show()

# plt.plot(omega, A_omega)
# plt.show()

# car_freq_index = np.where(A_omega == A_omega.max())[0][0]
# plt.plot(omega[car_freq_index - 1000:car_freq_index + 1000], A_omega[car_freq_index - 1000:car_freq_index + 1000])
# plt.show()

"""#Решаем диффур"""

T = 1e-12
Z = 2e-4
l = T*A.group_speed
N = 10 ** 3
z = np.linspace(0, l, N, endpoint=False)
t = np.linspace(0, T, 2 * N, endpoint=False)
h, tau = z[1] - z[0], t[1] - t[0]
A_0_t = A.A_z_t(0, t)
A_z_0 = A.A_z_t(z, 0)
A_l_t = np.zeros(len(A_0_t))
# A_l_t = A.A_z_t(l, t)

def solvePDE(a_z_0, a_0_t, u, h, tau):
    a_z_t = np.zeros((len(a_0_t), len(a_z_0)), dtype=float)
    a_z_t[0, :], a_z_t[:, 0] = a_z_0, a_0_t

    for z in range(1, len(a_z_0)):
        for t in range(1, len(a_0_t)):
            a_z_t[t, z] = (1 / (2 * tau) * (a_z_t[t - 1, z] + a_z_t[t - 1, z - 1] - a_z_t[t, z - 1])
                           + u / (2 * h) * (a_z_t[t, z - 1] + a_z_t[t - 1, z - 1] - a_z_t[t - 1, z])) / (
                                  1 / (2 * tau) + u / (2 * h))
    return a_z_t[:abs(len(a_0_t)-len(a_z_0)), :]


def solvePDE2(a_z_0, a_0_t, k, h, tau, a_l_t=None, sigma=0.):
    a_z_t = np.zeros((len(a_0_t), len(a_z_0)), dtype=float)
    if a_l_t is None:
        if h > tau ** 2 / (2 * abs(k)):
            raise InequalityException
        if sigma != 0:
            raise SigmaException
        if len(a_z_0) > len(a_0_t):
            raise LengthConditionsException

        a_z_t[0, :], a_z_t[:, 0] = a_z_0, a_0_t
        for z in range(1, len(a_z_0)):
            for t in range(1, len(a_0_t) - z):
                a_z_t[t, z] = (k / tau ** 2 * (a_z_t[t - 1, z - 1] + a_z_t[t + 1, z - 1])
                               + 1 / h * a_z_t[t, z - 1])\
                              / (1 / h + 2 * k / tau ** 2)
        return a_z_t[:abs(len(a_0_t)-len(a_z_0)), :]


    a_z_t[0, :], a_z_t[:, 0], a_z_t[:, -1] = a_z_0, a_0_t, a_l_t
    for z in range(1, len(a_z_0)):
        for t in range(1, len(a_0_t)-1):
            a_z_t[t, z] = (k * sigma / tau ** 2 * (a_z_t[t - 1, z] + a_z_t[t + 1, z])
                           + k * (1 - sigma) / tau ** 2 * (a_z_t[t - 1, z - 1] + a_z_t[t + 1, z - 1])
                           + 1 / h * a_z_t[t, z - 1]) \
                          / (1 / h + 2 * k * sigma / tau ** 2 + 2 * k * (1 - sigma) / tau ** 2)
    return a_z_t[:abs(len(a_0_t)-len(a_z_0)), :]

A_z_t = solvePDE(A_z_0, A_0_t, A.group_speed, h, tau)
#A_z_t = solvePDE2(A_z_0, A_0_t, 1j*0.5*A.k2(), h, tau, a_l_t=A_l_t, sigma=0.5)
# A_z_t = solvePDE2(a_z_0=A_z_0, a_0_t=A_0_t, k=1j*0.5*A.k2(), h=h, tau=tau, a_l_t=None)
# for z in range(0, len(A_z_0)):
#     for t in range(0, len(A_0_t)):
#         A_z_t = A.A_z_t(z, t)

app = QApplication(sys.argv)
main_window = MainWindow(z, t[:abs(len(A_0_t)-len(A_z_0))], A_z_t.real)
main_window.show()
app.exec()
