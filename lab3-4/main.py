import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

Steps = 1000
t = np.linspace(0, 25, Steps)
m1 = 0.1
m2 = 1
m3 = 0.1
l = 4
r = 1
c = 10
g = 9.8




def odesys(y, t, m1, m2, m3, l, r, c):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = l * (m1 / 3 + m2 + m3)
    a12 = m3 * r * np.cos(y[0] + y[1])
    a21 = m3 * r * np.cos(y[0] + y[1])
    a22 = (m2 / 2 + m3) * r

    b1 = (m1 / 2 + m2 + m3) * g * np.sin(y[1]) - c / l * (y[1] + y[0]) + m3 * r * (y[2] ** 2) * np.sin(y[0] + y[1])
    b2 = m3 * g * np.sin(y[0]) - c / r * (y[0] + y[1]) + m3 * l * (y[3] ** 2) * np.sin(y[0] + y[1])

    dy[2] = (b1 * a22 - b2 * a12)/(a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21)/(a11 * a22 - a12 * a21)
    return dy

# 0.35 0.35 0.025 0.025

psi0 = 0
phi0 = 0
dpsi0 = 0
dphi0 = 0





y0 = [psi0, phi0, dpsi0, dphi0]

Y = odeint(odesys, y0, t, (m1, m2, m3, l, r, c))

psi = Y[:, 0]
phi = Y[:, 1]


fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(1, 2, 1)
ax.axis('equal')
ax.set(xlim=[-7, 7], ylim=[-7, 3])

X_Ground = [-1, 1]
Y_Ground = [0, 0]
ax.plot(X_Ground, Y_Ground, color='black', linewidth=4)

L = 4
WheelR = 1
T = 2 * 3.14 * np.sqrt(L / g)
omega = 2 * 3.14 / T

A = np.sin(phi[0]) * l

X_A = A * np.sin(phi)
Y_A = -np.sqrt(L ** 2 - X_A ** 2)

X_V_A = np.diff(X_A)
Y_V_A = np.diff(Y_A)

X_W_A = np.diff(X_V_A)
Y_W_A = np.diff(Y_V_A)


ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(X_V_A)
plt.title('Vx of dot')
plt.xlabel('t values')
plt.ylabel('Vx values')

ax2 = fig.add_subplot(4, 2, 4)
ax2.plot(Y_V_A)
plt.title('Vy of dot')
plt.xlabel('t values')
plt.ylabel('Vy values')

ax2 = fig.add_subplot(4, 2, 6)
ax2.plot(X_W_A)
plt.title('Wx of dot')
plt.xlabel('t values')
plt.ylabel('Wx values')

ax2 = fig.add_subplot(4, 2, 8)
ax2.plot(Y_W_A)
plt.title('Wy of dot')
plt.xlabel('t values')
plt.ylabel('Wy values')

plt.subplots_adjust(wspace=0.3, hspace=0.7)


tetta = np.linspace(0, 6.28, 25)
X_Wheel = WheelR * np.sin(tetta)
Y_Wheel = WheelR * np.cos(tetta)
Drawed_Wheel = ax.plot(X_A[0] + X_Wheel, Y_A[0] + Y_Wheel)[0]

Point_O = ax.plot(0, 0, marker='o')[0]
Point_A = ax.plot(X_A[0], Y_A[0], marker='o')[0]
Line_AO = ax.plot([X_A[0], 0], [Y_A[0], 0])[0]

A = np.sin(psi[0]) * WheelR
X_B = A * np.sin(phi)
Y_B = np.sqrt(WheelR ** 2 - X_B ** 2)
Point_B = ax.plot(X_B[0] + X_A[0], Y_B[0] + Y_A[0], marker='o')[0]
Line_AB = ax.plot([X_A[0], X_B[0] + X_A[0]], [Y_A[0], Y_B[0] + Y_A[0]])[0]

Nv = 3
R1 = 0.1
R2 = 0.5
gretta = np.linspace(0, Nv * 6.28 - np.arctan((X_A[0] / Y_A[0])), 100)
X_SpiralSpr = -(R1 + gretta * (R2 - R1) / gretta[-1]) * np.sin(gretta)
Y_SpiralSpr = (R1 + gretta * (R2 - R1) / gretta[-1]) * np.cos(gretta)
Drawed_SpiralSpring = ax.plot(X_SpiralSpr + X_A[0], Y_SpiralSpr + Y_A[0])[0]


def anima(i):
    Line_AO.set_data([X_A[i], 0], [Y_A[i], 0])
    Point_A.set_data(X_A[i], Y_A[i])
    Drawed_Wheel.set_data(X_A[i] + X_Wheel, Y_A[i] + Y_Wheel)
    Point_B.set_data(X_B[i] + X_A[i], Y_B[i] + Y_A[i])
    Line_AB.set_data([X_A[i], X_B[i] + X_A[i]], [Y_A[i], Y_B[i] + Y_A[i]])

    gretta = np.linspace(0, Nv * 6.28 - np.arctan((X_A[i] / Y_A[i])), 100)
    X_SpiralSpr = -(R1 + gretta * (R2 - R1) / gretta[-1]) * np.sin(gretta)
    Y_SpiralSpr = (R1 + gretta * (R2 - R1) / gretta[-1]) * np.cos(gretta)
    Drawed_SpiralSpring.set_data(X_SpiralSpr + X_A[i], Y_SpiralSpr + Y_A[i])

    return [Line_AO, Point_A, Drawed_Wheel, Line_AB, Point_B, Drawed_SpiralSpring]

anim = FuncAnimation(fig, anima, frames=Steps, interval=100, blit=True)

plt.show()