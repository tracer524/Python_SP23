import sympy as sym
V, T = sym.symbols("V, T", positive = True)
a, b, R, s = sym.symbols("a, b, R, s", positive = True)
#各气体状态方程
perfect = R*T/V
vdW = R*T/(V-b) - a/V**2
Berthelot = R*T/(V-b) - a/(T*V**2)
Dieterici_1 = R*T*sym.exp(-a/(R*T*V)) / (V-b)
Dieterici = R*T*sym.exp(-a/(R*T**s*V)) / (V-b)
Redlich = R*T/(V-b) - a/(sym.sqrt(T)*V*(V+b))

##计算各种气体临界体积、温度
gas = [perfect, vdW, Berthelot, Dieterici_1]
for p in gas:
    p_V = sym.diff(p, V, 1)
    p_V2 = sym.diff(p, V, 2)
    c = sym.simplify(sym.nonlinsolve([p_V/T, p_V2/T], V, T))
    #print(c)
"""
EmptySet
FiniteSet((3*b, 8*a/(27*R*b)))
FiniteSet((3*b, -2*sqrt(6)*sqrt(a)/(9*sqrt(R)*sqrt(b))), (3*b, 2*sqrt(6)*sqrt(a)/(9*sqrt(R)*sqrt(b))))
FiniteSet((2*b, a/(4*R*b)))
"""



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def title_labels(ax, title):
    ax.set_title(title, fontsize = 10)
    ax.set_xlabel("$V/L/mol$", fontsize = 10)
    ax.set_ylabel("$T/K$", fontsize = 10)
    ax.set_zlabel("$p/kPa$", fontsize = 10)
    

##绘制p-V-T图像
fig, axes = plt.subplots(2, 2, figsize = (10, 10), subplot_kw = {'projection': '3d'})
x = np.linspace(1, 3, 100)
y = np.linspace(150, 350, 100)
X, Y = np.meshgrid(x, y)

#R = 8.3144
#范德华气体常数（二氧化碳）
#a = 364.77, b = 0.04286
p_perfect = lambda V, T: 8.3144*T/V
p_vdW = lambda V, T: 8.3144*T/(V-0.04286) - 364.77/V**2

Z = p_perfect(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[0, 0].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[0, 0].set_zlim(0, 3000)
cb = fig.colorbar(P, ax=axes[0, 0], pad=0.1, shrink = 0.8)
title_labels(axes[0, 0], "perfect gas")

Z = p_vdW(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[0, 1].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[0, 1].set_zlim(0, 3000)
cb = fig.colorbar(P, ax=axes[0, 1], pad=0.1, shrink = 0.8)
title_labels(axes[0, 1], "vdW gas")

#Berthelot气体常数
#a = 111290, b = 0.04286
p_Berthelot = lambda V, T: 8.3144*T/(V-0.04286) - 111290/(T*V**2)

Z = p_Berthelot(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[1, 0].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[1, 0].set_zlim(0, 3000)
cb = fig.colorbar(P, ax=axes[1, 0], pad=0.1, shrink = 0.8)
title_labels(axes[1, 0], "Berthelot gas")

#Dieterici气体常数
#a = 650.42, b = 0.06429, s = 1
p_Dieterici = lambda V, T: 8.3144*T*np.exp(-650.42/(8.3144*T**1*V)) / (V-0.06429)

Z = p_Dieterici(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[1, 1].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[1, 1].set_zlim(0, 3000)
cb = fig.colorbar(P, ax=axes[1, 1], pad=0.1, shrink = 0.8)
title_labels(axes[1, 1], "Dieterici gas")

fig.tight_layout()
fig.savefig('CO2_p-V-T.png')




##绘制临界点附近的图像
fig, axes = plt.subplots(2, 2, figsize = (10, 10), subplot_kw = {'projection': '3d'})
x = np.linspace(0.1, 0.4, 100)
y = np.linspace(250, 350, 100)
X, Y = np.meshgrid(x, y)

Z = p_perfect(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[0, 0].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[0, 0].set_zlim(3000, 15000)
cb = fig.colorbar(P, ax=axes[0, 0], pad=0.1, shrink = 0.8)
title_labels(axes[0, 0], "perfect gas")

Z = p_vdW(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[0, 1].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[0, 1].set_zlim(3000, 15000)
cb = fig.colorbar(P, ax=axes[0, 1], pad=0.1, shrink = 0.8)
title_labels(axes[0, 1], "vdW gas")

Z = p_Berthelot(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[1, 0].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[1, 0].set_zlim(3000, 15000)
cb = fig.colorbar(P, ax=axes[1, 0], pad=0.1, shrink = 0.8)
title_labels(axes[1, 0], "Berthelot gas")

Z = p_Dieterici(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
P = axes[1, 1].plot_surface(X, Y, Z, linewidth = 0, rcount = 20, ccount = 20, norm = norm, cmap = "rainbow")
axes[1, 1].set_zlim(3000, 15000)
cb = fig.colorbar(P, ax=axes[1, 1], pad=0.1, shrink = 0.8)
title_labels(axes[1, 1], "Dieterici gas")

fig.tight_layout()
fig.savefig('CO2_p-V-T by low pressure.png')




##绘制临界点附近的等温线，并标出临界点
fig, axes = plt.subplots(2, 2, figsize = (12, 10), subplot_kw = {'projection': '3d'})
x = np.linspace(0.07, 0.3, 100)
y = np.linspace(250, 350, 100)
X, Y = np.meshgrid(x, y)

Z = p_perfect(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
cset = axes[0, 0].contour(X, Y, Z, zdir='y', levels = 20, norm=norm, cmap="rainbow")
axes[0, 0].set_zlim(3000, 15000)
title_labels(axes[0, 0], "perfect gas")

Z = p_vdW(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
cset = axes[0, 1].contour(X, Y, Z, zdir='y', levels = 20, norm=norm, cmap="rainbow")
axes[0, 1].set_zlim(3000, 15000)
R = 8.3144; a = 364.77; b = 0.04286
v = 3*b; t = 8*a/(27*R*b)
#print(v, t, p_vdW(v, t))
"""0.12858 303.2929078983176 7354.4638165824"""
axes[0, 1].text(v, t, p_vdW(v, t), 'C point')
title_labels(axes[0, 1], "vdW gas")

Z = p_Berthelot(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
cset = axes[1, 0].contour(X, Y, Z, zdir='y', levels = 20, norm=norm, cmap="rainbow")
axes[1, 0].set_zlim(3000, 15000)
a = 111290; b = 0.04286
v = 3*b; t = 2*np.sqrt(6)*np.sqrt(a)/(9*np.sqrt(R)*np.sqrt(b))
#print(v, t, p_Berthelot(v, t))
"""0.12858 304.19329849263966 7376.29713307046"""
axes[1, 0].text(v, t, p_Berthelot(v, t), 'C point')
title_labels(axes[1, 0], "Berthelot gas")

Z = p_Dieterici(X, Y)
norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
cset = axes[1, 1].contour(X, Y, Z, zdir='y', levels = 20, norm=norm, cmap="rainbow")
axes[1, 1].set_zlim(3000, 10000)
a = 650.42; b = 0.06429
v = 2*b; t = a/(4*R*b)
#print(v, t, p_Dieterici(v, t))
"""0.12858 304.2002423439793 5324.245596942433"""
axes[1, 1].text(v, t, p_Dieterici(v, t), 'C point')
title_labels(axes[1, 1], "Dieterici gas")

fig.tight_layout()
fig.savefig('CO2 critical point.png')




##焦汤效应与转换温度曲线
from scipy import optimize as optm

V, T = sym.symbols("V, T", positive = True)
a, b, R, s = sym.symbols("a, b, R, s", positive = True)
gas = [vdW, Berthelot, Dieterici]
for p in gas:
    p_V = sym.diff(p, V, 1)
    p_T = sym.diff(p, T, 1)
    expr = -T*p_T/p_V - V
    V_t = sym.solveset(expr, V)
    #print(V_t)
"""
Complement(FiniteSet(0, sqrt(2)*sqrt(R)*sqrt(T)*sqrt(a)*b**(3/2)/(R*T*b - 2*a) + 2*a*b/(-R*T*b + 2*a), -sqrt(2)*sqrt(R)*sqrt(T)*sqrt(a)*b**(3/2)/(R*T*b - 2*a) + 2*a*b/(-R*T*b + 2*a)), FiniteSet(-(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))/(3*(sqrt(-4*(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))**3 + (-54*a*b**2/(R*T) + 72*a**2*b/(R**2*T**2) - 16*a**3/(R**3*T**3))**2)/2 - 27*a*b**2/(R*T) + 36*a**2*b/(R**2*T**2) - 8*a**3/(R**3*T**3))**(1/3)) - (sqrt(-4*(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))**3 + (-54*a*b**2/(R*T) + 72*a**2*b/(R**2*T**2) - 16*a**3/(R**3*T**3))**2)/2 - 27*a*b**2/(R*T) + 36*a**2*b/(R**2*T**2) - 8*a**3/(R**3*T**3))**(1/3)/3 + 2*a/(3*R*T), -(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))/(3*(-1/2 - sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))**3 + (-54*a*b**2/(R*T) + 72*a**2*b/(R**2*T**2) - 16*a**3/(R**3*T**3))**2)/2 - 27*a*b**2/(R*T) + 36*a**2*b/(R**2*T**2) - 8*a**3/(R**3*T**3))**(1/3)) - (-1/2 - sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))**3 + (-54*a*b**2/(R*T) + 72*a**2*b/(R**2*T**2) - 16*a**3/(R**3*T**3))**2)/2 - 27*a*b**2/(R*T) + 36*a**2*b/(R**2*T**2) - 8*a**3/(R**3*T**3))**(1/3)/3 + 2*a/(3*R*T), -(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))/(3*(-1/2 + sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))**3 + (-54*a*b**2/(R*T) + 72*a**2*b/(R**2*T**2) - 16*a**3/(R**3*T**3))**2)/2 - 27*a*b**2/(R*T) + 36*a**2*b/(R**2*T**2) - 8*a**3/(R**3*T**3))**(1/3)) - (-1/2 + sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T) + 4*a**2/(R**2*T**2))**3 + (-54*a*b**2/(R*T) + 72*a**2*b/(R**2*T**2) - 16*a**3/(R**3*T**3))**2)/2 - 27*a*b**2/(R*T) + 36*a**2*b/(R**2*T**2) - 8*a**3/(R**3*T**3))**(1/3)/3 + 2*a/(3*R*T)))
Complement(FiniteSet(0, sqrt(3)*sqrt(R)*T*sqrt(a)*b**(3/2)/(R*T**2*b - 3*a) + 3*a*b/(-R*T**2*b + 3*a), -sqrt(3)*sqrt(R)*T*sqrt(a)*b**(3/2)/(R*T**2*b - 3*a) + 3*a*b/(-R*T**2*b + 3*a)), FiniteSet(-(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))/(3*(sqrt(-4*(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))**3 + (-54*a*b**2/(R*T**2) + 72*a**2*b/(R**2*T**4) - 16*a**3/(R**3*T**6))**2)/2 - 27*a*b**2/(R*T**2) + 36*a**2*b/(R**2*T**4) - 8*a**3/(R**3*T**6))**(1/3)) - (sqrt(-4*(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))**3 + (-54*a*b**2/(R*T**2) + 72*a**2*b/(R**2*T**4) - 16*a**3/(R**3*T**6))**2)/2 - 27*a*b**2/(R*T**2) + 36*a**2*b/(R**2*T**4) - 8*a**3/(R**3*T**6))**(1/3)/3 + 2*a/(3*R*T**2), -(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))/(3*(-1/2 - sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))**3 + (-54*a*b**2/(R*T**2) + 72*a**2*b/(R**2*T**4) - 16*a**3/(R**3*T**6))**2)/2 - 27*a*b**2/(R*T**2) + 36*a**2*b/(R**2*T**4) - 8*a**3/(R**3*T**6))**(1/3)) - (-1/2 - sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))**3 + (-54*a*b**2/(R*T**2) + 72*a**2*b/(R**2*T**4) - 16*a**3/(R**3*T**6))**2)/2 - 27*a*b**2/(R*T**2) + 36*a**2*b/(R**2*T**4) - 8*a**3/(R**3*T**6))**(1/3)/3 + 2*a/(3*R*T**2), -(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))/(3*(-1/2 + sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))**3 + (-54*a*b**2/(R*T**2) + 72*a**2*b/(R**2*T**4) - 16*a**3/(R**3*T**6))**2)/2 - 27*a*b**2/(R*T**2) + 36*a**2*b/(R**2*T**4) - 8*a**3/(R**3*T**6))**(1/3)) - (-1/2 + sqrt(3)*I/2)*(sqrt(-4*(-12*a*b/(R*T**2) + 4*a**2/(R**2*T**4))**3 + (-54*a*b**2/(R*T**2) + 72*a**2*b/(R**2*T**4) - 16*a**3/(R**3*T**6))**2)/2 - 27*a*b**2/(R*T**2) + 36*a**2*b/(R**2*T**4) - 8*a**3/(R**3*T**6))**(1/3)/3 + 2*a/(3*R*T**2)))
Complement(FiniteSet(0, a*b*(s + 1)/(-R*T**s*b + a*s + a)), FiniteSet(-T**(-s)*sqrt(a)*sqrt(-4*R*T**s*b + a)/(2*R) + T**(-s)*a/(2*R), T**(-s)*sqrt(a)*sqrt(-4*R*T**s*b + a)/(2*R) + T**(-s)*a/(2*R)))
"""
_Vt_vdW = (sym.sqrt(2*a*R*T)*b**(3/2) + 2*a*b)/(-R*T*b + 2*a)
_Vt_Berthelot = (sym.sqrt(3*a*R)*T*b**(3/2) + 3*a*b)/(-R*T**2*b + 3*a)
_Vt_Dieterici = a*b*(s + 1)/(-R*T**s*b + a*s + a)

p_vdW = lambda V, T: 8.3144*T/(V-0.04286) - 364.77/V**2
p_Berthelot = lambda V, T: 8.3144*T/(V-0.04286) - 111290/(T*V**2)
p_Dieterici_1 = lambda V, T: 8.3144*T*np.exp(-650.42/(8.3144*T**1*V)) / (V-0.06429)

Vt_vdW = lambda T: (np.sqrt(2*364.77*8.3144*T)*0.04286**(3/2) + 2*364.77*0.04286)/(-8.3144*T*0.04286 + 2*364.77)
Vt_Berthelot = lambda T: (np.sqrt(3*111290*8.3144)*T*0.04286**(3/2) + 3*111290*0.04286)/(-8.3144*T**2*0.04286 + 3*111290)
Vt_Dieterici_1 = lambda T: 650.42*0.06429*(1 + 1)/(-8.3144*T**1*0.06429 + 650.42*1 + 650.42)

x = np.linspace(0.1, 3000, 100)
fig, ax = plt.subplots(figsize = (7, 4))
ax.plot(x, p_vdW(Vt_vdW(x), x), color = 'g', linestyle = '-', label = r'$vdW gas$')
ax.plot(x, p_Berthelot(Vt_Berthelot(x), x), color = 'b', linestyle = '-', label = r'$Berthelot gas$')
ax.plot(x, p_Dieterici_1(Vt_Dieterici_1(x), x), color = 'r', linestyle = '-', label = r'$Dieterici gas$')
ax.set_ylim(0, 1e5)
ax.grid()
ax.set_xlabel('$T/K$'); ax.set_ylabel('$p/kPa$')
ax.set_title('Inversion Curve')
ax.legend()
fig.savefig('CO2 inversion curve.png')

#求最大转换温度和最大转换压强
_pt = vdW.subs(V, _Vt_vdW)
_pt_vdW = _pt.subs([(R, 8.3144), (a, 364.77), (b, 0.04286)])
_pt = Berthelot.subs(V, _Vt_Berthelot)
_pt_Berthelot = _pt.subs([(R, 8.3144), (a, 111290), (b, 0.04286)])
_pt = Dieterici.subs(V, _Vt_Dieterici)
_pt_Dieterici = _pt.subs([(R, 8.3144), (a, 650.42), (b, 0.06429), (s, 1)])
pts = [_pt_vdW, _pt_Berthelot, _pt_Dieterici]
for _pt in pts:
    pt = sym.lambdify(T, _pt, 'numpy')
    pt_T = sym.lambdify(T, sym.diff(_pt, T, 1), 'numpy')
    Tt_max = optm.bisect(pt, 500, 2500)
    #print(Tt_max)
    """
    2047.2271283136402
    967.93714870042
    2433.6019387518318
    """
    Tt_pt_max = optm.bisect(pt_T, 275, 2500)
    #print(Tt_pt_max, pt(Tt_pt_max))
    """
    909.8787236949512 66190.1743492415
    483.968574350208 62589.95667409242
    1216.8009693759209 95446.45319844982
    """

m = 1
p_Dieterici = lambda V, T: 8.3144*T*np.exp(-650.42/(8.3144*T**m*V)) / (V-0.06429)
Vt_Dieterici = lambda T: 650.42*0.06429*(m + 1)/(-8.3144*T**m*0.06429 + 650.42*m + 650.42)
fig, ax = plt.subplots(figsize = (7, 4))
ax.plot(x, p_Dieterici(Vt_Dieterici(x), x), color = 'g', linestyle = '-', label = r'$s=1.0$')
m = 1.1
ax.plot(x, p_Dieterici(Vt_Dieterici(x), x), color = 'b', linestyle = '-', label = r'$s=1.1$')
m = 1.2
ax.plot(x, p_Dieterici(Vt_Dieterici(x), x), color = 'r', linestyle = '-', label = r'$s=1.2$')
m = 1.3
ax.plot(x, p_Dieterici(Vt_Dieterici(x), x), color = 'y', linestyle = '-', label = r'$s=1.3$')
m = 1.4
ax.plot(x, p_Dieterici(Vt_Dieterici(x), x), color = 'c', linestyle = '-', label = r'$s=1.4$')
ax.set_ylim(0, 1e5)
ax.grid()
ax.set_xlabel('$T/K$'); ax.set_ylabel('$p/kPa$')
ax.set_title('Inversion Curve')
ax.legend()
fig.savefig('CO2 Dieterici inversion curve by s.png')