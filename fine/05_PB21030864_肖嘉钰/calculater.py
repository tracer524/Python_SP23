p_c = float(input("Please input the critical pressure/kPa\n"))
T_c = float(input("Please input the critical temperature/K\n"))

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optm

V, T = sym.symbols("V, T", positive = True)
a, b, R = sym.symbols("a, b, R", positive = True)
_pc, _Tc = sym.symbols("pc, Tc", positive = True)

p = dict({})
Vc = dict({}); Tc = dict({}); pc = dict({})
Vt = dict({}); pt = dict({})

p['vdW'] = R*T/(V-b) - a/V**2
Vc['vdW'] = 3*b; Tc['vdW'] = 8*a/(27*R*b)
pc['vdW'] = p['vdW'].subs([(V, Vc['vdW']), (T, Tc['vdW'])]) #a/(27*b**2)
Vt['vdW'] = (sym.sqrt(2*a*R*T)*b**(3/2) + 2*a*b)/(-R*T*b + 2*a)
pt['vdW'] = p['vdW'].subs(V, Vt['vdW']) 
#R*T/(-b + (sqrt(2)*sqrt(R)*sqrt(T)*sqrt(a)*b**1.5 + 2*a*b)/(-R*T*b + 2*a)) - a*(-R*T*b + 2*a)**2/(sqrt(2)*sqrt(R)*sqrt(T)*sqrt(a)*b**1.5 + 2*a*b)**2

p['Ber'] = R*T/(V-b) - a/(T*V**2)
Vc['Ber'] = 3*b; Tc['Ber'] = 2*sym.sqrt(6*a)/(9*sym.sqrt(R*b))
pc['Ber'] = p['Ber'].subs([(V, Vc['Ber']), (T, Tc['Ber'])]) #sqrt(6)*sqrt(R)*sqrt(a)/(36*b**(3/2))
Vt['Ber'] = (sym.sqrt(3*a*R)*T*b**(3/2) + 3*a*b)/(-R*T**2*b + 3*a)
pt['Ber'] = p['Ber'].subs(V, Vt['Ber']) 
#R*T/(-b + (sqrt(3)*sqrt(R)*T*sqrt(a)*b**1.5 + 3*a*b)/(-R*T**2*b + 3*a)) - a*(-R*T**2*b + 3*a)**2/(T*(sqrt(3)*sqrt(R)*T*sqrt(a)*b**1.5 + 3*a*b)**2)

p['Die'] = R*T*sym.exp(-a/(R*T*V)) / (V-b)
Vc['Die'] = 2*b; Tc['Die'] = a/(4*R*b)
pc['Die'] = p['Die'].subs([(V, Vc['Die']), (T, Tc['Die'])]) #a*exp(-2)/(4*b**2)
Vt['Die'] = a*b*2/(-R*T*b + a*2)
pt['Die'] = p['Die'].subs(V, Vt['Die']) 
#R*T*exp(-(-R*T*b + 2*a)/(2*R*T*b))/(2*a*b/(-R*T*b + 2*a) - b)

res = []
res.append(sym.nonlinsolve([pc['vdW']-_pc, Tc['vdW']-_Tc], a, b))
res.append(sym.nonlinsolve([pc['Ber']-_pc, Tc['Ber']-_Tc], a, b))
res.append(sym.nonlinsolve([pc['Die']-_pc, Tc['Die']-_Tc], a, b))
#print(res)
"""
[FiniteSet((27*R**2*Tc**2/(64*pc), R*Tc/(8*pc))), 
 FiniteSet((27*R**2*Tc**3/(64*pc), R*Tc/(8*pc))), 
 FiniteSet((4*R**2*Tc**2*exp(-2)/pc, R*Tc*exp(-2)/pc))]
"""

_a = dict({}); _b = dict({})

_a['vdW'] = 27*R**2*_Tc**2/(64*_pc); _b['vdW'] = R*_Tc/(8*_pc)
_a['Ber'] = 27*R**2*_Tc**3/(64*_pc); _b['Ber'] = R*_Tc/(8*_pc)
_a['Die'] = 4*R**2*_Tc**2*np.exp(-2)/_pc; _b['Die'] = R*_Tc*np.exp(-2)/_pc



def get_parameter(p_c, T_c):
    a_ = dict({}); b_ = dict({}); Vc_ = dict({})
    for temp in ['vdW', 'Ber', 'Die']:
        a_[temp] = _a[temp].subs([(R, 8.3144), (_Tc, T_c), (_pc, p_c)])
        b_[temp] = _b[temp].subs([(R, 8.3144), (_Tc, T_c), (_pc, p_c)])
        Vc_[temp] = Vc[temp].subs([(R, 8.3144), (a, a_[temp]), (b, b_[temp])])
    return a_, b_, Vc_

A = get_parameter(p_c, T_c)[0]
B = get_parameter(p_c, T_c)[1]
V_c = get_parameter(p_c, T_c)[2]

def get_inversion_curve(A, B, tmp, L, H):
    pt_ = dict({}); _pt = dict({})
    for temp in ['vdW', 'Ber', 'Die']:
        pt_[temp] = pt[temp].subs([(a, A[temp]), (b, B[temp]), (R, 8.3144)])
        _pt[temp] = sym.lambdify(T, pt_[temp], 'numpy')
    x = np.linspace(0.1, L, 100)
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.plot(x, _pt[tmp](x), color = 'g', linestyle = '-', label = tmp)
    ax.set_ylim(0, 1e5*H)
    ax.grid()
    ax.set_xlabel('$T/K$'); ax.set_ylabel('$p/kPa$')
    ax.set_title('Inversion Curve')
    ax.legend()
    fig.savefig('inversion curve by calculater.png')
    print("Picture is saved!")
    return 0

def get_inversion_max(A, B, tmp, l, h):
    pt_ = dict({}); _pt = dict({}); pt_T = dict({}); _pt_T = dict({})
    for temp in ['vdW', 'Ber', 'Die']:
        pt_[temp] = pt[temp].subs([(a, A[temp]), (b, B[temp]), (R, 8.3144)])
        _pt[temp] = sym.lambdify(T, pt_[temp], 'numpy')
        pt_T[temp] = sym.diff(pt_[temp], T, 1)
        _pt_T[temp] = sym.lambdify(T, pt_T[temp], 'numpy')
    p_inv = _pt[tmp]; p_inv_T = _pt_T[tmp]
    Tt_max = optm.bisect(p_inv, l, h)
    Tt_p_max = optm.bisect(p_inv_T, l, h)
    pt_max = p_inv(Tt_p_max)
    return Tt_max, pt_max
  
t = 1
while True:
    t = int(input("(Input 0 to quit)\nPlease choose the type of gas:\n1.vdW   2.Berthelot   3.Dieterici\n"))
    if t == 0: break
    elif t == 3: tmp = 'Die'
    elif t == 2: tmp = 'Ber'
    else: tmp = 'vdW'
    print("a: %f"%A[tmp])
    print("b: %f"%B[tmp])
    print("V_c: %f"%V_c[tmp])
    L = 3000; H = 1.5
    get_inversion_curve(A, B, tmp, L, H)
    t1 = int(input("Do you need to change the Ub of p and T in the picture?\n0.No   1.Yes\n"))
    while t1 == 1:
        L = float(input("Please input the Ub of the T/K\n"))
        H = float(input("Please input the Ub of the p/100MPa\n"))
        get_inversion_curve(A, B, L, H)
        t1 = int(input("Do you need to change the Ub of p and T in the picture?\n0.No   1.Yes\n"))
    l = float(input("Please input the Lb of the inversion_max\n"))
    h = float(input("Please input the Ub of the inversion_max\n"))
    result = get_inversion_max(A, B, tmp, l, h)
    print("T_inv_max: %f"%result[0])
    print("p_inv_max: %f"%result[1])
