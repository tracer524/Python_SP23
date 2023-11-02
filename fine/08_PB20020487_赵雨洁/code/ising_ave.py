# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
def sol_s(T,h):#T,h表示和Tc，hc相对大小
    s1=0
    s0=1
    while(np.abs(s1-s0)>1e-20):
        s0=s1
        s1=np.tanh((h+s0)/T)#通过迭代法求解

    return s1


# %%
T=np.linspace(0.01,5,100)
h=np.linspace(0.01,5,100)
st=[]
sh=[]
for i in T:
    st.append(sol_s(i,1))
for q in h:
    sh.append(sol_s(1,q))


# %%
plt.figure(figsize=(7, 7))
plt.title("T,h-ave_s")
plt.ylabel("ave_s")
plt.xlabel("T")
plt.plot(T,st,label='T_diffrential')
plt.plot(T,sh,label='h_diffrential')
plt.legend()
plt.show()

# %%



