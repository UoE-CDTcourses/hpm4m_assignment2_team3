import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
p = 3.1415926536
script_location = os.path.dirname(__file__)
results_para = pd.read_csv(os.path.join(script_location, 'para_heat.out'), header = 0)
results_serial = pd.read_csv(os.path.join(script_location, 'serial_heat.out'), header = None).to_numpy()

x_p = results_para['x'].to_numpy()
U_p = results_para['U'].to_numpy()
x_s = results_serial[0,:]
U_s = results_serial[1,:]
T = results_serial[2,0]


U_true = np.exp(-4*np.power(np.pi,2)*T)*np.sin(2*np.pi*x_s) + 2*np.exp(-25*np.power(np.pi,2)*T)*np.sin(5*np.pi*x_s) + 3*np.exp(-400*np.power(np.pi,2)*T)*np.sin(20*np.pi*x_s)

fig1,ax1 = plt.subplots(1,1)
ax1.plot(x_s,U_s, label = 'Numerical serial solution')
ax1.plot(x_s,U_true, label = 'Exact solution')
ax1.plot(x_p,U_p, label = 'Numerical parallel solution')
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax1.legend()

fig2,ax2 = plt.subplots(1,1)
ax2.plot(x_p,U_p-U_s, label = 'Difference between parallel and serial result')
ax2.set_xlabel('x')
ax2.set_ylabel('Difference')
ax2.legend()

