#In this code, we are solving a CSTR reactor problem. Follow the comments to know the details.
#lets begin!
#first_import libraries. 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# then define mixing model of CSTR
def reactor(x, t, q, qf, Caf, Tf):
    # Inputs= (4):
    # qf= I/L VFR (L/min)
    # q= O/L VFR (L/min)
    # Caf= Feed conc. (mol/L)
    # Tf= Feed temp (K)

    # States (3):
    # Volume (L)
    V = x[0]
    # Conc. of A (mol/L)
    Ca = x[1]
    # Temp (K)
    T = x[2]

    # Parameters
    # Reactions
    rA = 0.0

    # Mass Balance volume derivation
    dVdt = qf - q

    # Species balance: concentration derivation
    # Chain rule: d(V*Ca)/dt= Ca * dV/dt+ V* dCa/dt
    dCadt = (qf * Caf - q * Ca) / V - rA - (Ca * dVdt / V)

    # Energy balance temperature derivative
    # Chain rule: d(V*T)/dt =T * dV/dt + V + dT/dt
    dTdt = (qf * Tf - q * T) / V - (T * dVdt / V)

    # Return derivatives
    return [dVdt, dCadt, dTdt]


# Initial Conditions for the States 
#(For a certain CSTR question, assume following initial condtions)
V0 = 1.0  # in liter
Ca0 = 0.0  # mol/L
T0 = 350  # in K
y0 = [V0, Ca0, T0]

# Time intervals (min)
t = np.linspace(0, 10, 100)

# I/L VFR (L/min)
qf = np.ones(len(t)) * 5.2
qf[50:] = 5.1

# O/L VFR (L/min)
q = np.ones(len(t)) * 1.0

# Feed conc (mol/L)
Caf = np.ones(len(t)) * 1.0
Caf[30:] = 0.5

# Feed temp (K)
Tf = np.ones(len(t)) * 300.0
Tf[70:] = 325.0

# Storage for results
V = np.ones(len(t)) * V0
Ca = np.ones(len(t)) * Ca0
T = np.ones(len(t)) * T0

# loop through each time step
for i in range(len(t) - 1):
    # Simulate
    inputs = (q[i], qf[i], Caf[i], Tf[i])
    ts = [t[i], t[i + 1]]
    y = odeint(reactor, y0, ts, args=inputs)
    # store results
    V[i + 1] = y[-1][0]
    Ca[i + 1] = y[-1][1]
    T[i + 1] = y[-1][2]
    # Adjust intial conditon for the next loop
    y0 = y[-1]

# Construct results and save data file
data = np.vstack((t, qf, q, Tf, Caf, V, Ca, T))  # Vertical stack
data = data.T  # transpose data
np.savetxt('data.txt', data, delimiter=',')

# Plot the inputs and results
plt.figure()

# feed flowrate data plot
plt.subplot(3, 2, 1)
plt.plot(t, qf, 'b--', linewidth=3)
plt.plot(t, q, 'b:', linewidth=3)
plt.ylabel("Flow rates (L/min)")
plt.legend(['I/L', 'O/L'], loc='best')

# Concentration feed flowrate data plot
plt.subplot(3, 2, 3)
plt.plot(t, Caf, 'r--', linewidth=3)
plt.ylabel('Caf (mol/l)')
plt.legend(['Feed Concentration'], loc='best')

# Temperature feed data plot
plt.subplot(3, 2, 5)
plt.plot(t, Tf, 'k--', linewidth=3)
plt.ylabel('Tf (K)')
plt.legend(['Feed Temperature'], loc='best')
plt.xlabel('Time(minute)')

# Volume data plot
plt.subplot(3, 2, 2)
plt.plot(t, V, 'b-', linewidth=3)
plt.ylabel('Volume(L)')
plt.legend(['Volume'], loc='best')

# Concentration outlet data plot
plt.subplot(3, 2, 4)
plt.plot(t, Ca, 'r-', linewidth=3)
plt.ylabel('Ca (mol/l)')
plt.legend(['Concentration'], loc='best')

# Temperature outlet data plot
plt.subplot(3, 2, 6)
plt.plot(t, T, 'k-', linewidth=3)
plt.ylabel('T (K)')
plt.legend(['Temperature'], loc='best')
plt.xlabel('Time (min)')

plt.show()

