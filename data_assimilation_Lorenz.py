#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Data assimilation on Lorenz model')


# In[47]:


# load librairies
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# In[108]:


# model parameters
s = 10.0
r = 28.0
b = 8.0/3.0


# In[109]:


# true initial condition
u0true = np.array([-4.62, -6.61, 17.94])


# In[110]:


# numerical model parameters
T = 4.0
dt = 0.001
nt = int(T/dt)+1
# frequency of observations
fobs = 100
dtobs = fobs*dt
ntobs = int(T/dtobs)+1


# In[111]:


# numerical model 
def lorenz(u0):
    u = np.zeros((nt,3))
    u[0,:] = u0
    for i in range(1,nt):
        u[i,0] = u[i-1,0] + dt*(s*(u[i-1,1]-u[i-1,0]))
        u[i,1] = u[i-1,1] + dt*(r*u[i-1,0]-u[i-1,1]-u[i-1,0]*u[i-1,2])
        u[i,2] = u[i-1,2] + dt*(u[i-1,0]*u[i-1,1]-b*u[i-1,2])
    return u


# In[112]:


utrue = lorenz(u0true)


# In[113]:


# plot solution
fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),utrue[:,0])
plt.legend(['True solution'])


# In[23]:


# background solution
u0b = np.array([-4.0, -6.0, 17.0])
ub = lorenz(u0b)
# plot solutions
fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),utrue[:,0],np.arange(0,T+0.5*dt,dt),ub[:,0])
plt.legend(['True solution','Background solution'])


# In[25]:


# observations
# plot solutions and observations
fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),utrue[:,0],np.arange(0,T+0.5*dt,dt),ub[:,0],
         np.arange(0,T+0.5*dt,dtobs),utrue[0:nt:fobs,0],'ro')
plt.legend(['True solution','Background solution','Observations'])


# In[50]:


# add noise
noiselevel = 1.0
noise = np.random.randn(nt,3)
uobs = utrue + noiselevel*noise
fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),utrue[:,0],np.arange(0,T+0.5*dt,dt),ub[:,0],
         np.arange(0,T+0.5*dt,dtobs),uobs[0:nt:fobs,0],'ro')
plt.legend(['True solution','Background solution','Observations'])


# In[51]:


# covariance matrices
R = np.identity(3)
# more noise on observations: R = 10*np.identity(3)
# less noise on observations: R = 0.1*np.identity(3)
B = np.identity(3)
# inverse covariance matrices
Rinv = np.linalg.inv(R)
Binv = np.linalg.inv(B)


# In[52]:


print('Coffee break')


# In[53]:


print('End of coffee break')


# In[54]:


print('4D-VAR')


# In[55]:


# cost function
def cost(u0):
    # relative weight Jo vs Jb
    wb = 1.0
    wo = 1/ntobs
    u = lorenz(u0)
    # Jb
    Jb = 0.5*np.dot(u0-u0b,Binv.dot(u0-u0b))
    # Jo
    Jo = 0.0
    for i in range(0,nt,fobs):
        Jo = Jo + 0.5*np.dot(u[i,:]-uobs[i,:],Rinv.dot(u[i,:]-uobs[i,:]))
    return (wb*Jb+wo*Jo)


# In[56]:


cost(u0b)


# In[57]:


cost(u0true)


# In[62]:


# minimize cost
res = minimize(cost,u0b,method='Nelder-Mead',options={'maxiter': 500})


# In[63]:


res


# In[83]:


# gradient of the cost function
def gradient(u0):
    # relative weight Jo vs Jb
    wb = 1.0
    wo = 1/ntobs
    # direct resolution
    u = lorenz(u0)
    # adjoint resolution
    uadj = lorenzadj(u)
    # gradient
    g = wb*Binv.dot(u0-u0b)+wo*uadj[0,:]
    return g


# In[84]:


# adjoint model
def lorenzadj(u):
    uadj = np.zeros((nt,3))
    uadj[nt-1,:] = 0
    # add observation forcing
    if np.mod(nt-1,fobs)==0: #we are at an observation time
        uadj[nt-1,:] = uadj[nt-1,:] + Rinv.dot(u[nt-1,:]-uobs[nt-1,:])
    for i in range(nt-1,0,-1):
        uadj[i-1,0] = uadj[i,0] - dt*(s*uadj[i,0]-r*uadj[i,1]+uadj[i,1]*u[i,2]-uadj[i,2]*u[i,1])
        uadj[i-1,1] = uadj[i,1] - dt*(-s*uadj[i,0]+uadj[i,1]-uadj[i,2]*u[i,0])
        uadj[i-1,2] = uadj[i,2] - dt*(b*uadj[i,2]+uadj[i,1]*u[i,0])
        # add observation forcing
        if np.mod(i-1,fobs)==0: #we are at an observation time
            uadj[i-1,:] = uadj[i-1,:] + Rinv.dot(u[i-1,:]-uobs[i-1,:])
    return uadj


# In[85]:


gradient(u0b)


# In[86]:


# test gradient
(cost(u0b+np.array([0,0,0.0001]))-cost(u0b))/0.0001


# In[88]:


# minimize cost with the gradient
res = minimize(cost,u0b,method='BFGS',jac=gradient,options={'maxiter': 500})


# In[89]:


res


# In[91]:


u0opt = res.x
uopt = lorenz(u0opt)
fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),utrue[:,0],np.arange(0,T+0.5*dt,dt),ub[:,0],
         np.arange(0,T+0.5*dt,dtobs),uobs[0:nt:fobs,0],'ro',
         np.arange(0,T+0.5*dt,dt),uopt[:,0])
plt.legend(['True solution','Background solution','Observations','4DVar solution'])


# In[90]:


print('Kalman filter')


# In[92]:


# Kalman solution
ukf = np.zeros((nt,3))
# initialisation
ukf[0,:] = u0b
# time loop ASSUMING NO OBSERVATIONS => NO ANALYSIS
for i in range(1,nt):
    # forecast step
    ukf[i,0] = ukf[i-1,0] + dt*(-s*ukf[i-1,0]+s*ukf[i-1,1])
    ukf[i,1] = ukf[i-1,1] + dt*((r-ub[i-1,2])*ukf[i-1,0]-ukf[i-1,1]-ub[i-1,0]*ukf[i-1,2])
    ukf[i,2] = ukf[i-1,2] + dt*(ub[i-1,1]*ukf[i-1,0]+ub[i-1,0]*ukf[i-1,1]-b*ukf[i-1,2])


# In[93]:


fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),ub[:,0],np.arange(0,T+0.5*dt,dt),ukf[:,0])
plt.legend(['Background solution','TLM solution'])


# In[100]:


# Kalman solution
ukf = np.zeros((nt,3))
# initialisation
ukf[0,:] = u0b
Pf = B
# time loop
for i in range(1,nt):
    # TLM matrix
    A = np.zeros((3,3))
    A[0,0] = -s
    A[0,1] = s
    A[1,0] = r-ub[i-1,2]
    A[1,1] = -1
    A[1,2] = -ub[i-1,0]
    A[2,0] = ub[i-1,1]
    A[2,1] = ub[i-1,0]
    A[2,2] = -b
    M = np.identity(3)+dt*A
    # forecast step
    #ukf[i,0] = ukf[i-1,0] + dt*(-s*ukf[i-1,0]+s*ukf[i-1,1])
    #ukf[i,1] = ukf[i-1,1] + dt*((r-ub[i-1,2])*ukf[i-1,0]-ukf[i-1,1]-ub[i-1,0]*ukf[i-1,2])
    #ukf[i,2] = ukf[i-1,2] + dt*(ub[i-1,1]*ukf[i-1,0]+ub[i-1,0]*ukf[i-1,1]-b*ukf[i-1,2])
    ukf[i,:] = M.dot(ukf[i-1,:])
    # update covariance matrix
    Pf = np.matmul(M,np.matmul(Pf,np.transpose(M)))
    # if available observations
    if np.mod(i,fobs)==0:
        # define Kalman matrix
        K = np.matmul(Pf,np.linalg.inv(Pf+R))
        # analysis step
        ua = ukf[i,:] + K.dot(uobs[i,:]-ukf[i,:])
        ukf[i,:] = ua
        # update covariance matrix
        Pf = np.multiply((np.identity(3)-K),Pf)


# In[101]:


fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),ub[:,0],np.arange(0,T+0.5*dt,dt),ukf[:,0])
plt.legend(['Background solution','Kalman solution'])


# In[102]:


print('Extended Kalman filter')


# In[104]:


# Extended Kalman solution
uekf = np.zeros((nt,3))
# initialisation
uekf[0,:] = u0b
Pf = B
# time loop
for i in range(1,nt):
    # TLM matrix
    A = np.zeros((3,3))
    A[0,0] = -s
    A[0,1] = s
    A[1,0] = r-ub[i-1,2]
    A[1,1] = -1
    A[1,2] = -ub[i-1,0]
    A[2,0] = ub[i-1,1]
    A[2,1] = ub[i-1,0]
    A[2,2] = -b
    M = np.identity(3)+dt*A
    # forecast step WITH THE NONLINEAR MODEL
    uekf[i,0] = uekf[i-1,0] + dt*(s*(uekf[i-1,1]-uekf[i-1,0]))
    uekf[i,1] = uekf[i-1,1] + dt*(r*uekf[i-1,0]-uekf[i-1,1]-uekf[i-1,0]*uekf[i-1,2])
    uekf[i,2] = uekf[i-1,2] + dt*(uekf[i-1,0]*uekf[i-1,1]-b*uekf[i-1,2])
    # update covariance matrix STILL WITH THE LINEAR MODEL
    Pf = np.matmul(M,np.matmul(Pf,np.transpose(M)))
    # if available observations
    if np.mod(i,fobs)==0:
        # define Kalman matrix
        K = np.matmul(Pf,np.linalg.inv(Pf+R))
        # analysis step
        ua = uekf[i,:] + K.dot(uobs[i,:]-uekf[i,:])
        uekf[i,:] = ua
        # update covariance matrix
        Pf = np.multiply((np.identity(3)-K),Pf)


# In[106]:


fig = plt.figure()
plt.plot(np.arange(0,T+0.5*dt,dt),ub[:,0],np.arange(0,T+0.5*dt,dt),utrue[:,0],
         np.arange(0,T+0.5*dt,dt),ukf[:,0],np.arange(0,T+0.5*dt,dt),uekf[:,0])
plt.legend(['Background solution','True solution','Kalman solution','Extended Kalman'])


# In[107]:


print('coffee break, again!')


# In[129]:


# new numerical model in which parameters are input
def lorenznew(u0,s,r,b):
    u = np.zeros((nt,3))
    u[0,:] = u0
    for i in range(1,nt):
        u[i,0] = u[i-1,0] + dt*(s*(u[i-1,1]-u[i-1,0]))
        u[i,1] = u[i-1,1] + dt*(r*u[i-1,0]-u[i-1,1]-u[i-1,0]*u[i-1,2])
        u[i,2] = u[i-1,2] + dt*(u[i-1,0]*u[i-1,1]-b*u[i-1,2])
    return u


# In[130]:


# forget also true parameters
# => background parameters
sb = 10.5
rb = 28.5
bb = 3.0
# control vector that contains all the values to be identified
Xb = np.zeros(6)
Xb[0:3] = u0true
Xb[3] = sb
Xb[4] = rb
Xb[5] = bb
# covariance matrix on background parameters
Bp = np.identity(3)
Bpinv = np.linalg.inv(Bp)


# In[131]:


# new cost function
def costnew(X):
    u0 = X[0:3]
    s = X[3]
    r = X[4]
    b = X[5]
    # relative weight Jo vs Jb
    wb = 0.0
    wo = 1/ntobs
    u = lorenznew(u0,s,r,b)
    # Jb state
    Jb = 0.5*np.dot(u0-u0b,Binv.dot(u0-u0b))
    # Jb param
    Jb = Jb + 0.5*np.dot(X[3:6]-Xb[3:6],Bpinv.dot(X[3:6]-Xb[3:6]))
    # Jo
    Jo = 0.0
    for i in range(0,nt,fobs):
        Jo = Jo + 0.5*np.dot(u[i,:]-uobs[i,:],Rinv.dot(u[i,:]-uobs[i,:]))
    return (wb*Jb+wo*Jo)


# In[132]:


# new adjoint model
def lorenzadjnew(u,s,r,b):
    uadj = np.zeros((nt,3))
    uadj[nt-1,:] = 0
    # add observation forcing
    if np.mod(nt-1,fobs)==0: #we are at an observation time
        uadj[nt-1,:] = uadj[nt-1,:] + Rinv.dot(u[nt-1,:]-uobs[nt-1,:])
    for i in range(nt-1,0,-1):
        uadj[i-1,0] = uadj[i,0] - dt*(s*uadj[i,0]-r*uadj[i,1]+uadj[i,1]*u[i,2]-uadj[i,2]*u[i,1])
        uadj[i-1,1] = uadj[i,1] - dt*(-s*uadj[i,0]+uadj[i,1]-uadj[i,2]*u[i,0])
        uadj[i-1,2] = uadj[i,2] - dt*(b*uadj[i,2]+uadj[i,1]*u[i,0])
        # add observation forcing
        if np.mod(i-1,fobs)==0: #we are at an observation time
            uadj[i-1,:] = uadj[i-1,:] + Rinv.dot(u[i-1,:]-uobs[i-1,:])
    return uadj


# In[133]:


# gradient of the new cost function
def gradientnew(X):
    g = np.zeros(6)
    u0 = X[0:3]
    s = X[3]
    r = X[4]
    b = X[5]
    # relative weight Jo vs Jb
    wb = 0.0
    wo = 1/ntobs
    # direct resolution
    u = lorenznew(u0,s,r,b)
    # adjoint resolution
    uadj = lorenzadjnew(u,s,r,b)
    # gradient with respect to initial condition
    g[0:3] = wb*Binv.dot(u0-u0b)+wo*uadj[0,:]
    # gradient with respect to parameters
    g[3:6] = wb*Bpinv.dot(X[3:6]-Xb[3:6])
    g[3] = g[3]+dt*np.dot(u[:,1]-u[:,0],uadj[:,0])
    g[4] = g[4]+dt*np.dot(u[:,0],uadj[:,1])
    g[5] = g[5]+dt*np.dot(-u[:,2],uadj[:,2])    
    return g


# In[136]:


# minimize cost with the gradient
res = minimize(costnew,Xb,method='Nelder-Mead',jac=gradientnew,options={'maxiter': 1000})


# In[137]:


res


# In[ ]:




