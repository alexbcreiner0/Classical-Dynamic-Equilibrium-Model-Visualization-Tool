import numpy as np
import scipy
import mesa

v = 8 #sectors

#Technology
A = np.random.rand(v,v)
r = max(abs(np.linalg.eigvals(A))) #spectral radius
rt = np.random.rand(1) #random spectral radius < 1
A = A*rt/r #normalize to random spectral radius < 1

#Eigendecomposition
lev = scipy.linalg.eig(A, left=True, right=False)[1]
rev = scipy.linalg.eig(A, left=False, right=True)[1]

indm = np.argmax(abs(np.linalg.eigvals(A)))

crt =  abs(lev[:,indm])#criticality
nb =  abs(rev[:,indm]) #balanced growth stock

crt = crt/np.inner(crt,nb)
nb = nb/np.inner(crt,nb)

#Network
alpha = 1.0E1*np.random.rand(v,) #consumption rate
lmbd = np.linalg.inv(np.eye(v)-A)@alpha #demand rate
rho_inv = (1.0E0*np.ones(v)) + (1.0E0*np.random.random(v)) #inverse utilization (efficiency)
mu = lmbd*(rho_inv.reshape((v,))) #production rate

interval_alpha = 1.0E-1*np.random.rand(v,) #consumption interval
batch_alpha = alpha*(interval_alpha.reshape((v,))) #consumption size

interval_mu = 1.0E-1*np.random.rand(v,) #production interval
batch_mu = mu*(interval_mu.reshape((v,))) #production size

dt = 1.0E-3 #time step

#Linear Program
S = np.vstack((np.eye(v)-A,-np.eye(v),-A))
con = np.append(alpha,-mu)

r_nvt = (1.0E2*np.ones(v)) + (9.0E2*np.random.random(v))
nvt = np.ones(v)*(r_nvt.reshape((v,))) #inventory level

#Diagnostic
print(rt)
print(min(rho_inv))
print(max(1/interval_mu)*dt)
print(max(1/interval_mu)*dt < 1.0E-1) #rerun everything til this is true, simulation quirk
print((np.eye(v)-A.T)@(crt*nvt*crt))

class Sector(mesa.Agent):
  """Site of production serving and dispatching orders"""
  def __init__(self, model, index, consumption_size, consumption_interval, production_size, production_interval, unit_inputs, init_inventory):
        super().__init__(model)

        self.index = index

        self.c_s = consumption_size
        self.c_i = consumption_interval

        self.p_s = production_size
        self.p_i = production_interval

        self.u_i = unit_inputs
        self.inventory = init_inventory

        self.orders = 0
        self.WIP = 0

  def step(self):
        #Consume Inventory
        if np.random.random() < (1 - np.exp(-1.0*dt/self.c_i)):
          cb = np.random.default_rng().exponential(scale=self.c_s)
          self.orders += cb
          if self.inventory > cb:
            self.inventory -= cb

        #Produce Inventory
        if self.orders > 0 and np.random.random() < (1 - np.exp(-1.0*dt/self.p_i)):
          self.inventory += self.WIP
          self.orders -= np.min((self.orders, self.WIP))
          self.WIP = np.random.default_rng().exponential(scale=self.p_s)
          #Check if short
          for agent in self.model.agents:
            if agent.inventory < self.u_i[agent.index]*self.WIP:
              self.WIP = 0
          #Use inputs
          for agent in self.model.agents:
            agent.inventory -= self.u_i[agent.index]*self.WIP
            agent.orders += self.u_i[agent.index]*self.WIP

class Economy(mesa.Model):
  """Productive Network"""

  def __init__(self, consumption_sizes, consumption_intervals, production_sizes, production_intervals, tech, init_inventories, cntrl):
        super().__init__()

        self.c_ss = consumption_sizes
        self.c_is = consumption_intervals

        self.p_ss = production_sizes
        self.p_is = production_intervals

        self.tech = tech
        self.init_inventories = init_inventories
        self.cntrl = cntrl

        for i in range(v):
          inputs = np.zeros(v)
          inputs[i] = 1.0
          inputs = self.tech@inputs #input agents will take from inventory

          sector = Sector(self, i, self.c_ss[i], self.c_is[i], self.p_ss[i], self.p_is[i], inputs, init_inventories[i])
          self.agents.add(sector)

  def step(self):
        if self.cntrl > 0:
          #Record orders
          q = np.zeros(v)
          stck_r = np.zeros(v)
          for agent in self.agents:
            q[agent.index] = agent.orders*(crt[agent.index]**2)
            stck_r[agent.index] = agent.inventory/agent.p_i

          #Solve plan
          cst = (np.eye(v)-A.T)@q
          plnr = scipy.optimize.linprog(-cst,A_ub = -S, b_ub = -np.append(con,-stck_r+alpha))
          pln  = plnr.x

          #Update plan
          if plnr.success:
            for agent in self.agents:
              agent.p_s = pln[agent.index]*agent.p_i

        self.agents.shuffle_do("step")

ts = 8**6

e = Economy(batch_alpha,interval_alpha,batch_mu,interval_mu,A,nvt,0) #uncontrolled
eC = Economy(batch_alpha,interval_alpha,batch_mu,interval_mu,A,nvt,1)#controlled

order_hst = np.zeros((v,ts))
inv_hst = np.zeros((v,ts))
pr_hst = np.zeros((v,ts))

order_hstC = np.zeros((v,ts))
inv_hstC = np.zeros((v,ts))
pr_hstC = np.zeros((v,ts))

for i in range(ts):
    for agent in e.agents:
      order_hst[agent.index,i] = agent.orders
      inv_hst[agent.index,i] = agent.inventory
      pr_hst[agent.index,i] = agent.p_s/agent.p_i
    e.step()

    for agent in eC.agents:
      order_hstC[agent.index,i] = agent.orders
      inv_hstC[agent.index,i] = agent.inventory
      pr_hstC[agent.index,i] = agent.p_s/agent.p_i
    eC.step()
