import mesa
import numpy as np
import scipy

class Sector(mesa.Agent):
    """Site of production serving and dispatching orders"""
    def __init__(self, model, index, consumption_size, consumption_interval, production_size, production_interval, unit_inputs, init_inventory, dt):
        super().__init__(model)

        self.index = index

        self.c_s = consumption_size
        self.c_i = consumption_interval

        self.p_s = production_size
        self.p_i = production_interval

        self.u_i = unit_inputs
        self.inventory = init_inventory

        self.dt = dt

        self.orders = 0
        self.WIP = 0

    def step(self):
        #Consume Inventory
        if np.random.random() < (1 - np.exp(-1.0*self.dt/self.c_i)):
            cb = np.random.default_rng().exponential(scale=self.c_s)
            self.orders += cb
            if self.inventory > cb:
                self.inventory -= cb

        #Produce Inventory
        if self.orders > 0 and np.random.random() < (1 - np.exp(-1.0*self.dt/self.p_i)):
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

    def __init__(self, consumption_sizes, consumption_intervals, production_sizes, production_intervals, tech, init_inventories, cntrl, dt, m_evec_l, con, S, alpha, A):
        super().__init__()

        self.c_ss = consumption_sizes
        self.c_is = consumption_intervals

        self.p_ss = production_sizes
        self.p_is = production_intervals

        self.tech = tech
        self.init_inventories = init_inventories
        self.cntrl = cntrl

        self.crt = m_evec_l
        self.con = con
        self.S = S

        self.alpha = alpha
        self.A = A

        for i in range(4):
            inputs = np.zeros(4)
            inputs[i] = 1.0
            inputs = self.tech@inputs #input agents will take from inventory

            sector = Sector(self, i, self.c_ss[i], self.c_is[i], self.p_ss[i], self.p_is[i], inputs, init_inventories[i], dt)
            self.agents.add(sector)

    def step(self):
        if self.cntrl > 0:
            #Record orders
            q = np.zeros(4)
            stck_r = np.zeros(4)
            for agent in self.agents:
                q[agent.index] = agent.orders*(self.crt[agent.index]**2)
                stck_r[agent.index] = agent.inventory/agent.p_i

            #Solve plan
            cst = (np.eye(4)-self.A.T)@q
            plnr = scipy.optimize.linprog(-cst,A_ub = -self.S, b_ub = -np.append(self.con,-stck_r+self.alpha))
            pln  = plnr.x

           #Update plan
            if plnr.success:
                for agent in self.agents:
                    agent.p_s = pln[agent.index]*agent.p_i

        self.agents.shuffle_do("step")
