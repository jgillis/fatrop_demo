import casadi as ca
from numpy import sin, cos, tan, pi

pos = ca.MX.sym('pos',2)
theta = ca.MX.sym('theta')

delta = ca.MX.sym('delta')
V     = ca.MX.sym('V')

# States
x = ca.vertcat(pos,theta)

# Controls
u = ca.vertcat(delta,V)

L = 1

# ODE rhs
# Bicycle model
# (S. LaValle. Planning Algorithms. Cambridge University Press, 2006, pp. 724–725.)

ode = ca.vertcat(V*ca.vertcat(cos(theta),sin(theta)),V/L*tan(delta))

# Discretize system
dt = ca.MX.sym("dt")
sys = {}
sys["x"] = x
sys["u"] = u
sys["p"] = dt
sys["ode"] = ode*dt # Time scaling

intg = ca.integrator('intg','rk',sys,0,1,{"simplify":True, "number_of_finite_elements": 4})

F = ca.Function('F',[x,u,dt],[intg(x0=x,u=u,p=dt)["xf"]],["x","u","dt"],["xnext"])


nx = x.numel()
nu = u.numel()

opti = ca.Opti()

N = 20
T0 = 10

T = opti.variable()
opti.set_initial(T,T0)
dt = T/N

X = [opti.variable(nx) for i in range(N+1)]
U = [opti.variable(nu) for i in range(N)]

# Round obstacle
p0 = ca.vertcat(0.2,5)
r0 = 1

X0 = opti.parameter(nx)
opti.set_value(X0, ca.vertcat(0,0,pi/2))

for k in range(N):
    # Multiple shooting gap-closing constraint
    opti.subject_to(X[k+1]==F(X[k],U[k],dt)) 
    
    if k==0:
        # Initial constraints
        opti.subject_to(X[0]==X0)
        
    opti.subject_to(0 <= (U[k][1] <=1)) # 0 <= V<=1
    opti.subject_to(-pi/6 <= (U[k][0] <= pi/6)) # -pi/6 <= delta<= pi/6
    
    # Obstacle avoidance
    p = X[k][:2]
    opti.subject_to(ca.sumsqr(p-p0)>=r0**2)
        
    if k==N-1:
        # Final constraints
        opti.subject_to(X[-1][:2]==ca.vertcat(0,10))
    
    opti.set_initial(U[k][1],1)
    opti.set_initial(X[k],ca.vertcat(0,k*T0/N,pi/2))

opti.set_initial(X[-1],ca.vertcat(0,T0,pi/2))

X = ca.hcat(X)

opti.minimize(ca.sumsqr(X[0,:])+T)

opti.solver('ipopt',{"expand":True})

sol = opti.solve()

print(sol.value(X))
