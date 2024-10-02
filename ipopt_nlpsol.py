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

f = 0 # Objective
x = [] # List of decision variable symbols
lbx = [];ubx = [] # Simple bounds
x0 = [] # Initial value
g = [] # Constraints list
lbg = [];ubg = [] # Constraint bounds
p = [] # Parameters
p_val = [] # Parameter values

N = 20
T0 = 10

# Create decision variable for T and store it
T = ca.MX.sym("T")
x.append(T)
x0.append(T0)
lbx.append(0);ubx.append(ca.inf);

dt = T/N

X = [ca.MX.sym("X",nx) for i in range(N+1)]
x += X
for k in range(N+1):
    x0.append(ca.vertcat(0,k*T0/N,pi/2))
    lbx.append(-ca.DM.inf(nx,1));ubx.append(ca.DM.inf(nx,1));
    
U = [ca.MX.sym("U",nu) for i in range(N)]
x += U
for k in range(N):
    x0.append(ca.vertcat(0,1))
    lbx.append(-pi/6);ubx.append(pi/6) # -pi/6 <= delta<= pi/6
    lbx.append(0);ubx.append(1) # 0 <= V<=1

# Round obstacle
pos0 = ca.vertcat(0.2,5)
r0 = 1

X0 = ca.MX.sym("X0",nx)
p.append(X0)
p_val.append(ca.vertcat(0,0,pi/2))

f = T # Time Optimal objective
for k in range(N):
    # Multiple shooting gap-closing constraint
    g.append(X[k+1]-F(X[k],U[k],dt))
    lbg.append(ca.DM.zeros(nx,1))
    ubg.append(ca.DM.zeros(nx,1))
    
    if k==0:
        # Initial constraints
        g.append(X[0]-X0)
        lbg.append(ca.DM.zeros(nx,1))
        ubg.append(ca.DM.zeros(nx,1))
        
    # Obstacle avoidance
    pos = X[k][:2]
    g.append(ca.sumsqr(pos-pos0))
    lbg.append(r0**2);ubg.append(ca.inf)
        
    if k==N-1:
        # Final constraints
        g.append(X[-1][:2])
        lbg.append(ca.vertcat(0,10));ubg.append(ca.vertcat(0,10))

# Add some regularization
for k in range(N+1):
    f += ca.sumsqr(X[k][0])

# Solve the problem

nlp = {}
nlp["f"] = f
nlp["g"] = ca.vcat(g)
nlp["x"] = ca.vcat(x)
nlp["p"] = ca.vcat(p)
solver = ca.nlpsol('solver',"ipopt",nlp,{"expand":True})

res = solver(x0 = ca.vcat(x0),
    lbg = ca.vcat(lbg),
    ubg = ca.vcat(ubg),
    lbx = ca.vcat(lbx),
    ubx = ca.vcat(ubx),
    p = ca.vcat(p_val)
)

