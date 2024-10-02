import casadi.*

% Define symbols
pos = MX.sym('pos', 2);
theta = MX.sym('theta');

delta = MX.sym('delta');
V = MX.sym('V');

% States
x = [pos; theta];

% Controls
u = [delta; V];

L = 1;

% ODE rhs
% Bicycle model
% (S. LaValle. Planning Algorithms. Cambridge University Press, 2006, pp. 724–725.)

ode = [V * [cos(theta); sin(theta)]; V/L * tan(delta)];

% Discretize system
dt = MX.sym('dt');
sys = struct('x', x, 'u', u, 'p', dt, 'ode', ode * dt); % Time scaling

intg = integrator('intg', 'rk', sys, 0, 1, struct('simplify', true, 'number_of_finite_elements', 4));

res = intg('x0', x, 'p', dt, 'u', u);

F = Function('F', {x, u, dt}, {res.xf}, {'x', 'u', 'dt'}, {'xnext'});

nx = numel(x);
nu = numel(u);

opti = Opti();

N = 20;
T0 = 10;

X = {};
T = {};
U = {};

for k = 1:N
    X{end+1} = opti.variable(nx);
    T{end+1} = opti.variable();
    U{end+1} = opti.variable(nu);
end
X{end+1} = opti.variable(nx);
T{end+1} = opti.variable();

% Round obstacle
p0 = [0.2; 5];
r0 = 1;

X0 = opti.parameter(nx);
opti.set_value(X0, [0; 0; pi/2]);

for k = 1:N
    % Multiple shooting gap-closing constraint
    opti.subject_to(X{k+1} == F(X{k}, U{k}, T{k}/N));
    opti.subject_to(T{k+1} == T{k});
    
    if k == 1
        % Initial constraints
        opti.subject_to(X{1} == X0);
    end
    
    opti.subject_to(0 <= U{k}(2) <= 1); % 0 <= V <= 1
    opti.subject_to(-pi/6 <= U{k}(1) <= pi/6); % -pi/6 <= delta <= pi/6
    
    % Obstacle avoidance
    p = X{k}(1:2);
    opti.subject_to(sumsqr(p - p0) >= r0^2);
    
    if k == N
        % Final constraints
        opti.subject_to(X{N+1}(1:2) == [0; 10]);
    end
    
    opti.set_initial(U{k}(2), 1);
    opti.set_initial(X{k}, [0; k*T0/N; pi/2]);
    opti.set_initial(T{k}, T0);
end

opti.set_initial(X{N+1}, [0; T0; pi/2]);
opti.set_initial(T{N+1}, T0);

X = [X{:}];
T = [T{:}];

opti.minimize(sumsqr(X(1,:)) + sum(T));

% Solver options
solver = 'fatrop';

options = struct;
options.expand = true;

if strcmp(solver, 'fatrop')
    options.fatrop.mu_init = 0.1;
    options.structure_detection = 'auto';
    options.debug = true;

    % (codegen of helper functions)
    % options.jit = true;
    % options.jit_temp_suffix = false;
    % options.jit_options.flags = {'-O3'};
    % options.jit_options.compiler = 'ccache gcc';
end

if strcmp(solver, 'ipopt')
    % Options for ipopt (if needed)
end

opti.solver(solver, options);

sol = opti.solve();

disp(sol.value(X)');