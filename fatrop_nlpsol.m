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

f = 0; %Objective
x = {}; % List of decision variable symbols
lbx = [];ubx = []; % Simple bounds
x0 = []; % Initial value
g = {}; % Constraints list
lbg = [];ubg = []; % Constraint bounds
equality = []; % Boolean indicator helping structure detection
p = {}; % Parameters
p_val = []; % Parameter values

N = 20;
T0 = 10;

X = {};
T = {};
U = {};

for k = 1:N+1
    X{k} = MX.sym(['X_' num2str(k)], nx);
    x{end+1} = X{k};
    x0 = [x0; [0; k*T0/N; pi/2]]; % Initial value
    lbx = [lbx; -inf(nx, 1)];
    ubx = [ubx; inf(nx, 1)];
    
    T{k} = MX.sym(['T_' num2str(k)]);
    x{end+1} = T{k};
    x0 = [x0; T0];
    lbx = [lbx; 0];
    ubx = [ubx; inf];
    
    if k <= N
        U{k} = MX.sym(['U_' num2str(k)], nu);
        x{end+1} = U{k};
        x0 = [x0; [0; 1]]; % Initial guess
        lbx = [lbx; -pi/6; 0]; % Bounds on delta and V
        ubx = [ubx; pi/6; 1];
    end
end

% Round obstacle
p0 = [0.2; 5];
r0 = 1;

X0 = MX.sym('X0', nx);
p{end+1} = X0;
p_val = [p_val; [0; 0; pi/2]];

f = sum1(vertcat(T{:})); % Time-optimal objective
for k = 1:N
    % Multiple shooting gap-closing constraint
    g{end+1} = X{k+1} - F(X{k}, U{k}, T{k}/N);
    lbg = [lbg; zeros(nx, 1)];
    ubg = [ubg; zeros(nx, 1)];
    equality = [equality; true(nx, 1)];
    
    g{end+1} = T{k+1} - T{k};
    lbg = [lbg; 0];
    ubg = [ubg; 0];
    equality = [equality; true];
    
    if k == 1
        % Initial constraint
        g{end+1} = X{1} - X0;
        lbg = [lbg; zeros(nx, 1)];
        ubg = [ubg; zeros(nx, 1)];
        equality = [equality; true(nx, 1)];
    end
    
    % Obstacle avoidance constraint
    pos = X{k}(1:2);
    g{end+1} = sumsqr(pos - p0);
    lbg = [lbg; r0^2];
    ubg = [ubg; inf];
    equality = [equality; false];
    
    if k == N
        % Final constraint
        g{end+1} = X{k+1}(1:2);
        lbg = [lbg; 0; 10];
        ubg = [ubg; 0; 10];
        equality = [equality; true; true];
    end
end

% Add regularization to the objective
for k = 1:N+1
    f = f + sumsqr(X{k}(1));
end

% Solver definition
nlp = struct('f', f, 'g', vertcat(g{:}), 'x', vertcat(x{:}), 'p', vertcat(p{:}));

opts = struct;
opts.expand = true;
opts.fatrop.mu_init = 0.1;
opts.structure_detection = 'auto';
opts.debug = true;
opts.equality = equality;

solver = nlpsol('solver', 'fatrop', nlp, opts);

res = solver('x0', x0, ...
             'lbx', lbx, ...
             'ubx', ubx, ...
             'lbg', lbg, ...
             'ubg', ubg, ...
             'p', p_val);
