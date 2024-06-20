import casadi as ca
import pylab as plt


actual = ca.Sparsity.from_file("debug_fatrop_actual.mtx")

A = ca.Sparsity.from_file("debug_fatrop_A.mtx")
B = ca.Sparsity.from_file("debug_fatrop_B.mtx")
C = ca.Sparsity.from_file("debug_fatrop_C.mtx")
D = ca.Sparsity.from_file("debug_fatrop_D.mtx")
I = ca.Sparsity.from_file("debug_fatrop_I.mtx")
errors = ca.Sparsity.from_file("debug_fatrop_errors.mtx").row()

plt.figure()
plt.spy(A,marker='o',color='r',markersize=5,label="expected A",markerfacecolor="white")
plt.spy(B,marker='o',color='b',markersize=5,label="expected B",markerfacecolor="white")
plt.spy(C,marker='o',color='g',markersize=5,label="expected C",markerfacecolor="white")
plt.spy(D,marker='o',color='y',markersize=5,label="expected D",markerfacecolor="white")
plt.spy(I,marker='o',color='k',markersize=5,label="expected I",markerfacecolor="white")
plt.spy(actual,marker='o',color='k',markersize=2,label="actual")

plt.hlines(errors, 0, A.shape[1],color='gray', linestyle='-',label="offending rows")

plt.title("Debug view of fatrop interface structure detection")
plt.legend()
plt.show()
