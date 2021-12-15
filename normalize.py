import tvm

n = tvm.var('n')
A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder((n,), name='B')
k = tvm.reduce_axis((10, n), 'k')
C = tvm.te.compute((1,), lambda _: tvm.te.sum(A[k] * B[k], axis=k), name='C')

s = tvm.te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")
s = s.normalize()
print(tvm.lower(s, [A, B, C], simple_mode=True))