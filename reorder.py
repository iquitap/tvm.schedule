import tvm

n = 1024
A = tvm.te.placeholder((n, n), name='A')
B = tvm.te.placeholder((n,n), name='B')
C = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = tvm.te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=32)
yo, yi = s[C].split(s[C].op.axis[1], factor=32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].reorder(xo, yo, yi, xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))