from scipy import sparse

a = sparse.dok_matrix((100, 10000))
b = sparse.dok_matrix((100, 100))
a[:,[1,2,3,4,5,6]] = 1
# b[:,1] = 1
b[:,1] = 1
# print(a[:,1])
x = a[1,:]
y = b[1,[2,3,4]]
z = [a[0,:] for i in range(10)]
mat_a = x
mat_b = sparse.dok_matrix((10000, 10000))
for i in range(9999):
    mat_a = sparse.vstack((mat_a, x))
    mat_b = sparse.vstack((mat_b,x))

mat = mat_a.dot(mat_b.T)
print(mat.shape)