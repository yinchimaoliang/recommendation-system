from scipy import sparse

b = sparse.dok_matrix((5, 5))
b[0][0] = 1
b[0][0] = 2
print(b[3,3])