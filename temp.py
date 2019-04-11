from scipy import sparse

b = sparse.dok_matrix((5021348, 5021348))
b[0][0] = 1
b[0][0] = 2
print(b[0,0])