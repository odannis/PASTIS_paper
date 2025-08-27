import numpy as np

def fourier_vector(dim,order)  -> list[np.ndarray]:
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int))
        
    coeffs = [ c for degree in coeffs[1:] for c in degree ]
    return coeffs


class Fourier:
    def __init__(self, dim :int, order : int = 4 ) -> None:
        """
        dim : dimension of the space
        order : order of the polynomials
        simple_base : if True, the basis is on the i dimension -> X_i X_mu X_nu ... up to order 'order'-1 (no constant term)
        """
        self.order = order
        self.dim = dim
        self.coeffs = fourier_vector(dim, order)
        self.base_function : list|None = None #self.polynomial_basis_function()
        self.non_zero_dim = []
        self.n_base = (2*len(self.coeffs) + 1)*self.dim
        print("Number of basis functions : ", self.n_base)

    def get_basis_values(self, X : np.ndarray):
        """
        Evaluate the polynomial basis with coefficients of order defined in the class at the
        points 'X'.
        """
        coeffs = self.coeffs
        l_name = []
        self.non_zero_dim = []
        center = np.mean(X, axis=0)
        width = 2.5 * np.abs(X-center).max(axis=(0))
        
        #center = np.ones_like(center)
       # width = np.ones_like(width)

        # if self.dim >= self.order:
        #     # Two paradigms for the computation of the function: sparse if
        #     # dim > order, dense otherwise.
        #     Xc = 2 * np.pi* (X - self.center) / self.width
        #     vals = np.ones((2*len(coeffs)+1, *X.shape))
        #     for i,c in enumerate(coeffs):
        #         vals[2*i+1] = np.cos(sum(Xc[c]))
        #         vals[2*i+2] = np.sin(sum(Xc[c])) 
        #     return vals
        # else:
        coeffs_lowdim = np.array([ [ list(c).count(i) for i in range(self.dim) ] for c in coeffs ])
        Xc = 2 * np.pi* (X - center) / width
        vals = np.ones((2*len(coeffs_lowdim)+1, *X.shape[:-1]))
        for i,c in enumerate(coeffs_lowdim):
            vals[2*i+1] = np.cos(Xc.dot(c))
            vals[2*i+2] = np.sin(Xc.dot(c))
        ### Put zero insteand of 1  but keep ones for the constant value
        name = np.array(["X_%s"%(i) for i in range(0, self.dim)])
        fourier_basis = np.zeros((self.dim*vals.shape[0], *X.shape))
        for dim in range(self.dim):
            for i in range(vals.shape[0]):
                #l_name.append("On %s : cos(%s)"%(name[dim], coeffs_lowdim[i]))
                if i == 0:
                    l_name.append("On %s : 1"%(name[dim]))
                else:
                    if i%2 == 1:
                        l_name.append("On %s : sin(%s)"%(name[dim], coeffs_lowdim[i//2]))
                    else:
                        l_name.append("On %s : cos(%s)"%(name[dim], coeffs_lowdim[i//2 - 1]))
                #fourier_basis[(2*i+1) + 2*(dim)*(len(coeffs_lowdim)) + dim,:,dim] = vals[2*i+1]
                #fourier_basis[2*i+2 + 2*(dim)*len(coeffs_lowdim) + dim,:,dim] = vals[2*i+2]
                fourier_basis[i + vals.shape[0]*dim,:,dim] = vals[i]
                self.non_zero_dim.append(dim)
        return fourier_basis, l_name
    
    
if __name__== '__main__':
    import unittest
    class TestPrime(unittest.TestCase):
        x = np.random.rand(100, 3)
        polynome_info = Fourier(dim=3, order=2)
        
        def test(self):
            base_evaluated, _ = self.polynome_info.get_basis_values(self.x)
            print(base_evaluated.shape, self.polynome_info.n_base)
            
        print("test prin")
    unittest.main()