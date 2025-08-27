import numpy as np

def polynomial_basis(dim, order):
    # A simple polynomial basis, X -> X_mu X_nu ... up to polynomial
    # degree 'order'.
    
    # We first generate the coefficients, ie the indices mu,nu.. of
    # the polynomials, in a non-redundant way. We start with the
    # constant polynomial (empty list of coefficients) and iteratively
    # add new indices.
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
    # Group all coefficients together
    coeffs = [ c for degree in coeffs for c in degree ]
    return [ lambda x : np.prod(x[c]) for c in coeffs ]
