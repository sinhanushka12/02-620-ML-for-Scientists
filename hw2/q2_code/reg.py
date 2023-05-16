import numpy as np
import math

class LogisticRegression:
    def __init__(self, d):
        self.w = np.random.randn(d)

    def compute_loss(self, X, Y):
        loss = 0
        for i in range(len(X)):
            dotProd = np.dot(self.w, X[i])
            loss += (1/len(X))*(math.log(1 + math.exp(dotProd))- (Y[i]* dotProd))
        """
        Compute l(w) with n samples.
        Inputs:
            X  - A numpy array of size (n, d). Each row is a sample.
            Y  - A numpy array of size (n,). Each element is 0 or 1.
        Returns:
            A float.
        """
        return loss
        raise NotImplementedError

    def compute_grad(self, X, Y):
        gradient = np.zeros(len(X[0])) 
        for i in range(len(X)):
            dotProd = np.dot(self.w, X[i])
            gradient = np.add(gradient, ((math.exp(dotProd)/(1+math.exp(dotProd)))* X[i]-(Y[i]*X[i])))
            
        """
        Compute the derivative of l(w).
        Inputs: Same as above.
        Returns:
            A numpy array of size (d,).
        """
        return gradient/len(X)
        raise NotImplementedError

    def train(self, X, Y, eta, rho):
        
        while True:
            descent = self.compute_grad(X, Y)
            if np.linalg.norm(descent) < rho: #length of vector descent
                break 
            self.w = self.w - descent
          
        # Initialize w 
        # while True do
        #     Compute gradient ∇wl(w) 
        #     If ∥∇wl(w)∥ < ρ then break 
        #     w ← w − η · ∇wl(w)
        # end while

        """
        Train the model with gradient descent.
        Update self.w with the algorithm listed in the problem.
        Returns: Nothing.
        """



if __name__ == '__main__':
    # Sample Input/Output
    d = 10
    n = 1000

    np.random.seed(0)
    X = np.random.randn(n, d)
    Y = np.array([0] * (n // 2) + [1] * (n // 2))
    eta = 1e-3
    rho = 1e-6

    reg = LogisticRegression(d)
    reg.train(X, Y, eta, rho)
    print(reg.w)

    # The output should be close to
    # [ 0.15289573 -0.063752   -0.06434498 -0.02005378  0.07812127 -0.04307333
    #  -0.0691539  -0.02769485 -0.04193284 -0.01156307]
    # Error should be less than 0.001 for each element