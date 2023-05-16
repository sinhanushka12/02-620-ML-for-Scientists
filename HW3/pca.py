import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pca(X, D):
    # mean center the data 
    # step 1: 
        # - compute KxK covariance matrix S 
            # (X^TX)
        # - eigen decomposition of S to obtain eigenvectors and eigenvalues 
            # e,v = run eig(cov) OR U, sigma, vh = run SVD(cov)
            # np.linalg.svd()
        # - pick the eigen vectors (principal components) with the largest D eigen values. 
    # step 2:
        # - Project each data point in X to the M dimensions described by D eigenvectors 
            # (i.e., multiply each row of X by each of the M eigen vectors )
    X_centered = X - X.mean()
    covMatrix = np.dot(X_centered.transpose(), X_centered)   # X^TX
    U, e, v = np.linalg.svd(covMatrix) 
    #e, v = np.linalg.eig(covMatrix)
    EigenVectors = v.transpose()[:, 0:D]
    newX = np.dot(X_centered, EigenVectors)
    return np.dot(newX, EigenVectors.transpose())
    
    #raise NotImplementedError


def sklearn_pca(X, D):
    """
    Your PCA implementation should be equivalent to this function.
    Do not use this function in your implementation!
    """
    from sklearn.decomposition import PCA
    p = PCA(n_components=D, svd_solver='full')
    trans_pca = p.fit_transform(X)
    X = p.inverse_transform(trans_pca)
    return X


    

if __name__ == '__main__':
    D = 256

    a = Image.open('data/20180108_171224.jpg').convert('RGB')
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Original')
    ax1.imshow(a)
    b = np.array(a)
    c = b.astype('float') / 255.
    for i in range(3):
        x = c[:, :, i]
        mu = np.mean(x)
        x = x - mu
        x_true = sklearn_pca(x, D)
        x = pca(x, D)
        assert np.allclose(x, x_true, atol=0.05)  # Test your results
        x = x + mu
        c[:, :, i] = x

    b = np.uint8(c * 255.)
    a = Image.fromarray(b)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Compressed')
    ax2.imshow(a)
    plt.show()
