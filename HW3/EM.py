import numpy as np
from matplotlib import pyplot as plt
import time

def MultiVarNormal(x,mean,cov):
    determinantCov = np.linalg.det(np.dot(2*np.pi,cov))
    inverseCov = np.linalg.inv(cov)
    gene_adjusted = x - mean
    exponent = np.matmul(np.matmul(gene_adjusted, inverseCov), gene_adjusted.transpose())
    scalar = exponent.item(0)
    prob =  np.power(determinantCov, -0.5)*(np.power(np.e, -0.5*scalar))
    return prob
    """
    MultiVarNormal implements the PDF for a mulitvariate gaussian distribution
    (You can do one sample at a time or all at once)
    Input:
        x - An (d) numpy array
            - Alternatively (n,d)
        mean - An (d,) numpy array; the mean vector
        cov - a (d,d) numpy arry; the covariance matrix
    Output:
        prob - a scaler
            - Alternatively (n,)

    Hints:
        - Use np.linalg.pinv to invert a matrix
        - if you have a (1,1) you can extrect the scalar with .item(0) on the array
            - this will likely only apply if you compute for one example at a time
    """
    

def UpdateMixProps(hidden_matrix):
    updatedMixProps = []
    n = len(hidden_matrix)
    for i in range(len(hidden_matrix[0])):
        mixprop = 0
        for x in hidden_matrix:
            mixprop += x[i]   # P(c^n = i|x^n)
        updatedMixProps.append(mixprop/n)
    return np.array(updatedMixProps)
    
    """
    Returns the new mixing proportions given a hidden matrix
    Input:
        hidden_matrix - A (n, k) numpy array
    Output:
        mix_props - A (k,) numpy array
    Hint:
        - See equation in Lecture 10 pg 42

    """
    

def UpdateMeans(X, hidden_matrix):
    updatedmeans = []
    for i in range(len(hidden_matrix[0])):
        mean_numerator = np.zeros(len(X[0]))
        mean_denominator = 0
        for n in range(len(X)):
            mean_numerator  = np.add(mean_numerator, np.dot(hidden_matrix[n][i], X[n]))
            mean_denominator += hidden_matrix[n][i]
        updatedmeans.append(np.dot(1/mean_denominator, mean_numerator))
    return np.array(updatedmeans)
    """
    Returns the new means for the gaussian distributions given the data and the hidden matrix
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
    Output:
        new_means - A (k,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    

def UpdateCovar(X, hidden_matrix, means, i):
    numerator = np.zeros((len(means[0]), len(means[0])))
    denominator = 0
    for n in range(len(X)): 
        product = np.matmul(np.array([X[n] - means[i]]).transpose(), np.array([X[n]-means[i]]))
        numerator += hidden_matrix[n][i]*product
        denominator += hidden_matrix[n][i]
    updatedCovar = numerator/denominator# dividing each entry of numerator (matrix) by the denominator (scalar)
    return updatedCovar
    """
    Returns new covariance for a single gaussian distribution given the data, hidden matrix, and distribution mean
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
        means - A (k,d) numpy array; the means for this distribution (of all clusters)
        i - the relevant cluster
    Output:
        new_cov - A (d,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    

def UpdateCovars(X, hidden_matrix, means): # get 3 (2-D lists) --> covariance matrix (d,d) for each cluster (k = 3)
    new_Covs = []
    for i in range(len(means)):
        new_Covs.append(UpdateCovar(X, hidden_matrix, means, i))
    return np.array(new_Covs)

    """
    Returns a new covariance matrix for all distributions using the function UpdateCovar()
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
        u - A (k,d) numpy array; All means for the distributions
    Output:
        new_covs - A (k,d,d) numpy array
    Hint:
        - Use UpdateCovar() function
    """
    


def HiddenMatrix(X, means, covs, mix_props):
    # hidden matrix
    hiddenMat = []
    for n in range(len(X)): # N
        hiddenMatRow = []
        denominator = 0
        for i in range(len(means)): # k         
            denominator += mix_props[i] * MultiVarNormal(X[n],means[i],covs[i])
        for i in range(len(means)):
            numerator = mix_props[i] * MultiVarNormal(X[n],means[i],covs[i])
            hiddenMatRow.append(numerator/denominator)
        hiddenMat.append(hiddenMatRow)

    # log likelihood 
    logLikelihood = 0
    for n in range(len(X)):
        summand = 0
        for j in range(len(means)):
            summand += mix_props[j] * MultiVarNormal(X[n],means[j],covs[j])
        logLikelihood += np.log(summand)
    
    return np.array(hiddenMat), logLikelihood
        
    """
    Computes the hidden matrix for the data. This function should also compute the log likelihood
    Input:
        X - An (n,d) numpy array
        means - An (k,d) numpy array; the mean vector
        covs - a (k,d,d) numpy arry; the covariance matrix
        mix_props - a (k,) array; the mixing proportions
    Output:
        hidden_matrix - a (n,k) numpy array
        ll - a scalar; the log likelihood
    Hints:
        - Construct an intermediate matrix of size (n,k). This matrix can be used to calculate the loglikelihood and the hidden matrix
            - Element t_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
            P(X_i | c = j)P(c = j)
        - Each rows of the hidden matrix should sum to 1
            - Element h_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
                P(X_i | c = j)P(c = j) / (Sum_{l=1}^{k}(P(X_i | c = l)P(c = l)))
    """



def GMM(X, init_means, init_covs, init_mix_props, thres=0.001):
    clusters = []
    log_likelihoods = [] # 2 element list, contains ll from current and the previous iteration 
    
    # E step
    h_mat, logLike = HiddenMatrix(X, init_means, init_covs, init_mix_props)
    log_likelihoods.append(logLike)
    
    # M step
    newMixProps = UpdateMixProps(h_mat)
    newMeans = UpdateMeans(X, h_mat)
    newCovars = UpdateCovars(X, h_mat, newMeans)
    
    # E step again
    h_mat, logLike = HiddenMatrix(X, newMeans, newCovars, newMixProps)
    log_likelihoods.append(logLike)

    while log_likelihoods[-1] - log_likelihoods[-2] >= thres:
        # M step again
        newMixProps = UpdateMixProps(h_mat)
        newMeans = UpdateMeans(X, h_mat)
        newCovars = UpdateCovars(X, h_mat, newMeans)
        # E step again
        h_mat, logLike = HiddenMatrix(X, newMeans, newCovars, newMixProps)
        log_likelihoods.append(logLike)
    
    for n in range(len(X)):
        prob = []
        # P(c = i | X_n) and assign X_n to the cluster (c = i) with the max P(c = i | X_n)
        for i in range(len(init_means)):
            prob.append(h_mat[n][i])
        clusters.append(prob.index(max(prob)))
    
    return clusters, log_likelihoods, h_mat


        
    """
    Runs the GMM algorithm
    Input:
        X - An (n,d) numpy array
        init_means - a (k,d) numpy array; the initial means
        init_covs - a (k,d,d) numpy arry; the initial covariance matrices
        init_mix_props - a (k,) array; the initial mixing proportions
    Output:
        - clusters: and (n,) numpy array; the cluster assignment for each sample
        - ll: the log likelihood at the stopping condition
    Hints:
        - Use all the above functions
        - Stoping condition should be when the difference between your ll from 
            the current iteration and the last iteration is below your threshold
    """
    


if __name__ == "__main__":
    data = np.loadtxt("data/mouse-data/hip1000.txt", dtype=np.float32,delimiter=',')
    test_means = np.loadtxt("data/test_mean.txt").T
    print('Data shape:',data.shape) # 208, 879
    print('test_means shape: ',test_means.shape) # 3, 208

    # initializations
    X = data[0:10].transpose()              # (879,10)                                 
    init_means = test_means[:, :10]         # (3, 10)
    init_mix_props = [0.3, 0.3, 0.4]
    init_cov = np.diag(np.ones(10), 0)      # (10, 10)
    lst = [init_cov, init_cov, init_cov]
    init_covs = np.array(lst)
    print("Running EM")
    clusters, ll, h_mat = GMM(X, init_means, init_covs, init_mix_props)
    print("Clusters: ", clusters)
    print("the log likelihood at the stopping condition: ", ll[-1])

    # Q6 a
    # y = ll
    # x = list(range(1, len(ll)+1))
    # plt.plot(x, y)
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("Log likelihoods")
    # plt.title("Log likelihoods of Data over iterations for k = 3")
    # plt.show()

    # Q6 b 
    # prob = []
    # for i in range(len(init_means)):
    #     prob.append(h_mat[0][i])
    # print("Probability of the first gene to belong to each of the three clusters: ", prob)

    # Q6 c 
    # Q6 c
    init_covs = np.array([init_cov, init_cov])
    all_loglikelihoods = []
for k in range(3, 11):
    init_means = 2 * np.random.rand(k, 10) - 1 # getting a matrix of (k, 10) with random number b/w (-1, 1)
    mix_props = np.random.rand(k,)
    init_mix_props = 1/np.sum(mix_props) * mix_props  # making elements in mix props add up to 1
    init_covs = np.append(init_covs, [init_cov], axis = 0) # every iteration, add one more initial covariance matrix into the list
    c, ll, h_mat = GMM(X, init_means, init_covs, init_mix_props)
    all_loglikelihoods.append(ll[-1]) # all likelihoods for k = 3, ..., 10
    

