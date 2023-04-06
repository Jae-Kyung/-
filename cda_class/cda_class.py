import numpy as np

class cda():
    # input 
    # predictors (n by p, 2D array)
    # response (n, 1D array)
    
    def __init__(self, predictors, response):
        self.predictors = predictors
        self.response = response
        self.sample_size, self.number_predictors = predictors.shape        
        # initialization
        self.beta = np.zeros(self.number_predictors)
        self.feed_forward()
        
    # 'feed_forward' 
    # calculdates fitted values, residuals, residuals sum of squares
    # given coefficients
    def feed_forward(self):
        predictors = self.predictors
        response = self.response
        beta = self.beta
        self.fitted_values = predictors.dot(beta)
        self.residuals = response - self.fitted_values
        self.rss = np.mean(self.residuals ** 2)
    
    # 'training'
    # iteratively learns coefficients based on the given dataset 
    def training(self, n_iter = 1000, eps = 1e-04, verbose = True):
        # inputs
        # n_iter (scalar)
        # eps (scalar)
        # verbose (boolean)
        predictors = self.predictors
        
        self.n_iter = n_iter
        self.eps = eps
        
        for iter in range(n_iter):
            before_rss = self.rss
            for j in range(self.number_predictors):
                partial_residuals = self.residuals + self.beta[j] * predictors[:, j]
                numer = np.mean(partial_residuals * predictors[:, j])
                denom = np.mean(predictors[:, j] ** 2)
                self.beta[j] = numer / denom
                self.residuals = partial_residuals - self.beta[j] * predictors[:, j]
            
            self.feed_forward()
            if (iter) % 100 == 0 and verbose:
                print(iter + 1, "/", n_iter, 'th interation runs')
                print("rss: ", self.rss)
            if (np.abs(before_rss - self.rss) < eps):
                print("\n --------------------------\n",
                      "training end at iteration", iter,
                      "\n---------------------------\n")
                break
    
    # 'predict'
    # calculates the predicted values for 'predictors_new'
    def predict(self, predictors_new):
        beta = self.beta
        predicted_values = predictors_new.dot(beta)
        return predicted_values
        
