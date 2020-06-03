# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:42:08 2019

@author: Martin Kamp Dalgaard
"""

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.special as scp_sp
from scipy import stats
from sklearn.linear_model import LinearRegression
import pickle

class entropy:
    def __init__(self, X=None, k=4, dim=(1000,4), estimator="KL",
                 order=5, mode="AR", nodes=None):
        self.X = X
        self.k = k
        self.dim = dim
        self.order = order
        self.mode = mode
        self.nodes = nodes
        if estimator == "B":
            self.estimator = self.B_estimator
        elif estimator == "KL":
            self.estimator = self.KL_estimator
        elif estimator == "Mu":
            self.estimator = self.Mu_estimator
        else:
            raise(NotImplementedError("Please choose B or KL for " \
                                      "the estimators that currently " \
                                      "have been implemented."))
    
    # Synthetic data
    def dataframe(self, X):
        col = [str(i) for i in range(1, np.shape(X)[1]+1)]
        df = pd.DataFrame(X, columns=col)
        return(df)
    
    def generate_data(self, a=0, b=5):
        n, d = self.dim
        if self.mode == "AR":
            X = self.AR_data()
        elif self.mode == "Gaussian":
            # Sigma is assumed to be the identity matrix
            X = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=d)
            X = X.T
            self.ana_val = n/2*(np.log(2*np.pi) + 1)
        elif self.mode == "Uniform":
            X = np.random.uniform(low=a, high=b, size=self.dim)
            self.ana_val = np.log(np.prod(abs(a-b)))
        else:
            print("Unknown mode for data generation.")
            return(None)
        return(X)
    
    def AR_data(self, p1=[0.05,-0.05], p2=[0.3,0.4],
                p3=[0.8,0.06,0.2], p4=[0.2,0.3,0.05]):
        n, d = self.dim
        LP = n//5
        p11, p12 = p1
        p21, p22 = p2
        p31, p32, p33 = p3
        p41, p42, p43 = p4
        X = np.zeros(shape=(n,4))
        W = np.random.normal(size=(n,4))
        X[:2,:] = W[:2,:]
        for t in range(2, n):
            X[t,0] = p11*X[t-1,0] - p12*X[t-2,0] + W[t,0]
            X[t,1] = p21*X[t-1,0] + p22*X[t-1,2] + W[t,1]
            X[t,2] = p31*X[t-1,0]**2 + p32*X[t-1,1] + p33*X[t-1,2] + W[t,2]
            X[t,3] = p41*X[t-1,0] + p42*X[t-1,2] + p43*X[t-1,3] + W[t,3]
        X = X[LP:,:]
        return(X)
    
    # Estimators
    def estimator_setup(self, X, norm=None):
        if type(X) == np.float64:
            n = 1
            d = 1
        else:
            try:
                n, d = np.shape(X)
            except:
                n = np.shape(X)[0]
                d = 1
        V_1 = np.pi**(d/2)/scp_sp.gamma(1 + d/2)
        if n < self.k:
            k = n-1
        else:
            k = self.k
        distances = np.zeros(n)
        for i in range(n):
            if d > 1:
                distance = np.sort(np.linalg.norm(X[i,:] - X, axis=1, ord=norm))
            else:
                distance = np.sort(np.abs(X[i] - X), axis=0)
            try:
                distances[i] = distance[k]
            except:
                distances[i] = 0
        return(n, d, k, V_1, distances)
        
    def B_estimator(self, X):
        n, d, k, V_1, distances = self.estimator_setup(X)
        T = np.zeros(n)
        if n > 1:
            for i in range(n):
                T[i] = np.log((n/k)*V_1*(distances[i]**d))
        else:
            T[0] = np.log(V_1)
        H_B = (1/n)*np.sum(T)
        return(H_B)
    
    def S_estimator(self, X):
        n, d, k, V_1, distances = self.estimator_setup(X)
        if n > 1:
            H_S = (d/n)*np.sum(np.log(distances)) + np.log(V_1) \
                                            + np.log(n) - scp_sp.digamma(k)
        else:
            H_S = np.log(V_1)
        return(H_S)
    
    def KL_estimator(self, X):
        n, d, k, V_1, distances = self.estimator_setup(X)
        if n > 1:
            H_KL = (d/n)*np.sum(np.log(distances)) + np.log(V_1) \
                                + scp_sp.digamma(n) - scp_sp.digamma(k)
        else:
            H_KL = np.log(V_1)
        return(H_KL)
    
    def Mu_estimator(self, X):
        n, d, k, V_1, distances = self.estimator_setup(X)
        v = uniform_grid(d, n)
        if n > 1:
            H_Mu = (d/n)*np.sum(np.log(distances)) + np.log(V_1) + v
        else:
            H_Mu = np.log(V_1)
        return(H_Mu)
	
#	def kpn_estimator(self, X):
#		n, d, k, V_1, distances = self.estimator_setup(X)
#		for i in range(n):
#            if d > 1:
#                dist = np.sort(np.linalg.norm(X[i,:] - X, axis=1, ord=norm))
#            else:
#                dist = np.sort(np.abs(X[i] - X), axis=0)
		
    
    def compute_errors(self, N):
        abs_err = np.zeros(N)
        rel_err = np.zeros(N)
        H = np.zeros(N)
        for i in range(N):
            X = self.generate_data()
            H[i] = self.estimator(X)
            abs_err[i] = abs(self.ana_val - H[i])
            try:
                rel_err[i] = abs(self.ana_val - H[i])/self.ana_val
            except:
                rel_err[i] = "error"
            if (i+1) % (N/10) == 0:
                print("- Iteration %d of %d done." %(i+1, N))
        return({"Analytic value": self.ana_val,
                "Estimates": H,
                "Absolute errors": abs_err,
                "Mean absolute error": np.mean(abs_err),
                "Relative errors": rel_err,
                "Mean relative error": np.mean(rel_err)})
    
    def compute_ccdi(self, X):
        # Computes I(X^n --> Y^n || Z^n)
        # X initially needs to be a dataframe
        # in order to easily switch the columns
        if not isinstance(X, pd.DataFrame):
            X = self.dataframe(X)
        
        nodes = self.nodes
        
        # Switching the columns' order to X, Y, Z
        l = nodes["X"] + nodes["Y"] + nodes["Z"]
        X = X[l]
        
        # We change X to a matrix
        X = X.values
        n, d = np.shape(X)
        X_new = np.zeros((self.order, n, d))
        
        # Creating lagged version of the variables
        # E.g.: np.allclose(X_new[2][5:], X[2:-3])
        for i in range(self.order):
            for j in range(self.order, n):
                X_new[i,j,:] = X[j-i-1,:]
        
        # We need the columns with Y and Z
        dn = l.index(nodes["X"][-1]) + 1
        f_ind = l.index(nodes["Y"][0])
        X_final = X_new[0,:,f_ind:]
        
        for i in range(1, self.order):
            X_res = np.reshape(X_new[i,:,f_ind:], (n,d-dn))
            # Note: iteratively concatenating with itself
            X_final = np.concatenate((X_final, X_res), axis=1)
        
        # Concatening according to Payam's mail
        joint1 = np.concatenate((np.reshape(X[:,1], (n,1)), X_final), axis=1)
        joint2 = X_final
        joint3 = np.concatenate((np.reshape(X[:,0], (n,1)),
                                 np.reshape(X[:,1], (n,1)), X_final), axis=1)
        joint4 = np.concatenate((np.reshape(X[:,0], (n,1)), X_final), axis=1)
        
        H1 = self.estimator(joint1[self.order:,])
        H2 = self.estimator(joint2[self.order:,])
        H3 = self.estimator(joint3[self.order:,])
        H4 = self.estimator(joint4[self.order:,])
        
        return(H1 - H2 - H3 + H4)
    
    def __call__(self, synthetic_data={"Generate": False, "Errors": False,
                                       "Return data": False, "N": 10},
                 compute_entropy=False, compute_ccdi=True):
        generate_data, r_errors, r_data, N = list(synthetic_data.values())
        return_package = {}
        if generate_data:
            X = self.generate_data()
            if r_errors:
                return_package["Errors"] = self.compute_errors(N)
            if compute_ccdi:
                if self.nodes == None or not all(self.nodes.values()):
                    self.nodes = {"X": ["1"], "Y": ["2"], "Z": ["3", "4"]}
                return_package["CCDI"] = self.compute_ccdi(X)
            if r_data:
                return_package["Data"] = X
            if compute_entropy:
                return_package["H"] = self.estimator(X)
            return(return_package)
        else:
            assert(isinstance(self.X, pd.DataFrame))
            assert(isinstance(self.nodes["X"], list))
            assert(isinstance(self.nodes["Y"], list))
            if compute_ccdi == True:
                if self.nodes["Z"] == None:
                    # Setting Z equal to all other nodes
                    l = self.nodes["X"] + self.nodes["Y"]
                    self.nodes["Z"] = [x for x in self.X.keys() if x not in l]
                return(self.compute_ccdi(self.X))
            else:
                print("If data is not generated, the CCDI should be computed.")

class Integral:
    def __init__(self, d_values, n_values, mode="Gaussian", test_num=1000,
                 bins=30, save_dicts=False, return_dicts=False):
        self.d_values = d_values
        self.n_values = n_values
        self.mode = mode
        self.test_num = test_num
        self.bins = bins
        self.save_dicts = save_dicts
        self.return_dicts = return_dicts
    
    def find_center(self, X):
        c = np.mean(X, axis=0)
        m = np.argmin(np.linalg.norm(X - c, axis=1, ord=None))
        return(X[m])
    
    def compute_range(self, ent):
        m1 = np.zeros(100)
        m2 = np.zeros(100)
        for i in range(100):
            X = ent.generate_data()
            c = np.mean(X, axis=0)
            m1[i] = np.min(np.linalg.norm(X - c, axis=1, ord=None))
            m2[i] = np.max(np.linalg.norm(X - c, axis=1, ord=None))
        return(np.min(m1), np.max(m2))
    
    def cfF(self):
        x_dict = {}
        f_dict = {}
        F_dict = {}
        proc = len(self.d_values)
        iters = 0
        
        for d in self.d_values:
            x_dict[str(d)] = {}
            f_dict[str(d)] = {}
            F_dict[str(d)] = {}
            
            for n in self.n_values:
                print("Testing d = %d, n = %d." %(d, n))
                ent = entropy(dim=(n, d), mode=self.mode)
                m1, m2 = self.compute_range(ent)
                
                x_dict[str(d)][str(n)] = np.linspace(0.9*m1, 1.1*m2, n)
                f_dict[str(d)][str(n)] = np.zeros((self.test_num, n))
                F_dict[str(d)][str(n)] = np.zeros((self.test_num, n))
                
                for t in range(self.test_num):
                    X = ent.generate_data()
                    x1 = self.find_center(X)
                    dist = np.sort(np.linalg.norm(X - x1, axis=1, ord=None))
                    f_dict[str(d)][str(n)][t,:] = dist
                    
                    for i in range(n):
                        count = len(X[dist < x_dict[str(d)][str(n)][i]])
                        F_dict[str(d)][str(n)][t,i] = count/n
                    if (t+1) % (self.test_num/2) == 0:
                        print("- Test %d of %d done." %(t+1, self.test_num))
            iters += 1
            print("- Process: {:.2%} done.".format(iters/proc))
            if self.save_dicts:
                save_results(x_dict, "/ch3/x_%s" %self.mode)
                save_results(f_dict, "/ch3/f_dens_%s" %self.mode)
                save_results(F_dict, "/ch3/F_%s" %self.mode)
        for key1 in F_dict.keys():
            for key2 in F_dict[key1].keys():
                F_dict[key1][key2] = (np.mean(F_dict[key1][key2], axis=0),
                                      np.var(F_dict[key1][key2], axis=0))
                f_dict[key1][key2] = (np.mean(f_dict[key1][key2], axis=0),
                                      np.var(f_dict[key1][key2], axis=0))

        return(x_dict, f_dict, F_dict)
    
    def compute_integral(self, x_dict, f_dict, F_dict):
        f_h = {}
        F_ly = {}
        f_y = {}
        product = {}
        int_dict = {}
        
        for key1 in F_dict.keys():
            f_h[key1] = {}
            F_ly[key1] = {}
            f_y[key1] = {}
            product[key1] = {}
            int_dict[key1] = {}
            
            for key2 in F_dict[key1].keys():
                f_h[key1][key2] = np.histogram(f_dict[key1][key2][0],
                                               density=True, bins=self.bins)
                
                F_ly[key1][key2] = np.log(F_dict[key1][key2][0])
                F_ly[key1][key2][(F_ly[key1][key2]==-np.inf)] = 0
                
                freq, bins = f_h[key1][key2][0], f_h[key1][key2][1]
                freq = np.concatenate((np.array([0]), freq))
                f_y[key1][key2] = np.interp(x_dict[key1][key2], bins, freq)
                
                product[key1][key2] = f_y[key1][key2]*F_ly[key1][key2]
                
                I = InterpolatedUnivariateSpline(x_dict[key1][key2],
                                                 product[key1][key2])
                int_dict[key1][key2] = I.integral(0, np.max(x_dict[key1][key2]))
            
        idxs = self.d_values
        cols = self.n_values
        df = pd.DataFrame(np.zeros((len(idxs), len(cols))),
                          index=idxs, columns=cols)
        for idx in df.index:
            df.loc[idx] = int_dict[str(idx)].values()
        
        return(df)
    
    def __call__(self):
        x_dict, f_dict, F_dict = self.cfF()
        int_df = self.compute_integral(x_dict, f_dict, F_dict)
        if self.save_dicts:
            save_results(int_df, "/ch3/integral_%s" %self.mode)
        if self.return_dicts:
            return(int_df, x_dict, f_dict, F_dict)
        else:
            return(int_df)

# Function for extracting a single session with eyes open/closed
def extract_session(data, subject_number, session_number, closed=True):
    # subject: int between 1 and 20
    # session_number: int between 1 and 5
    # closed: whether the eyes are closed or not
    df = data["subject_%s" %subject_number]
    si = list(df[(df["EyesClosed"] == 1)|(df["EyesOpened"] == 1)].index)
    eci = list(df[df["EyesClosed"] == 1].index)
    eoi = list(df[df["EyesOpened"] == 1].index)
    
    if closed:
        session_index = eci[session_number-1]
    else:
        session_index = eoi[session_number-1]
        
    if si[si.index(session_index)] == si[-1]:
        df_session = df[session_index:]
    else:
        df_session = df[session_index:si[si.index(session_index)+1]]
    
    return(df_session)

def save_results(results, filename):
    with open('pickle/%s.pickle' %(filename), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(filename):
    with open("pickle/%s.pickle" %(filename), "rb") as input_file:
        file = pickle.load(input_file)
    return(file)

def compute_ci(array, axis=1, level=0.05):
    n = len(array)
    m = np.mean(array, axis=axis)
    s = np.sqrt(np.var(array, axis=axis))/np.sqrt(n)
    t = stats.t.ppf(1 - level/2, n-1)
    uci = m + t*s
    lci = m - t*s
    return(uci, lci)

def uniform_grid(d, n):
    if "grid" not in globals().keys():
        grid = load_results("ch3/integral_Uniform")
    else:
        grid = globals()["grid"]
    return(grid.loc[d][n])

def p_nearest(X, idx, p):
    n, d = np.shape(X)
    dictionary = {}
    for i in range(d):
        dictionary[str(i)] = X[:,i]
    dist = np.linalg.norm(X[idx,:] - X, axis=1, ord=None)
    dictionary["Dist"] = dist
    df = pd.DataFrame(dictionary)
    sorted_df = df.sort_values(by="Dist")
    df_p = sorted_df[:p]
    return(df_p.drop(columns=["Dist"]).values)