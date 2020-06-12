# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:08:42 2019

@author: Martin Kamp Dalgaard
"""

import numpy as np
import library as lib
import statsmodels.tsa.stattools as ts
import time
from datetime import datetime, timedelta

# =============================================================================
# Chapter 3 - Numerical integration
# =============================================================================

d_values = np.arange(1, 30)
n_values = np.arange(100, 6100, 500)

Int = lib.Integral(d_values=d_values, test_num=100, n_values=n_values,
                   mode="Uniform", save_dicts=True)

int_df_uniform = Int()

Int = lib.Integral(d_values=d_values, test_num=100, n_values=n_values,
                   mode="Normal", save_dicts=True)

int_df_gaussian = Int()

# =============================================================================
# Chapter 4 - Errors and computation times of estimators
# =============================================================================

def compute_errors(r_range, dim=(1000, 4), N=200, mode="k_values"):
    start_time = time.time()
    n, d = dim
    p = 0
    errors_dict = {}
    for estimator in ["B", "KL", "Mu"]:
        errors_dict[estimator] = {}
        for m in ["Gaussian", "Uniform"]:
            errors_dict[estimator][m] = {}
            for r in r_range:
                errors_dict[estimator][m][str(r)] = {}
                if mode[:8] == "k_values":
                    est = lib.entropy(k=r, estimator=estimator,
                                      dim=(n, d), mode=m)
                elif mode[:8] == "d_values":
                    est = lib.entropy(estimator=estimator, dim=(n, r), mode=m)
                elif mode[:8] == "n_values":
                    est = lib.entropy(estimator=estimator, dim=(r, d), mode=m)
                else:
                    return("Unknown mode. Choose k_values or d_values.")
                i_time_start = time.time()
                H = est(synthetic_data={"Generate": True, "Errors": True,
                                        "Return data": False, "N": N},
                        compute_entropy=False, compute_ccdi=False)
                i_time_stop = time.time() - i_time_start
                errors_dict[estimator][m][str(r)]["Errors"] = H["Errors"]
                errors_dict[estimator][m][str(r)]["Time"] = i_time_stop/N
                lib.save_results(errors_dict, mode)
                p += 1
                
                # Printing progress
                elapsed_time = time.time() - start_time
                da = datetime(1,1,1) + timedelta(seconds=elapsed_time)
                print("Progress: {:.2%} done.".format(p/(3*2*len(r_range))))
                print("Elapsed time: %d:%d:%d:%d" % (da.day-1, da.hour,
                                                     da.minute, da.second))
    return(errors_dict)

# k values
k_range = range(1, 21, 1)
errs_k = compute_errors(r_range=k_range, mode="k_values")

# Dimension
dim_range = range(1, 49, 4)
errs_d = compute_errors(r_range=dim_range, mode="d_values")

# Sample size
n_range1 = range(100, 6100, 1000)
errs_n = compute_errors(r_range=n_range1, mode="n_values")

# Low values of n
n_range2 = range(25, 100, 25)
errs_lown = compute_errors(r_range=n_range2, mode="n_values_lown")

# =============================================================================
# Chapter 4 - CCDI of AR processes
# =============================================================================

start_time = time.time()
CCDI_AR = {}
r = np.arange(-1, 1.1, 0.1)
test_iter = 500
o_iter = 0
o_comp = 2*len(r)*3*test_iter # Total number of iterations
u_iter = 0
for est in ["B", "KL"]:
    CCDI_AR[est] = {}
    ent = lib.entropy(estimator=est,
                      nodes={"X": ["1"], "Y": ["2"], "Z": ["3", "4"]})
    iter_max = 50000
    
    for b1 in r:
        b1 = round(b1, 1)
        CCDI_AR[est][str(b1)] = {"case 1": np.zeros(test_iter),
                                 "case 2": np.zeros(test_iter),
                                 "case 3": np.zeros(test_iter),
                                 "ns": []}
        
        # Case 1
        c_iter = 0
        for j in range(iter_max):
            X = ent.AR_data(p1=[0.8,0], p2=[b1,0], p3=[0,0,0], p4=[0,0,0])
            pvals = np.zeros(4)
            for i in range(4):
                res = ts.adfuller(X[:,i])
                pvals[i] = res[1]
                
            if np.all(pvals < 0.05):
                CCDI_AR[est][str(b1)]["case 1"][c_iter] = ent.compute_ccdi(X)
                c_iter += 1
                o_iter += 1
            else:
                u_iter += 1
                u_str = ""
                for i in range(4):
                    if pvals[i] >= 0.05:
                        u_str += "-%d-p:%.4f-b1:%.1f" %(i+1, pvals[i], b1)
                CCDI_AR[est][str(b1)]["ns"].append("Case 1, %s\n" %(u_str))
                
            # Printing process
            if (o_iter+1) % 500 == 0:
                elapsed_time = time.time() - start_time
                d = datetime(1,1,1) + timedelta(seconds=elapsed_time)
                print("Process: {:.2%} done.".format(o_iter/o_comp))
                print("Elapsed time: %d:%d:%d:%d" % (d.day-1, d.hour,
                                                 d.minute, d.second))
        
            lib.save_results(CCDI_AR, "CCDI_AR")
            
            if c_iter == test_iter:
                break
        
        # Case 2
        c_iter = 0
        for j in range(iter_max):
            X = ent.AR_data(p1=[0.8,0], p2=[b1,0.7],
                            p3=[0.6,0,0.8], p4=[0,0,0])
            pvals = np.zeros(4)
            for i in range(4):
                res = ts.adfuller(X[:,i])
                pvals[i] = res[1]
                
            if np.all(pvals < 0.05):
                CCDI_AR[est][str(b1)]["case 2"][c_iter] = ent.compute_ccdi(X)
                c_iter += 1
                o_iter += 1
            else:
                u_iter += 1
                u_str = ""
                for i in range(4):
                    if pvals[i] >= 0.05:
                        u_str += "-%d-p:%.4f-b1:%.1f" %(i+1, pvals[i], b1)
                CCDI_AR[est][str(b1)]["ns"].append("Case 2, %s\n." %(u_str))
                
            # Printing process
            if (o_iter+1) % 500 == 0:
                elapsed_time = time.time() - start_time
                d = datetime(1,1,1) + timedelta(seconds=elapsed_time)
                print("Process: {:.2%} done.".format(o_iter/o_comp))
                print("Elapsed time: %d:%d:%d:%d" % (d.day-1, d.hour,
                                                 d.minute, d.second))
            
            lib.save_results(CCDI_AR, "CCDI_AR")
            
            if c_iter == test_iter:
                break
        
        # Case 3
        c_iter = 0
        for j in range(iter_max):
            X = ent.AR_data(p1=[0.8,0], p2=[b1,0.7],
                            p3=[0.6,0,0.8], p4=[0,0.9,0])
            pvals = np.zeros(4)
            for i in range(4):
                res = ts.adfuller(X[:,i])
                pvals[i] = res[1]
                
            if np.all(pvals < 0.05):
                CCDI_AR[est][str(b1)]["case 3"][c_iter] = ent.compute_ccdi(X)
                c_iter += 1
                o_iter += 1
            else:
                u_iter += 1
                u_str = ""
                for i in range(4):
                    if pvals[i] >= 0.05:
                        u_str += "-%d-p:%.4f-b1:%.1f" %(i+1, pvals[i], b1)
                CCDI_AR[est][str(b1)]["ns"].append("Case 3, %s\n" %(u_str))
            
            # Printing process
            if (o_iter+1) % 500 == 0:
                elapsed_time = time.time() - start_time
                d = datetime(1,1,1) + timedelta(seconds=elapsed_time)
                print("Process: {:.2%} done.".format(o_iter/o_comp))
                print("Elapsed time: %d:%d:%d:%d" % (d.day-1, d.hour,
                                                 d.minute, d.second))
            
            lib.save_results(CCDI_AR, "CCDI_AR")
            
            if c_iter == test_iter:
                break

# =============================================================================
# Chapter 4 - CCDI of actual data
# =============================================================================

def make_map(l, pickle_name, all_subj_sess=True, subjects=None,
             sessions=None, closed=True):
    # These data are not publicly available on GitHub.
    data = lib.load_results("../data/truncated_data")
    
    CCDI_map = np.zeros((len(l), len(l)))
    
    n_ses = 0
    
    if all_subj_sess:
        subjects = range(1,21)
        sessions = range(1,6)
    
    for subject in subjects:
        for session in sessions:
            X = lib.extract_session(data, subject, session, closed)
            X = X[l]
            x_ind = 0
            p_ses = 0
            n_ses += 1
            
            for key1 in X.keys():
                y_ind = 0
                for key2 in X.keys():
                    if key1 == key2:
                        CCDI_map[x_ind, y_ind] = 0
                    else:
                        ent = lib.entropy(X, nodes={"X": [key1], "Y": [key2],
                                                    "Z": None})
                        CCDI_map[x_ind, y_ind] += ent(compute_ccdi=True)
                    
                    y_ind += 1
                    p_ses += 1
                    
                    print("Session {:d}/{:d} {:.2%} done.".format(n_ses,
                                                     sessions[-1]*subjects[-1],
                                                     p_ses/(len(l)**2)))
                    
                    lib.save_results(CCDI_map, pickle_name)
                
                x_ind += 1
        
        print("Overall process: {:.2%} done.".format(subject/subjects[-1]))
    
    CCDI_map /= subjects[-1]*sessions[-1]
    
    lib.save_results(CCDI_map, pickle_name)
    
    return(CCDI_map)

l1 = ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'O1', 'Oz', 'O2']
CCDI_closed = make_map(l1, "CCDI_closed", all_subj_sess=True, closed=True)
CCDI_opened = make_map(l1, "CCDI_opened", all_subj_sess=True, closed=False)
