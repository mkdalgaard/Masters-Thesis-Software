# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 04:06:00 2020

@author: Martin Kamp Dalgaard
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from decimal import Decimal
import scipy.optimize as opt
import scipy.special as scp_sp
import scipy.stats
import library2 as lib

# =============================================================================
# Chapter 3 - Numerical integration
# =============================================================================

x_G = lib.load_results("ch3/x_Gaussian")
f_G = lib.load_results("ch3/f_dens_Gaussian")
F_G = lib.load_results("ch3/F_Gaussian")

x_u = lib.load_results("ch3/x_Uniform")
f_u = lib.load_results("ch3/f_dens_Uniform")
F_u = lib.load_results("ch3/F_Uniform")

# Histograms
d_values = ["1", "17", "45"]
colors = [(0, 0, 0.9, 1), (1, 0.647, 0, 0.75), (0, 0.5, 0, 0.75)]
plt.figure()
i = 0
for key in d_values:
    if key != "1":
        m = np.mean(f_G[key]["1100"][0])
        s = np.var(f_G[key]["1100"][0])
        r = np.arange(0, 11, 0.001)
        plt.plot(r, scipy.stats.norm.pdf(r, m, s))
    plt.hist(f_G[key]["1100"][0], label="$d=%s$" %key,
             fc=colors[i], density=True, bins=20)
    i += 1
plt.legend()
plt.title("Multivariate normal random variables")
plt.xlabel("$r_1$")
plt.ylabel("$\hat{f}(r_1|\mathbf{x}_1)$")
plt.savefig("output/f_G_hist.png", dpi=500)

plt.figure()
i = 0
for key in d_values:
    if key != "1":
        m = np.mean(f_u[key]["1100"][0])
        s = np.var(f_u[key]["1100"][0])
        r = np.arange(0, 16, 0.001)
        plt.plot(r, scipy.stats.norm.pdf(r, m, s))
    plt.hist(f_u[key]["1100"][0], label="$d=%s$" %key,
             fc=colors[i], density=True, bins=20)
    i += 1
plt.legend()
plt.title("Uniform random variables")
plt.xlabel("$r_1$")
plt.ylabel("$\hat{f}(r_1|\mathbf{x}_1)$")
plt.savefig("output/f_u_hist.png", dpi=500)

# Plots of F
d_values = ["1", "9", "17"]
plt.figure()
for key in d_values:
    plt.plot(x_G[key]["1100"], F_G[key]["1100"][0], label="$d=%s$" %key)
plt.legend()
plt.title("Multivariate normal random variables")
plt.xlabel("$r_1$")
plt.ylabel("$\hat{F}(r_1|\mathbf{x}_1)$")
plt.savefig("output/F_G.png", dpi=500)

plt.figure()
for key in d_values:
    plt.plot(x_u[key]["1100"], F_u[key]["1100"][0], label="$d=%s$" %key)
plt.legend()
plt.title("Uniform random variables")
plt.xlabel("$r_1$")
plt.ylabel("$\hat{F}(r_1|\mathbf{x}_1)$")
plt.savefig("output/F_u.png", dpi=500)

# Plots of the integral values
int_uniform = lib.load_results("ch3/integral_Uniform")
int_normal = lib.load_results("ch3/integral_Gaussian")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(18,6))
fig.subplots_adjust(wspace=0.05)
fig.suptitle("Integral values of $\hat{f}(r_1|\mathbf{x}_1) \cdot" \
             "\ln(\hat{F}(r_1|\mathbf{x}_1))$")

uci, lci = lib.compute_ci(int_uniform.values, axis=1)

ax1.plot(int_uniform.mean(axis=1), ".-r")
ax1.fill_between(int_uniform.index, lci, uci)
ax1.tick_params(axis='x', which='both', bottom=False,
                top=False, labelbottom=False)
ax1.set_ylabel("Uniform random variables")
ax1.set_title("Mean across values of $n$")

uci, lci = lib.compute_ci(int_uniform.values, axis=0)

ax2.plot(int_uniform.mean(axis=0), ".-r")
ax2.fill_between(int_uniform.columns, lci, uci)
ax2.tick_params(axis='x', which='both', bottom=False,
                top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', left=False,
                right=False, labelbottom=False)
ax2.set_title("Mean across values of $d$")

uci, lci = lib.compute_ci(int_normal.values, axis=1)

ax3.plot(int_normal.mean(axis=1), ".-r")
ax3.fill_between(int_normal.index, lci, uci)
ax3.set_xlabel("$d$")
ax3.set_ylabel("Multivariate normal random variables")

uci, lci = lib.compute_ci(int_normal.values, axis=0)

ax4.plot(int_normal.mean(axis=0), ".-r")
ax4.fill_between(int_normal.columns, lci, uci)
ax4.tick_params(axis='y', which='both', left=False,
                right=False, labelbottom=False)
ax4.set_xlabel("$n$")

fig.savefig("output/integral_means.pdf", bbox_inches="tight")

m = np.mean(int_uniform.values, axis=None)
uci, lci = lib.compute_ci(int_uniform.values, axis=None)
print("Mean, confidence interval for entire uniform grid: %.5f, [%.5f, %.5f]."
                                                                %(m, uci, lci))

m = np.mean(int_normal.values, axis=None)
uci, lci = lib.compute_ci(int_normal.values, axis=None)
print("Mean, confidence interval for entire normal grid: %.5f, [%.5f, %.5f]."
                                                                %(m, uci, lci))

# =============================================================================
# Chapter 3 - Difference between ln(n) and psi(n)
# =============================================================================

n = np.arange(1, 51, 1)
ln = np.log(n)
psi = scp_sp.digamma(n)

plt.figure()
plt.plot(n, ln, label="$\ln(n)$")
plt.plot(n, psi, label="$\psi(n)$")
plt.xlabel("$n$")
plt.legend()
plt.savefig("output/ln_psi.png", dpi=500)

plt.figure()
plt.plot(n, np.abs(ln - psi))
plt.xlabel("$n$")
plt.ylabel("$|\ln(n) - \psi(n)|$")
plt.savefig("output/ln_psi_diff.png", dpi=500)

# =============================================================================
# Chapter 4 - Errors and computation times of estimators
# =============================================================================

errs_k = lib.load_results("ch4/k_values")
errs_d = lib.load_results("ch4/d_values")
errs_n = lib.load_results("ch4/n_values")
errs_lown = lib.load_results("ch4/n_values_lown")

# Functions for extracting and plotting results
def extract_results(errors_dict, res_t="Errors"):
    idx = list(errors_dict["B"]["Gaussian"].keys())
    G_dict = {}
    u_dict = {}
    for key1 in errors_dict.keys():
        i = 0
        G_arr = np.zeros(len(idx))
        G_ci_arr1 = np.zeros(len(idx))
        G_ci_arr2 = np.zeros(len(idx))
        U_arr = np.zeros(len(idx))
        U_ci_arr1 = np.zeros(len(idx))
        U_ci_arr2 = np.zeros(len(idx))
        for key2 in errors_dict[key1]["Gaussian"].keys():
            if res_t == "Errors":
                G_arr[i] = errors_dict[key1]["Gaussian"] \
                                [key2][res_t]["Mean absolute error"]
                G_ci_arr1[i], G_ci_arr2[i] = lib.compute_ci(errors_dict[key1] \
                                 ["Gaussian"][key2][res_t]["Absolute errors"],
                                                                        axis=0)
                U_arr[i] = errors_dict[key1]["Uniform"] \
                                [key2][res_t]["Mean absolute error"]
                U_ci_arr1[i], U_ci_arr2[i] = lib.compute_ci(errors_dict[key1] \
                                 ["Uniform"][key2][res_t]["Absolute errors"],
                                                                        axis=0)
            elif res_t == "Time":
                G_arr[i] = errors_dict[key1]["Gaussian"][key2][res_t]
                U_arr[i] = errors_dict[key1]["Uniform"][key2][res_t]
            i += 1
        G_dict[key1 + "_G"] = G_arr
        u_dict[key1 + "_u"] = U_arr
        if res_t == "Errors":
            G_dict[key1 + "_G_ul"] = G_ci_arr1
            G_dict[key1 + "_G_ll"] = G_ci_arr2
            u_dict[key1 + "_u_ul"] = U_ci_arr1
            u_dict[key1 + "_u_ll"] = U_ci_arr2
    
    df = pd.concat((pd.DataFrame(G_dict, index=idx),
                    pd.DataFrame(u_dict, index=idx)), axis=1)
    return(df)

def plot_figure(title, dictionary, xlabel, filename,
                xticks=None, xticks_pos=None, plot_ci=True):
    x_value = dictionary.index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.58)
    fig.text(0.04, 0.5, title, va='center', rotation='vertical')
    
    ax1.plot(x_value, dictionary["B_G"], ".-g", label="$\hat{h}_{B}$")
    ax1.plot(x_value, dictionary["KL_G"], ".-y", label="$\hat{h}_{KL}$")
    ax1.plot(x_value, dictionary["Mu_G"], ".-b", label="$\hat{h}_{Mu}$")
    if plot_ci:
        ax1.fill_between(x_value, dictionary["B_G_ul"], dictionary["B_G_ll"])
        ax1.fill_between(x_value, dictionary["KL_G_ul"], dictionary["KL_G_ll"])
        ax1.fill_between(x_value, dictionary["Mu_G_ul"], dictionary["Mu_G_ll"])
    ax1.xaxis.set_ticks_position('none')
    ax1.set_title("Multivariate normal random variables, " \
                  "$\hat{h}(\mathbf{X}_{n})$")
    
    ax2.plot(x_value, dictionary["B_u"], ".-g", label="$\hat{h}_{B}$")
    ax2.plot(x_value, dictionary["KL_u"], ".-y", label="$\hat{h}_{KL}$")
    ax2.plot(x_value, dictionary["Mu_u"], ".-b", label="$\hat{h}_{Mu}$")
    if plot_ci:
        ax2.fill_between(x_value, dictionary["B_u_ul"], dictionary["B_u_ll"])
        ax2.fill_between(x_value, dictionary["KL_u_ul"], dictionary["KL_u_ll"])
        ax2.fill_between(x_value, dictionary["Mu_u_ul"], dictionary["Mu_u_ll"])
    if (isinstance(xticks, np.ndarray)) & (isinstance(xticks_pos, np.ndarray)):
        ax2.set_xticks(xticks_pos)
        ax2.set_xticklabels(xticks)
    ax2.set_xlabel(xlabel)
    ax2.set_title("Uniform random variables, $\hat{h}(\mathbf{X}_{u})$")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.53),
               ncol=3, fancybox=True)
    fig.savefig("output/%s.png" %filename, dpi=500)

# Mean absolute errors
mae_k = extract_results(errors_dict=errs_k)
mae_d = extract_results(errors_dict=errs_d)
mae_n = extract_results(errors_dict=errs_n)
mae_ln = extract_results(errors_dict=errs_lown)

plot_figure("Mean absolute error", mae_k, "$k$", "k_values",
            np.arange(1, 21, 2), np.arange(0, 20, 2))
plot_figure("Mean absolute error", mae_d, "$d$", "d_values")

mae_n_combined = pd.concat((mae_ln, mae_n))
plot_figure("Mean absolute error", mae_n_combined, "$n$", "n_values",
            mae_n_combined.index, np.arange(0, 10, 1))

# Analysis of linear relationship
def compute_coefs(dictionary, keys):
    coef_dict = {"Increase": np.zeros(6), "Coefficient": np.zeros(6)}
    i = 0
    for key in keys:
        LinReg = LinearRegression()
        x = np.array(dictionary[key].index).reshape(-1, 1)
        y = dictionary[key].values
        LinReg.fit(x, y)
        coef_dict["Increase"][i] = LinReg.coef_
        coef_dict["Coefficient"][i] = LinReg.score(x, y)
        i += 1
    return(pd.DataFrame(coef_dict, index=keys))

lin_reg_coefs = compute_coefs(mae_d, ["B_G", "KL_G", "Mu_G",
                                      "B_u", "KL_u", "Mu_u"])

np.mean(lin_reg_coefs.loc["B_G":"Mu_G"]["Increase"].values)
np.mean(lin_reg_coefs.loc["B_u":"Mu_u"]["Increase"].values)
np.mean(lin_reg_coefs.loc["B_G":"Mu_G"]["Coefficient"].values)
np.mean(lin_reg_coefs.loc["B_u":"Mu_u"]["Coefficient"].values)

# Norms of errors
df = pd.DataFrame(index=["d", "k", "n"], columns=["B_G", "B_u", "KL_G",
                                                  "KL_u", "Mu_G", "Mu_u"])

for m in ["B_G", "B_u", "KL_G", "KL_u", "Mu_G", "Mu_u"]:
    df.loc["d"][m] = np.linalg.norm(mae_d[m])
for m in ["B_G", "B_u", "KL_G", "KL_u", "Mu_G", "Mu_u"]:
    df.loc["k"][m] = np.linalg.norm(mae_k[m])
for m in ["B_G", "B_u", "KL_G", "KL_u", "Mu_G", "Mu_u"]:
    df.loc["n"][m] = np.linalg.norm(mae_n_combined[m])

width = 0.35
x = [1, 3]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(18,6))

ax1.bar(x[0] - width, df.loc["d"]["B_G"], width, color="g")
ax1.bar(x[0], df.loc["d"]["KL_G"], width, color="y")
ax1.bar(x[0] + width, df.loc["d"]["Mu_G"], width, color="b")

ax1.bar(x[1] - width, df.loc["d"]["B_G"], width, color="g")
ax1.bar(x[1], df.loc["d"]["KL_G"], width, color="y")
ax1.bar(x[1] + width, df.loc["d"]["Mu_G"], width, color="b")
ax1.set_xticks(x)
ax1.set_xticklabels(["Normal", "Uniform"])
ax1.set_title("Values of $d$")

ax2.bar(x[0] - width, df.loc["k"]["B_G"], width, color="g")
ax2.bar(x[0], df.loc["k"]["KL_G"], width, color="y")
ax2.bar(x[0] + width, df.loc["k"]["Mu_G"], width, color="b")

ax2.bar(x[1] - width, df.loc["k"]["B_G"], width,
        color="g", label="$\hat{h}_{B}$")
ax2.bar(x[1], df.loc["k"]["KL_G"], width,
        color="y", label="$\hat{h}_{KL}$")
ax2.bar(x[1] + width, df.loc["k"]["Mu_G"], width,
        color="b", label="$\hat{h}_{Mu}$")
ax2.set_xticks(x)
ax2.legend(loc='center', bbox_to_anchor=(0.5, -0.1),
           ncol=3, fancybox=True)
ax2.set_xticklabels(["Normal", "Uniform"])
ax2.set_title("Values of $k$")

ax3.bar(x[0] - width, df.loc["d"]["B_G"], width, color="g")
ax3.bar(x[0], df.loc["d"]["KL_G"], width, color="y")
ax3.bar(x[0] + width, df.loc["d"]["Mu_G"], width, color="b")

ax3.bar(x[1] - width, df.loc["n"]["B_G"], width, color="g")
ax3.bar(x[1], df.loc["n"]["KL_G"], width, color="y")
ax3.bar(x[1] + width, df.loc["n"]["Mu_G"], width, color="b")
ax3.set_xticks(x)
ax3.set_xticklabels(["Normal", "Uniform"])
ax3.set_title("Values of $n$")

fig.savefig("output/errors_norms.pdf", bbox_inches="tight")

# Computation times
time_n = extract_results(errors_dict=errs_n, res_t="Time")
time_d = extract_results(errors_dict=errs_d, res_t="Time")
time_k = extract_results(errors_dict=errs_k, res_t="Time")

plot_figure("Computation time (seconds)", time_n,
            "$n$", "n_time", plot_ci=False)
plot_figure("Computation time (seconds)", time_d,
            "$d$", "d_time", plot_ci=False)
plot_figure("Computation time (seconds)", time_k, "$k$", "d_time",
            np.arange(1, 21, 2), np.arange(0, 20, 2), plot_ci=False)

# Fitting the computation times when n is changed to different functions
def twod_poly(x, a, b, c):
    return(a*x**2 + b*x + c)

def nlogn(x, a, b):
    return(a*x*np.log2(b*x))

idx = np.array(time_n.index, dtype=np.int)
values = time_n["KL_G"].values

popt_twod, pcov = opt.curve_fit(twod_poly, idx, values)
popt_nlogn, pcov = opt.curve_fit(nlogn, idx, values)
y_twod = twod_poly(idx, *popt_twod)
y_nlogn = nlogn(idx, *popt_nlogn)

plt.figure()
plt.plot(time_n["KL_G"], label="Actual values")
plt.plot(time_n.index, y_nlogn,
         label="$(%.2E) \cdot n \cdot \log_2((%.2E) \cdot n)$"
         %(Decimal(popt_nlogn[0]), Decimal(popt_nlogn[1])))
plt.plot(time_n.index, y_twod,
         label="$(%.2E) \cdot n^2 + (%.2E) \cdot n + (%.2E)$"
         %(Decimal(popt_twod[0]), Decimal(popt_twod[1]), Decimal(popt_twod[2])))
plt.xlabel("$n$")
plt.ylabel("Computation time (seconds)")
plt.xticks(time_n.index)
plt.legend()
plt.savefig("output/Computation_fit_n.png", dpi=500)

def sqrt(x, a, b):
    return(np.sqrt(a*x) + b)

idx = np.array(time_d.index, dtype=np.int)
values = time_d["B_G"].values

popt_sqrt, pcov = opt.curve_fit(sqrt, idx, values)

y_logd = sqrt(idx, *popt_sqrt)

plt.figure()
plt.plot(time_d["B_G"], label="Actual values")
plt.plot(time_d.index, y_logd,
         label="$\sqrt{(%.2E) \cdot d} + (%.2E)$"
         %(Decimal(popt_sqrt[0]), Decimal(popt_sqrt[1])))
plt.xlabel("$d$")
plt.ylabel("Computation time (seconds)")
plt.xticks(time_d.index)
plt.legend()
plt.savefig("output/Computation_fit_d.png", dpi=500)

# =============================================================================
# Chapter 4 - CCDI of AR processes
# =============================================================================

CCDI_AR = lib.load_results("ch4/CCDI_AR")

def extract_AR_results(errors_dict, case):
    ind = np.zeros(len(errors_dict["B"].keys()))
    dictio = {}
    for key1 in errors_dict.keys():
        i = 0
        CI = np.zeros((len(ind),2))
        arr = np.zeros(len(ind))
        for key2 in errors_dict[key1].keys():
            ind[i] = float(key2)
            arr[i] = np.mean(errors_dict[key1][key2][case])
            CI[i,:] = lib.compute_ci(errors_dict[key1][key2][case], axis=0)
            i += 1
        dictio[key1] = arr
        dictio[key1 + "_ul"] = CI[:,0]
        dictio[key1 + "_ll"] = CI[:,1]
    df = pd.DataFrame(dictio, index=ind)
    return(df)

mae_CCDI_AR1 = extract_AR_results(CCDI_AR, "case 1")
mae_CCDI_AR2 = extract_AR_results(CCDI_AR, "case 2")
mae_CCDI_AR3 = extract_AR_results(CCDI_AR, "case 3")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18,6))
fig.subplots_adjust(wspace=0.1)

ax1.plot(mae_CCDI_AR1["B"], ".-g")
ax1.plot(mae_CCDI_AR1["KL"], ".-y")
ax1.plot(mae_CCDI_AR1["Mu"], ".-b")
ax1.fill_between(mae_CCDI_AR1.index,
                 mae_CCDI_AR1["Mu_ll"],
                 mae_CCDI_AR1["Mu_ul"])
ax1.set_xlabel("$b_1$")
ax1.set_xticks(np.arange(-1.0, 1.5, 0.5))
ax1.set_ylabel("Causal conditional directed information")
ax1.set_yticks(np.arange(-0.02, 0.24, 0.04))
ax1.set_title("Case 1")

ax2.plot(mae_CCDI_AR2["B"], ".-g")
ax2.plot(mae_CCDI_AR2["KL"], ".-y")
ax2.plot(mae_CCDI_AR2["Mu"], ".-b")
ax2.fill_between(mae_CCDI_AR2.index,
                 mae_CCDI_AR2["Mu_ll"],
                 mae_CCDI_AR2["Mu_ul"])
ax2.set_xlabel("$b_1$")
ax2.set_xticks(np.arange(-1.0, 1.5, 0.5))
ax2.tick_params(axis='y', which='both', left=False,
                right=False, labelbottom=False)
ax2.set_title("Case 2")

ax3.plot(mae_CCDI_AR3["B"], ".-g", label=r"$\hat{h}_B$")
ax3.plot(mae_CCDI_AR3["KL"], ".-y", label=r"$\hat{h}_{KL}$")
ax3.plot(mae_CCDI_AR3["Mu"], ".-b", label=r"$\hat{h}_{Mu}$")
ax3.fill_between(mae_CCDI_AR3.index,
                 mae_CCDI_AR3["Mu_ll"],
                 mae_CCDI_AR3["Mu_ul"])
ax3.set_xlabel("$b_1$")
ax3.set_xticks(np.arange(-1.0, 1.5, 0.5))
ax3.tick_params(axis='y', which='both', left=False,
                right=False, labelbottom=False)
ax3.legend()
ax3.set_title("Case 3")

fig.savefig("output/CCDI_AR.pdf", bbox_inches="tight")

# =============================================================================
# Chapter 4 - CCDI of actual data
# =============================================================================

CCDI_opened = lib.load_results("ch4/CCDI_opened")
CCDI_closed = lib.load_results("ch4/CCDI_closed")

def plot_CCDI(array, l, mode):
    fig, ax = plt.subplots()
    im = ax.pcolor(array, lw=1)
    fig.colorbar(im, ax=ax)
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set(ticks=np.arange(0.5, len(l)), ticklabels=l)
    
    fig.suptitle(r"$DI\left(X^n \rightarrow Y^n \parallel Z^n\right)$",
                 fontsize=14, fontweight='bold')
    ax.set_title(r"$Z^n$ is the other 6 electrodes besides $X^n$ and $Y^n$",
                 fontsize=10)
    ax.set_xlabel(r"$X^n$")
    ax.set_ylabel(r"$Y^n$")
    
    plt.savefig("output/CCDI_%s.png" %mode, dpi=500)

CCDI_closed_norm = CCDI_closed/np.max(CCDI_closed)
CCDI_opened_norm = CCDI_opened/np.max(CCDI_opened)

l1 = ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'O1', 'Oz', 'O2']
plot_CCDI(CCDI_closed_norm, l1, "closed")
plot_CCDI(CCDI_opened_norm, l1, "opened")

# =============================================================================
# Appendix A - Stationary processes
# =============================================================================

np.random.seed(1111) # For reproducibility
W = np.random.normal(size=(1000,3))
X = np.zeros((1000,3))
X[0,:] = W[0,:]

for i in range(1,len(X)):
    X[i,0] = 0.7*X[i-1,0] + W[i,0]
    X[i,1] = 1.0*X[i-1,1] + W[i,1]

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.1)
ax1.plot(X[:,0], label=r"$\phi=0.7$")
ax1.set_title(r"$X(t) = \phi X(t-1) + W(t)$")
ax1.legend()
ax1.tick_params(axis='x', which='both', bottom=False,
                top=False, labelbottom=False)
ax2.plot(X[:,1], label=r"$\phi=1.0$")
ax2.set_xlabel("$t$")
ax2.legend()
fig.savefig("output/AR_exam.png", dpi=500)

# =============================================================================
# Appendix B - Length of sessions before being truncated
# =============================================================================

merged_data = lib.load_results("/../data/merged_data")

subs = 20
sess = 10

idxs = np.zeros((subs,sess+1))

for subject in range(1, subs+1):
    df = merged_data["subject_%s" %subject]
    sub_idxs = list(df[(df["EyesClosed"]==1) | (df["EyesOpened"]==1)].index)
    sub_idxs.append(len(df))
    idxs[subject-1,:] = sub_idxs

sess_length = np.zeros((subs,sess))

for i in range(subs):
    for j in range(sess):
        sess_length[i,j] = idxs[i,j+1] - idxs[i,j]

sess_length = sess_length.reshape((200))

plt.hist(sess_length, bins=30)
plt.xlabel("Number of samples")
plt.ylabel("Frequency")
plt.savefig("output/session_lengths.png", dpi=500)

# =============================================================================
# Appendix B - Shapiro-Wilk test for normality of errors
# =============================================================================

def absolute_errors(dictionary, estimator, mode, number):
    return(dictionary[estimator][mode][number]["Errors"]["Absolute errors"])

def perform_shapiro_test(dictionary, index):
    df = pd.DataFrame(index=index, columns=["B_G", "KL_G", "Mu_G",
                                            "B_U", "KL_U", "Mu_U"])
    for est in ["B", "KL", "Mu"]:
        for mode in ["Gaussian", "Uniform"]:
            for idx in df.index:
                df.loc[idx]["%s_%s" %(est, mode[0])] = \
                scipy.stats.shapiro(absolute_errors(dictionary, est,
                                                    mode, str(idx)))[1]
    return(df)

shapiro_d = perform_shapiro_test(errs_d, mae_d.index)
shapiro_k = perform_shapiro_test(errs_k, mae_k.index)
shapiro_n = perform_shapiro_test(errs_n, mae_n.index)
shapiro_lown = perform_shapiro_test(errs_lown, range(25, 100, 25))
shapiro_n = pd.concat((shapiro_lown, shapiro_n))

def plot_shapiro(df, xlabel, filename, xticks=None, xticks_pos=None):
    x = df.index
    pval = np.repeat(0.05, len(x))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.58)
    fig.text(0.04, 0.5, "P-values", va='center', rotation='vertical')
    
    ax1.plot(x, pval, "-r")
    ax1.plot(df["B_G"], ".g", label="$\hat{h}_{B}$")
    ax1.plot(df["KL_G"], ".y", label="$\hat{h}_{KL}$")
    ax1.plot(df["Mu_G"], ".b", label="$\hat{h}_{Mu}$")
    ax1.xaxis.set_ticks_position('none')
    ax1.set_title("Multivariate normal random variables, " \
                  "$\hat{h}(\mathbf{X}_{n})$")
    
    ax2.plot(x, pval, "-r")
    ax2.plot(df["B_U"], ".g", label="$\hat{h}_{B}$")
    ax2.plot(df["KL_U"], ".y", label="$\hat{h}_{KL}$")
    ax2.plot(df["Mu_U"], ".b", label="$\hat{h}_{Mu}$")
    ax2.set_title("Uniform random variables, $\hat{h}(\mathbf{X}_{u})$")
    ax2.set_xlabel(xlabel)
    if (isinstance(xticks, np.ndarray)) & (isinstance(xticks_pos, np.ndarray)):
        ax2.set_xticks(xticks_pos)
        ax2.set_xticklabels(xticks)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.53),
               ncol=3, fancybox=True)
    fig.savefig("output/%s.png" %filename, dpi=500)

plot_shapiro(shapiro_d, "$d$", "shapiro_test_d")
plot_shapiro(shapiro_k, "$k$", "shapiro_test_k",
             xticks=np.arange(1, 21, 2), xticks_pos=np.arange(0, 20, 2))
plot_shapiro(shapiro_n, "$n$", "shapiro_test_n")

val_d = shapiro_d.values.reshape(-1,1)
val_k = shapiro_k.values.reshape(-1,1)
val_n = shapiro_n.values.reshape(-1,1)

print("Percentage for d: {:.2%}.".format(len(val_d[val_d < 0.05])/len(val_d)))
print("Percentage for k: {:.2%}.".format(len(val_k[val_k < 0.05])/len(val_k)))
print("Percentage for n: {:.2%}.".format(len(val_n[val_n < 0.05])/len(val_n)))
