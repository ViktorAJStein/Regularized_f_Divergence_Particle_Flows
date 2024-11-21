import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch


def plot_lambdas(lambdas, lower_bds_lambd, plot_name, folder_name):
    fig, ax = plt.subplots()
    plt.plot(lambdas, label=r'$\lambda$')
    plt.plot(lower_bds_lambd, label=r'lower bound on $\lambda$')
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    plt.yscale('log')
    plt.xlabel('iterations')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(frameon=False)
    plt.savefig(folder_name + plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_gaps(a, duality_gaps, pseudo_duality_gaps, relative_duality_gaps, plot_name, folder_name):
    fig, ax = plt.subplots()
    if not a == '':
        plt.plot(duality_gaps, label='duality gap')
        plt.plot(relative_duality_gaps, '-.', label='relative duality gap')
    plt.plot(pseudo_duality_gaps, ':', label='pseudo duality gap')
    plt.plot(relative_pseudo_duality_gaps, label='relative pseudo-duality gap')
    plt.axhline(y=1e-2, linestyle='--', color='gray', label='tolerance')
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(folder_name + plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_W2(W2, plot_name, folder_name):
    fig, ax = plt.subplots()
    plt.plot(W2)
    plt.yscale('log')
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    plt.xlabel('iterations')
    plt.ylabel(r'$W_2(\mu, \nu)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(folder_name + plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_func_values(a, dual_values, pseudo_dual_values, func_values, lambd, plot_name, folder_name):
    fig, ax = plt.subplots()
    if not a == '':
        plt.plot(dual_values, label='dual objective')
    else:
        plt.plot(pseudo_dual_values, label='pseudo dual values')
    plt.plot(func_values, '--', label='primal objective')
    plt.yscale('log')
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    plt.xlabel('iterations')
    plt.ylabel(r'$D_{f_{\alpha}}^{{' + str(lambd) + r'}}(\mu \mid \nu)$')
    plt.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(folder_name + plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_MMD(MMD, plot_name, folder_name):
    fig, ax = plt.subplots()
    plt.plot(MMD)
    plt.xlabel('iterations')
    plt.ylabel(r'$\frac{1}{2} d_{K}(\mu, \nu)^2$')
    plt.yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    plt.savefig(folder_name + plot_name, dpi=300, bbox_inches='tight')
    plt.close()