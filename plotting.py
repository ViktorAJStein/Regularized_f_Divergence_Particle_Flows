import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch


def plot_particles(X, target, grad, target_name, img_name, folder_name, arrows):
    d = X.shape[1]
    if d == 2:
        plt.figure()
        plt.plot(target[:, 1], target[:, 0], '.', c='orange', ms=2)
        plt.plot(X[:, 1], X[:, 0], 'b.', ms=2)
        if arrows and n > 0:
            minus_grad = - grad.cpu()
            plt.quiver(X[:, 1], X[:, 0], minus_grad[:, 1], minus_grad[:, 0], angles='xy', scale_units='xy', scale=1)
        if target_name == 'circles':
            plt.ylim([-1.0, 1.0])
            plt.xlim([-2.0, 0.5])
        if target_name == 'bananas':
            plt.ylim([-7.5, 7.5])
            plt.xlim([-5.0, 10.0])
        if target_name == 'four_wells':
            plt.xlim([-2.5, 7.0])
            plt.ylim([-2.5, 7.0])
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
        plt.close()
    elif d == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        fig.add_axes(ax)
        ax.view_init(azim=-66, elev=12)
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='orange', s=2)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], 'b.', s=2)
        plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
        plt.close()

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
    

def plot_gaps(a, duality_gaps, relative_pseudo_duality_gaps, pseudo_duality_gaps, relative_duality_gaps, plot_name, folder_name):
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
    
    
def plot_func_values(dual, dual_values, pseudo_dual_values, func_values, lambd, plot_name, folder_name):
    fig, ax = plt.subplots()
    if dual:
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