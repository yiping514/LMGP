# LMGP Visualization
# 
#
#

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_sep(positions, levels, perm, constraints_flag = True, suptitle= None):
    # 
    # plot latent values
    if positions.ndim > 2:
        positions = positions.mean(axis = 0)
    else:
        positions = positions.detach()

    if positions.ndim > 2:
        positions = positions.mean(axis = 0) 

    # applying the constrains
    if constraints_flag:
        positions = constrains(positions)


    positions = positions.detach().numpy()

    plt.rcParams.update({'font.size': 19})
    fig,axs = plt.subplots(1, len(levels),figsize=(12,6))
    #colors = {0:'blue', 1:'r', 2:'g', 3:'c', 4:'m', 5:'k', 6:'y'}
    tab20 = plt.get_cmap('tab10')
    colors = tab20.colors
    # loop over the number of variables
    if len(levels) < 2:
        for j in range(len(levels)):
            for i in range(levels[j]+1):
                index = torch.where(perm[:,j] == i) 
                if i<=10:
                    #col = list(map(lambda x: colors[x], np.ones(index[0].numpy().shape) * i))
                    axs.scatter(positions[index][...,0], positions[index][...,1], label = 'level' + str(i+1), color = colors[i%10], marker='X',s=300)#marker=r'$\clubsuit$'
                    axs.set_xlabel(r'$z_1$')
                    axs.set_ylabel(r'$z_2$')
                    axs.legend()
                    tempxi = np.min(positions[...,0])-0.2 * (np.abs(np.min(positions[...,0])) +5)
                    tempxx = np.max(positions[...,0]) + 0.2 * (np.abs(np.max(positions[...,0])) +5)
                    tempyi = np.min(positions[...,1])-0.2 * (np.abs(np.min(positions[...,1])) +5)
                    tempyx = np.max(positions[...,1]) + 0.2 * (np.abs(np.max(positions[...,1])) +5)
                    axs.set_xlim(tempxi, tempxx)
                    axs.set_ylim(tempyi, tempyx)

                else: 
                    axs.scatter(positions[index][...,0], positions[index][...,1], label = 'level' + str(i+1))
                    #axs.set_title('Variable ' + str(j), fontsize = 15)
                    axs.set_xlabel(r'$z_1$', fontsize = 25)
                    axs.set_ylabel(r'$z_2$', fontsize = 25)
                    axs.legend()
                    tempxi = np.min(positions[...,0])-0.2 * (np.abs(np.min(positions[...,0])) +5)
                    tempxx = np.max(positions[...,0]) + 0.2 * (np.abs(np.max(positions[...,0])) +5)
                    tempyi = np.min(positions[...,1])-0.2 * (np.abs(np.min(positions[...,1])) +5)
                    tempyx = np.max(positions[...,1]) + 0.2 * (np.abs(np.max(positions[...,1])) +5)
                    axs.set_xlim(tempxi, tempxx)
                    axs.set_ylim(tempyi, tempyx)
    
    else:
        # loop over the number of variables
        for j in range(len(levels)):
            for i in range(levels[j]+1):
                index = torch.where(perm[:,j] == i) 
                #col = list(map(lambda x: colors[x], np.ones(index[0].numpy().shape) * i))
                axs[j].scatter(positions[index][:,0], positions[index][:,1], label = 'level' + str(i+1), color = colors[i%10], s = (i+1)*50/levels[j])
                axs[j].set_title('Variable ' + str(j), fontsize = 15)
                axs[j].set_xlabel(r'$z_1$', fontsize = 15)
                axs[j].set_ylabel(r'$z_2$', fontsize = 15)
                axs[j].legend()

            fig.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle,fontsize=20)







def plot_ls(model, constraints_flag = True, suptitle= None):
    # 
    # plot latent values

    zeta = torch.tensor(model.zeta, dtype = torch.float64)
    #A = model.nn_model.weight.detach()
    perm = model.perm
    levels = model.num_levels_per_var
    #positions = torch.matmul(zeta, A.T)   # this gives the position of each combination in latent space

    positions = model.nn_model(zeta)
    if positions.ndim > 2:
        positions = positions.mean(axis = 0)
    else:
        positions = positions.detach()

    if positions.ndim > 2:
        positions = positions.mean(axis = 0) 

    # applying the constrains
    if constraints_flag:
        positions = constrains(positions)


    positions = positions.detach().numpy()

    plt.rcParams.update({'font.size': 19})
    fig,axs = plt.subplots(1, len(levels),figsize=(12,6))
    #colors = {0:'blue', 1:'r', 2:'g', 3:'c', 4:'m', 5:'k', 6:'y'}
    tab20 = plt.get_cmap('tab10')
    colors = tab20.colors
    # loop over the number of variables
    if len(levels) < 2:
        for j in range(len(levels)):
            for i in range(levels[j]):
                index = torch.where(perm[:,j] == i) 
                if i<=4:
                    #col = list(map(lambda x: colors[x], np.ones(index[0].numpy().shape) * i))
                    axs.scatter(positions[index][...,0], positions[index][...,1], label = 'level' + str(i+1), color = colors[i%10], marker='X',s=300)#marker=r'$\clubsuit$'
                    axs.set_xlabel(r'$z_1$')
                    axs.set_ylabel(r'$z_2$')
                    axs.legend()
                    tempxi = np.min(positions[...,0])-0.2 * (np.abs(np.min(positions[...,0])) +5)
                    tempxx = np.max(positions[...,0]) + 0.2 * (np.abs(np.max(positions[...,0])) +5)
                    tempyi = np.min(positions[...,1])-0.2 * (np.abs(np.min(positions[...,1])) +5)
                    tempyx = np.max(positions[...,1]) + 0.2 * (np.abs(np.max(positions[...,1])) +5)
                    axs.set_xlim(tempxi, tempxx)
                    axs.set_ylim(tempyi, tempyx)

                else: 
                    axs.scatter(positions[index][...,0], positions[index][...,1], label = 'level' + str(i+1))
                    #axs.set_title('Variable ' + str(j), fontsize = 15)
                    axs.set_xlabel(r'$z_1$', fontsize = 25)
                    axs.set_ylabel(r'$z_2$', fontsize = 25)
                    axs.legend()
                    tempxi = np.min(positions[...,0])-0.2 * (np.abs(np.min(positions[...,0])) +5)
                    tempxx = np.max(positions[...,0]) + 0.2 * (np.abs(np.max(positions[...,0])) +5)
                    tempyi = np.min(positions[...,1])-0.2 * (np.abs(np.min(positions[...,1])) +5)
                    tempyx = np.max(positions[...,1]) + 0.2 * (np.abs(np.max(positions[...,1])) +5)
                    axs.set_xlim(tempxi, tempxx)
                    axs.set_ylim(tempyi, tempyx)
    
    else:
        # loop over the number of variables
        for j in range(len(levels)):
            for i in range(levels[j]):
                index = torch.where(perm[:,j] == i) 
                #col = list(map(lambda x: colors[x], np.ones(index[0].numpy().shape) * i))
                axs[j].scatter(positions[index][:,0], positions[index][:,1], label = 'level' + str(i+1), color = colors[i%10], s = (i+1)*50/levels[j])
                axs[j].set_title('Variable ' + str(j), fontsize = 15)
                axs[j].set_xlabel(r'$z_1$', fontsize = 15)
                axs[j].set_ylabel(r'$z_2$', fontsize = 15)
                axs[j].legend()

            fig.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle,fontsize=20)


def constrains(z):
    n = z.shape[0]
    z = z - z[0,:]

    if z[1,0] < 0:
        z[:, 0] *= -1
    
    rot = torch.atan(-1 * z[1,1]/z[1,0])
    R = torch.tensor([ [torch.cos(rot), -1 * torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]])

    z = torch.matmul(R, z.T)
    z = z.T
    if z.shape[1] > 2 and z[2,1] < 0:
        z[:, 1] *= -1
    
    return z
    


