""" explainers.py

    Define the different explainers: GraphSHAP + benchmarks
"""
import random
import os
import time
import statistics
from copy import copy, deepcopy
from math import sqrt
import warnings
import time
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.special
import seaborn as sns
import tensorboardX
import torch
import torch_geometric
from sklearn.linear_model import (Lasso, LassoLars, LassoLarsCV,
                                  LinearRegression, Ridge)
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx
from tqdm import tqdm

from models import LinearRegressionModel
# GraphLIME
from plots import denoise_graph, k_hop_subgraph, log_graph, visualize_subgraph
from utils.io_utils import gen_explainer_prefix, gen_prefix

warnings.filterwarnings("ignore")


class GraphSHAP():

    def __init__(self, data, model,  adj, writer, args_dataset, gpu=False):
        self.model = model
        self.data = data
        self.adj = adj
        self.writer = writer
        self.args_dataset = args_dataset
        self.gpu = gpu
        self.F = None  # number of non zero node features
        self.neighbours = None  # neighbours considered
        self.M = None  # number of nonzero features - for each node index

        self.model.eval()

    def explain(self,
             node_indexes=[0],
             hops=2,
             num_samples=10,
             info=True,
             multiclass=False,
             args_hv='compute_pred',
             args_feat='Expectation',
             args_coal='Smarter',
             args_g='WLS',
             regu=None,
             vizu=False):
        """ Explain prediction for a particular node - GraphSHAP method

        Args:
            node_index (int, optional): index of the node of interest. Defaults to 0.
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSHAP's new dataset.
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings.
                                                    And include vizualisation. Defaults to True.
            args_hv (str, optional): strategy used to convert simplified input z' to original
                                                    input space z
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z'
            args_g (str, optional): method used to train model g on (z', f(z))
            multiclass (bool, optional): extension - consider predicted class only or all classes
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features

        Returns:
                [type]: shapley values for features/neighbours that influence node v's pred
        """
        # Time
        start = time.time()

        # Accept a subset of nodes for explanations
        phi_list = []
        for node_index in node_indexes:

        # Compute true prediction of model, for original instance
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                with torch.no_grad():
                    true_pred, attention_weights = self.model(self.data.x.cuda(), self.adj.cuda())
                    true_conf, true_pred = true_pred[0, node_index, :].max(dim=0)
            else:
                with torch.no_grad():
                        true_pred, attention_weights = self.model(self.data.x, self.adj)
                        true_conf, true_pred = true_pred[0, node_index, :].max(dim=0)

            # Construct the k-hop subgraph of the node of interest (v)
            self.neighbours, _, _, edge_mask =\
                        torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                            num_hops=hops,
                                                            edge_index=self.data.edge_index)
            # Stores the indexes of the neighbours of v (+ index of v itself)

            # Retrieve 1-hop neighbours of v
            one_hop_neighbours, _, _, _ =\
                        torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                        num_hops=1,
                                        edge_index=self.data.edge_index)

            # Specific case: my new method - rigorous
            if args_hv == 'node_specific':
                discarded_feat_idx = []

                # Consider only relevant entries for v only
                if args_feat == 'Null':
                    feat_idx = self.data.x[node_index, :].nonzero()
                    self.F = feat_idx.size()[0]
                elif args_feat == 'All':
                    self.F = self.data.x[node_index, :].shape[0]
                    feat_idx = torch.unsqueeze(
                        torch.arange(self.data.x.size(0)), 1)
                else:
                    # Stats dataset
                    std = self.data.x.std(axis=0)
                    mean = self.data.x.mean(axis=0)
                    # Feature intermediate rep
                    mean_subgraph = self.data.x[node_index, :]
                    # Select relevant features only - (E-e,E+e)
                    mean_subgraph = torch.where(mean_subgraph > mean - 0.25*std, mean_subgraph,
                                                torch.ones_like(mean_subgraph)*100)
                    mean_subgraph = torch.where(mean_subgraph < mean + 0.25*std, mean_subgraph,
                                                torch.ones_like(mean_subgraph)*100)
                    feat_idx = (mean_subgraph == 100).nonzero()
                    discarded_feat_idx = (mean_subgraph != 100).nonzero()
                    self.F = feat_idx.shape[0]
                    del mean, mean_subgraph, std

                # Remove node v index from neighbours and store their number in D
                self.neighbours = self.neighbours[self.neighbours != node_index]
                D = self.neighbours.shape[0]

                # Total number of features + neighbours considered for node v
                self.M = self.F+D

                # Def range of endcases considered
                args_K = 2

                weights = torch.zeros(num_samples, dtype=torch.float64)
                # Features only
                num = num_samples//2
                z_bis = eval('self.' + args_coal)(num,
                                                  args_K, 1)  # SmarterRegu
                s = (z_bis != 0).sum(dim=1)
                weights[:num] = self.shapley_kernel(s, self.F)
                z_ = torch.zeros(num_samples, self.M)
                z_[:num, :self.F] = z_bis
                # Node only
                z_bis = eval('self.' + args_coal)(
                    num + num_samples % 2, args_K, 0)  # SmarterRegu
                s = (z_bis != 0).sum(dim=1)
                weights[num:] = self.shapley_kernel(s, D)
                z_[num:, :] = torch.ones(num + num_samples % 2, self.M)
                z_[num:, self.F:] = z_bis
                del z_bis, s

            else:
                # Determine z': features and neighbours whose importance is investigated
                discarded_feat_idx = []
                # Consider only non-zero entries in the subgraph of v
                if args_feat == 'Null':
                    feat_idx = self.data.x[self.neighbours, :].mean(
                        axis=0).nonzero()
                    self.F = feat_idx.size()[0]

                # Consider all features (+ use expectation like below)
                elif args_feat == 'All':
                    self.F = self.data.x[node_index, :].shape[0]
                    feat_idx = torch.unsqueeze(
                        torch.arange(self.data.x.size(0)), 1)

                # Consider only features whose aggregated value is different from expected one
                else:
                    # Stats dataset
                    std = self.data.x.std(axis=0)
                    mean = self.data.x.mean(axis=0)
                    # Feature intermediate rep
                    mean_subgraph = self.data.x[self.neighbours, :].mean(
                        axis=0)
                    # Select relevant features only - (E-e,E+e)
                    mean_subgraph = torch.where(mean_subgraph > mean - 0.25*std, mean_subgraph,
                                                torch.ones_like(mean_subgraph)*100)
                    mean_subgraph = torch.where(mean_subgraph < mean + 0.25*std, mean_subgraph,
                                                torch.ones_like(mean_subgraph)*100)
                    feat_idx = (mean_subgraph == 100).nonzero()
                    discarded_feat_idx = (mean_subgraph != 100).nonzero()
                    self.F = feat_idx.shape[0]
                    del mean, mean_subgraph, std

                # Potentially do a feature selection with Lasso (or otherwise)
                # Long process

                # Remove node v index from neighbours and store their number in D
                self.neighbours = self.neighbours[self.neighbours != node_index]
                D = self.neighbours.shape[0]

                # Total number of features + neighbours considered for node v
                self.M = self.F+D

                # Def range of endcases considered
                args_K = 2

                ### COALITIONS: sample z' - binary vector of dimension (num_samples, M)
                z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

                # Compute |z'| for each sample z': number of non-zero entries
                s = (z_ != 0).sum(dim=1)

                ### GRAPHSHAP KERNEL: define weights associated with each sample
                weights = self.shapley_kernel(s, self.M)
                if max(weights) > 9 and info:
                    print('!! Empty or/and full coalition is included !!')


            # H_V: Create dataset (z', f(hv(z'))=(z', f(z)), stored as (z_, fz)
            # Retrive z from z' and x_v, then compute f(z)
            fz = eval('self.' + args_hv)(node_index, num_samples, D, z_,
                                         feat_idx, one_hop_neighbours, args_K, args_feat,
                                         discarded_feat_idx, multiclass, true_pred)

            # g: Weighted Linear Regression to learn shapley values
            phi, base_value = eval('self.' + args_g)(z_, weights, fz, multiclass, info)
            if info:
                print('Base value', base_value, 'for class ', true_pred.item())

            # REGU
            if type(regu) == int and not multiclass:
                expl = (true_conf.cpu() - base_value).detach().numpy()
                phi[:self.F] = (regu * expl / sum(phi[:self.F])) * phi[:self.F]
                phi[self.F:] = ((1-regu) * expl / sum(phi[self.F:])) * phi[self.F:]

            # PRINT some information
            if info:
                self.print_info(D, node_index, phi, feat_idx,
                                true_pred, true_conf, multiclass)

            # VISUALISATION
            if vizu:
                self.vizu(edge_mask, node_index, phi, true_pred, hops, multiclass)


            ### Visualisation
            weighted_edge_mask = self.weighted_edge_mask(edge_mask, node_index, phi, true_pred, hops)
            
            G = self.denoise_graph(weighted_edge_mask, phi[self.F:], 
                        node_index, feat=self.data.x, label=self.data.y, threshold_num=10)

            self.log_graph(
                            self.writer,
                            G,
                            "graph/{}_{}".format(self.args_dataset,
                                                    node_index),
                            identify_self=True,
                        )

            # Time
            # TODO: remove after tests
            end = time.time()
            if info:
                print('Time: ', end - start)

            # Append explanations for this node to list of expl.
            phi_list.append(phi)

        return phi_list

    ################################
    # Coalition sampler
    ################################
    def SmarterSoftRegu(self, num_samples, args_K, regu):
        """ Coalition sampling that favour neighbours or features 
            Soft because unfavoured categories still receive some
            z'j=1 (random, full coalition) 

        """

        # Favour features - special coalitions don't study node's effect
        if regu > 0.5:
            # Define empty and full coalitions
            # self.M = self.F
            z_ = torch.ones(num_samples, self.M)
            z_[1::2] = torch.zeros(num_samples//2, self.M)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * self.F < num_samples and k == 1:
                    z_[i:i+self.F, :] = torch.ones(self.F, self.M)
                    z_[i:i+self.F, :].fill_diagonal_(0)
                    z_[i+self.F:i+2*self.F, :] = torch.zeros(self.F, self.M)
                    z_[i+self.F:i+2*self.F, :].fill_diagonal_(1)
                    i += 2 * self.F
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 9*(num_samples - i)//10
                    while i < samp and k <= min(args_K, self.F):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(self.F), k))
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(
                                    num_samples-i, self.M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(
                                    num_samples-i, self.M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                    return z_
            return z_

        # Favour neighbour
        elif regu < 0.5:
            # Define empty and full coalitions
            D = len(self.neighbours)
            #self.M = D
            #self.F = 0
            z_ = torch.ones(num_samples, self.M)
            z_[1::2] = torch.zeros(num_samples//2, self.M)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * D < num_samples and k == 1:
                    z_[i:i+D, :] = torch.ones(D, self.M)
                    z_[i:i+D, self.F:].fill_diagonal_(0)
                    z_[i+D:i+2*D, :] = torch.zeros(D, self.M)
                    z_[i+D:i+2*D, self.F:].fill_diagonal_(1)
                    i += 2 * D
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 9*(num_samples - i)//10
                    while i < samp and k <= min(args_K, D):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(self.F, self.M), k))
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(
                                    num_samples-i, self.M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(
                                    num_samples-i, self.M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                    return z_
        else:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        return z_

    def SmarterRegu(self, num_samples, args_K, regu):
        """ Coalition sampling that consider exclusively neighbours or features 
        No random coalition at the end

        """
        if regu=='None':
            z_ = self.Smarter(num_samples, args_K, regu) 
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            # Define empty and full coalitions
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M)
            # z_[1, :] = torch.empty(1, self.M).random_(2)
            i = 2 
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * self.F < num_samples and k == 1:
                    z_[i:i+self.F, :] = torch.ones(self.F, M)
                    z_[i:i+self.F, :].fill_diagonal_(0)
                    z_[i+self.F:i+2*self.F, :] = torch.zeros(self.F, M)
                    z_[i+self.F:i+2*self.F, :].fill_diagonal_(1)
                    i += 2 * self.F
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = num_samples
                    while i<samp and k<=min(args_K, self.F):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order. 
                        L = list( combinations(range(self.F),k) )
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                #z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                #z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions 
                    z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                    return z_
            return z_
        
        # Favour neighbour
        else: 
            # Define empty and full coalitions
            D = len(self.neighbours)
            M = D
            # self.F = 0 
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M)
            i = 2 
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * D < num_samples and k == 1:
                    z_[i:i+D, :] = torch.ones(D, M)
                    z_[i:i+D, :].fill_diagonal_(0)
                    z_[i+D:i+2*D, :] = torch.zeros(D, M)
                    z_[i+D:i+2*D, :].fill_diagonal_(1)
                    i += 2 * D
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = num_samples
                    while i<samp and k<=min(args_K, D):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order. 
                        L = list( combinations(range(0, M), k) )
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                #z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                #z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions 
                    z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                    return z_
            return z_

    def SmarterPlus(self, num_samples, args_K, *unused):
        """ Sample coalitions cleverly given shapley kernel def
        Consider nodes and features separately to better capture their effect

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        # No coalitions
        i = 0
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9*(num_samples - i)//10
                while i < samp and k <= min(args_K, self.F, self.M-self.F):
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + list(combinations(range(self.F, self.M), k) )
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)

        return z_
        
    def Smarter(self, num_samples, args_K, *unused):
        """ Sample coalitions cleverly given shapley kernel def
        Consider nodes and features separately to better capture their effect

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        # z_[1, :] = torch.empty(1, self.M).random_(2)
        i = 2
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9*(num_samples - i)//10
                while i < samp and k <= min(args_K, self.F, self.M-self.F):
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + list(combinations(range(self.F, self.M), k) )
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_
        return z_

    def Smart(self, num_samples, *unused):
        """ Sample coalitions cleverly given shapley kernel def

        Args:
            num_samples ([int]): total number of coalitions z_

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        k = 1
        i = 2
        while i < num_samples:
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1
            elif k == 1:
                M = list(range(self.M))
                random.shuffle(M)
                for j in range(self.M):
                    z_[i, M[j]] = torch.zeros(1)
                    i += 1
                    if i == num_samples:
                        return z_
                    z_[i, M[j]] = torch.ones(1)
                    i += 1
                    if i == num_samples:
                        return z_
                k += 1
            elif k == 2:
                M = list(combinations(range(self.M), 2))[:num_samples-i+1]
                random.shuffle(M)
                for j in range(len(M)):
                    z_[i, M[j][0]] = torch.tensor(0)
                    z_[i, M[j][1]] = torch.tensor(0)
                    i += 1
                    if i == num_samples:
                        return z_
                    z_[i, M[j][0]] = torch.tensor(1)
                    z_[i, M[j][1]] = torch.tensor(1)
                    i += 1
                    if i == num_samples:
                        return z_
                k += 1
            else:
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_

        return z_

    def Random(self, num_samples, *unused):
        z_ = torch.empty(num_samples, self.M).random_(2)
        # z_[0, :] = torch.ones(self.M)
        # z_[1, :] = torch.zeros(self.M)
        return z_

    ################################
    # GraphSHAP kernel
    ################################
    def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample 

        Args:
            s (tensor): contains dimension of z' for all instances
                (number of features + neighbours included)

        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == M:
                shap_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                shap_kernel.append(1/M)
            else:
                shap_kernel.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))
        return torch.tensor(shap_kernel)

    ################################
    # COMPUTE PREDICTIONS f(z)
    ################################
    def custom_to_networkx(self, data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
        r"""Converts a :class:`torch_geometric.data.Data` instance to a
        :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
        an undirected :obj:`networkx.Graph` otherwise.

        Args:
            data (torch_geometric.data.Data): The data object.
            node_attrs (iterable of str, optional): The node attributes to be
                copied. (default: :obj:`None`)
            edge_attrs (iterable of str, optional): The edge attributes to be
                copied. (default: :obj:`None`)
            to_undirected (bool, optional): If set to :obj:`True`, will return a
                a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
                undirected graph will correspond to the upper triangle of the
                corresponding adjacency matrix. (default: :obj:`False`)
            remove_self_loops (bool, optional): If set to :obj:`True`, will not
                include self loops in the resulting graph. (default: :obj:`False`)
        """

        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        G.add_nodes_from(range(data.num_nodes))

        values = {}
        for key, item in data.__dict__.items():
            if torch.is_tensor(item):
                values[key] = item.squeeze().tolist()
            else:
                values[key] = item
            if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
                values[key] = item[0]

        for i, (u, v) in enumerate(data.edge_index.t().tolist()):

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)
            for key in edge_attrs if edge_attrs is not None else []:
                G[u][v][key] = values[key][i]

        for key in node_attrs if node_attrs is not None else []:
            for i, feat_dict in G.nodes(data=True):
                feat_dict.update({key: values[key][i]})

        return G

    def compute_pred(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        # To networkx
        G = self.custom_to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)
            # 'All' and 'Expectation'

        # Store nodes and features not sampled
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            # Special case - consider only feat. influence if too few nei included
            if D - len(ex_nei) >= min(self.F - len(ex_feat), args_K):
                A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
            # May delete - should be an approximation
            #if args_feat == 'Expectation':
            #    for val in discarded_feat_idx:
            #        X[self.neighbours, val] = av_feat_values[val].repeat(D)


            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):

                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())

                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values

            # Usual case - exclude features for the whole subgraph
            else:
                for val in ex_feat:
                    X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

            # new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
            # for i in range(A.shape[1]):
            #     new_adj[A[0, i], A[1, i]] = 1.0
            # new_adj = new_adj.unsqueeze(0)
            # del A

            # Transform new data (X, A) to original input form
            A = torch_geometric.utils.to_dense_adj(A)
            
            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    pred, attention_weights = self.model(X.cuda(), A.cuda())
                    proba = pred[0, node_index, :]
            else:
                with torch.no_grad():
                        pred, attention_weights = self.model(X, A)
                        proba = pred[0, node_index, :]
            # Softmax ? No exp(), log()

            # Store predicted class label in fz
            if multiclass:
                #fz[key] = proba
                fct = torch.nn.Softmax(dim=0)
                fz[key]= fct(proba)
                
            else:
                #fz[key] = proba[true_pred]
                fct = torch.nn.Softmax(dim=0)
                fz[key]= fct(proba)[true_pred]
                
        return fz

    def basic_default_2hop(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        G = self.custom_to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # or random feature vector made of random value across each col of X
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            # NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
            X = deepcopy(self.data.x)

            # Set discarded features to an average value when few neighbours
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            for val in ex_feat:
                X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0


            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):
                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                #if len(self.neighbours) - len(ex_nei) < args_K:
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values
            
            # Transform new data (X, A) to original input form
            A = torch_geometric.utils.to_dense_adj(A)

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    pred, attention_weights = self.model(X.cuda(), A.cuda())
                    proba = pred[0, node_index, :]
            else:
                with torch.no_grad():
                    pred, attention_weights = self.model(X, A)
                    proba = pred[0, node_index, :]
            # Softmax ? No exp(), log()

            # Store predicted class label in fz
            if multiclass:
                #fz[key] = proba
                fct = torch.nn.Softmax(dim=0)
                fz[key] = fct(proba)

            else:
                #fz[key] = proba[true_pred]
                fct = torch.nn.Softmax(dim=0)
                fz[key] = fct(proba)[true_pred]

        return fz

    def basic_default(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # or random feature vector made of random value across each col of X
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest and the whole subgraph
            # NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            for val in ex_feat:
                X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

            # Transform new data (X, A) to original input form
            A = torch_geometric.utils.to_dense_adj(A)
            
            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    pred, attention_weights = self.model(X.cuda(), A.cuda())
                    proba = pred[0, node_index, :]
            else:
                with torch.no_grad():
                        pred, attention_weights = self.model(X, A)
                        proba = pred[0, node_index, :]
            
            # Store predicted class label in fz
            if multiclass:
                #fz[key] = proba
                fct = torch.nn.Softmax(dim=0)
                fz[key]= fct(proba)
                
            else:
                #fz[key] = proba[true_pred]
                fct = torch.nn.Softmax(dim=0)
                fz[key]= fct(proba)[true_pred]

        return fz

    def node_specific(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        G = self.custom_to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)
        # or random feature vector made of random value across each col of X

        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)    
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and D - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(D - len(ex_nei), args_K):
                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                #if len(self.neighbours) - len(ex_nei) < args_K:
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values

            # Transform new data (X, A) to original input form
            A = torch_geometric.utils.to_dense_adj(A)

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    pred, attention_weights = self.model(X.cuda(), A.cuda())
                    proba = pred[0, node_index, :]
            else:
                with torch.no_grad():
                    pred, attention_weights = self.model(X, A)
                    proba = pred[0, node_index, :]

            # Store predicted class label in fz
            if multiclass:
                #fz[key] = proba
                fct = torch.nn.Softmax(dim=0)
                fz[key] = fct(proba)

            else:
                #fz[key] = proba[true_pred]
                fct = torch.nn.Softmax(dim=0)
                fz[key] = fct(proba)[true_pred]


        return fz

    def neutral(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # or random feature vector made of random value across each col of X

        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)


        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)

            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            A = self.data.edge_index
            X[ex_nei, :] = av_feat_values.repeat(len(ex_nei), 1)
            # Only for node index
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Transform new data (X, A) to original input form
            A = torch_geometric.utils.to_dense_adj(A)

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    pred, attention_weights = self.model(X.cuda(), A.cuda())
                    proba = pred[0, node_index, :]
            else:
                with torch.no_grad():
                    pred, attention_weights = self.model(X, A)
                    proba = pred[0, node_index, :]

            # Store predicted class label in fz
            if multiclass:
                #fz[key] = proba
                fct = torch.nn.Softmax(dim=0)
                fz[key] = fct(proba)

            else:
                #fz[key] = proba[true_pred]
                fct = torch.nn.Softmax(dim=0)
                fz[key] = fct(proba)[true_pred]


        return fz

    ################################
    def compute_pred_regu(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        # To networkx
        G = torch_geometric.utils.to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        ### Look only at nodes
        if self.M == self.F:
            excluded_feat = {}

            for i in range(num_samples):
                feats_id = []
                for j in range(self.F):
                    if z_[i, j].item() == 0:
                        feats_id.append(feat_idx[j].item())
                excluded_feat[i] = feats_id

            for key, ex_feat in tqdm(excluded_feat.items()):
                # Change feature vector for node of interest - excluded and discarded features
                X = deepcopy(self.data.x)
                X[node_index, ex_feat] = av_feat_values[ex_feat]
                for val in ex_feat:
                    X[self.neighbours, val] = av_feat_values[val].repeat(D)

                # Apply model on (X,A) as input.
                if self.gpu:
                    with torch.no_grad():
                        proba = self.model(x=X.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index]
                else:
                    with torch.no_grad():
                        proba = self.model(x=X, edge_index=self.data.edge_index).exp()[
                            node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        ### Look only at neighbours
        elif self.M == len(self.neighbours):
            excluded_nei = {}

            for i in range(num_samples):
                nodes_id = []
                for j in range(D):
                    if z_[i, j] == 0:
                        nodes_id.append(self.neighbours[j].item())
                # Dico with key = num_sample id, value = excluded neighbour index
                excluded_nei[i] = nodes_id

            for key, ex_nei in tqdm(excluded_nei.items()):
                positions = []
                for val in ex_nei:
                    pos = (self.data.edge_index == val).nonzero()[
                        :, 1].tolist()
                    positions += pos
                # Create new adjacency matrix for that sample
                positions = list(set(positions))
                A = np.array(self.data.edge_index)
                A = np.delete(A, positions, axis=1)
                A = torch.tensor(A)
                X = deepcopy(self.data.x)

                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                #if len(self.neighbours) - len(ex_nei) < args_K:
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values
                # Transform new data (X, A) to original input form
                A = torch_geometric.utils.to_dense_adj(A)

                # Apply model on (X,A) as input.
                if self.gpu:
                    with torch.no_grad():
                        pred, attention_weights = self.model(X.cuda(), A.cuda())
                        proba = pred[0, node_index, :]
                else:
                    with torch.no_grad():
                        pred, attention_weights = self.model(X, A)
                        proba = pred[0, node_index, :]

                # Store predicted class label in fz
                if multiclass:
                    #fz[key] = proba
                    fct = torch.nn.Softmax(dim=0)
                    fz[key] = fct(proba)

                else:
                    #fz[key] = proba[true_pred]
                    fct = torch.nn.Softmax(dim=0)
                    fz[key] = fct(proba)[true_pred]
                    

        else:
            fz = self.compute_pred(node_index, num_samples, D, z_, feat_idx,
                                   one_hop_neighbours, args_K, args_feat, discarded_feat_idx)

        return fz

    ################################
    # LEARN MODEL G
    ################################
    def WLR(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        """
        # Define model
        if multiclass:
            our_model = LinearRegressionModel(z_.shape[1], self.data.num_classes)
        else:
            our_model = LinearRegressionModel(z_.shape[1], 1)
        our_model.train()

        # Define optimizer and loss function
        def weighted_mse_loss(input, target, weight):
            return (weight * (input - target) ** 2).mean()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(our_model.parameters(), lr=0.2)
        #optimizer = torch.optim.Adam(our_model.parameters(), lr=0.2)

        # Dataloader
        train = torch.utils.data.TensorDataset(z_, fz)
        train_loader = torch.utils.data.DataLoader(train, batch_size=10)

        # Repeat for several epochs
        for epoch in range(100):

            av_loss = []
            # for x,y,w in zip(z_,fz, weights):
            for batch_idx, (dat, target) in enumerate(train_loader):
                x, y = Variable(dat), Variable(target)

                # Forward pass: Compute predicted y by passing x to the model
                pred_y = our_model(x)

                # Compute loss
                loss = weighted_mse_loss(pred_y, y, weights[batch_idx])
                # loss = criterion(pred_y,y)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store batch loss
                av_loss.append(loss.item())
            if info:
                print('av loss epoch: ', np.mean(av_loss))

        # Evaluate model
        our_model.eval()
        with torch.no_grad():
            pred = our_model(z_)
        if info:
            print('weighted r2 score: ', r2_score(
                pred, fz, multioutput='variance_weighted'))
            if multiclass:
                print(r2_score(pred, fz, multioutput='raw_values'))
            print('r2 score: ', r2_score(pred, fz, weights))

        phi, base_value = [param.T for _, param in our_model.named_parameters()]
        phi = np.squeeze(phi, axis=1)
        return phi.detach().numpy().astype('float64'), base_value

    def WLR_sklearn(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()
        # Fit weighted linear regression
        reg = LinearRegression()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)
        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))
        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

    def WLS(self, z_, weights, fz, multiclass, info):
        """ Ordinary Least Squares Method, weighted
            Estimates shapely value coefficients

        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z'
            fz ([type]): prediction f(z) where z is a new instance - formed from z' and x

        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z', f(z))
            Dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            if info:
                print('WLS: Matrix not invertible')
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(10**(-5) * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
                    np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

        # Test accuracy
        y_pred = z_.detach().numpy() @ phi
        if info:
            print('r2: ', r2_score(fz, y_pred))
            print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]

        ################################

    ################################
    # INFO ON EXPLANATIONS
    ################################
    def print_info(self, D, node_index, phi, feat_idx, true_pred, true_conf, multiclass):
        """
        Displays some information about explanations - for a better comprehension and audit
        """

        # Print some information
        print('Explanations include {} node features and {} neighbours for this node\
        for {} classes'.format(self.F, D, self.data.num_classes))

        # Compare with true prediction of the model - see what class should truly be explained
        print('Prediction of orignal model is class {} with confidence {}, while label is {}'
                    .format(true_pred, true_conf, self.data.y[node_index]))

        # Isolate explanations for predicted class - explain model choices
        if multiclass:
            pred_explanation = phi[true_pred, :]
        else:
            pred_explanation = phi

        # print('Explanation for the class predicted by the model:', pred_explanation)

        # Look at repartition of weights among neighbours and node features
        # Motivation for regularisation
        print('Weights for node features: ', sum(pred_explanation[:self.F]),
                    'and neighbours: ', sum(pred_explanation[self.F:]))
        print('Total Weights (abs val) for node features: ', sum(np.abs(pred_explanation[:self.F])),
                    'and neighbours: ', sum(np.abs(pred_explanation[self.F:])))

        # Note we focus on explanation for class predicted by the model here, so there is a bias towards
        # positive weights in our explanations (proba is close to 1 everytime).
        # Alternative is to view a class at random or the second best class

        # Select most influential neighbours and/or features (+ or -)
        if self.F + D > 10:
            _, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation)), 6)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                if idx.item() < self.F:
                    influential_feat[feat_idx[idx]] = val
                else:
                    influential_nei[self.neighbours[idx-self.F]] = val
            print('Most influential features: ', len([(item[0].item(), item[1].item()) for item in list(influential_feat.items())]),
                            'and neighbours', len([(item[0].item(), item[1].item()) for item in list(influential_nei.items())]))

        # Most influential features splitted bewteen neighbours and features
        if self.F > 5:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[:self.F])), 3)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            for idx, val in zip(idxs, vals):
                influential_feat[feat_idx[idx]] = val
            print('Most influential features: ', [
                            (item[0].item(), item[1].item()) for item in list(influential_feat.items())])

        # Most influential features splitted bewteen neighbours and features
        if D > 5 and self.M!=self.F:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[self.F:])), 3)
            vals = [pred_explanation[self.F + idx] for idx in idxs]
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                influential_nei[self.neighbours[idx]] = val
            print('Most influential neighbours: ', [
                            (item[0].item(), item[1].item()) for item in list(influential_nei.items())])

    def vizu(self, edge_mask, node_index, phi, predicted_class, hops, multiclass):
        """ Vizu of important nodes in subgraph around node_index

        Args:
            edge_mask ([type]): vector of size data.edge_index with False 
                                            if edge is not included in subgraph around node_index
            node_index ([type]): node of interest index
            phi ([type]): explanations for node of interest
            predicted_class ([type]): class predicted by model for node of interest 
            hops ([type]):  number of hops considered for subgraph around node of interest 
            multiclass: if we look at explanations for all classes or only for the predicted one
        """
        if multiclass:
            phi = torch.tensor(phi[predicted_class, :])
        else:
            phi = torch.from_numpy(phi).float()

        # Replace False by 0, True by 1 in edge_mask
        mask = torch.zeros(self.data.edge_index.shape[1])
        for i, val in enumerate(edge_mask):
            if val.item() == True:
                mask[i] = 1

        # Identify one-hop neighbour
        one_hop_nei, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                    node_index, 1, self.data.edge_index, relabel_nodes=True,
                    num_nodes=None)

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                # Remove importance of 1-hop neighbours to 2-hop nei.
                if nei in one_hop_nei:
                    if self.data.edge_index[1, idx] in one_hop_nei:
                        mask[idx] = phi[self.F + i]
                    else:
                        pass
                elif mask[idx] == 1:
                    mask[idx] = phi[self.F + i]
            # mask[mask.nonzero()[i].item()]=phi[i, predicted_class]

        # Set to 0 importance of edges related to 0
        mask[mask == 1] = 0

        # Increase coef for visibility and consider absolute contribution
        mask = torch.abs(mask)

        # Vizu nodes
        ax, G = visualize_subgraph(self.model,
                             node_index,
                             self.data.edge_index,
                             mask,
                             hops,
                             y=self.data.y,
                             threshold=None)

        plt.savefig('log/graph/GS_{}_{}'.format(node_index,
                                            self.model.__class__.__name__
                                            ),
                    bbox_inches='tight')

        # Other visualisation
        G = denoise_graph(self.data, mask, phi[self.F:], self.neighbours,
                          node_index, feat=None, label=self.data.y, threshold_num=10)

        log_graph(G,
                    identify_self=True,
                    nodecolor="label",
                    epoch=0,
                    fig_size=(4, 3),
                    dpi=300,
                    label_node_feat=False,
                    edge_vmax=None,
                    args=None)

        plt.savefig('log/graph/graphshap_{}_{}'.format(node_index,
                                           self.model.__class__.__name__
                                           ),
                                                  bbox_inches='tight')

        # plt.show()

    def weighted_edge_mask(self, edge_mask, node_index, phi, predicted_class, hops):
        """
        :param edge_mask: vector of size data.edge_index with False if edge is not included in subgraph around node_index
        :param node_index: node of interest
        :param phi: explanations for node of interest
        :param predicted class: class predicted by model for node of interest 
        :param hops: number of hops considered for subgraph around node of interest 
        Vizu of important nodes in subgraph of node_index
        """

        # Replace False by 0, True by 1 in edge_mask
        mask = torch.zeros(self.data.edge_index.shape[1])
        for i, val in enumerate(edge_mask):
            if val.item() == True:
                mask[i] = 1

        # Identify one-hop neighbour
        one_hop_nei, _, _, _ = k_hop_subgraph(
                    node_index, 1, self.data.edge_index, relabel_nodes=True,
                    num_nodes=None, flow=self.__flow__(self.model))

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                # Remove importance of 1-hop neighbours to 2-hop nei.
                if nei in one_hop_nei:
                    if self.data.edge_index[1, idx] in one_hop_nei:
                        mask[idx] = torch.tensor(phi[self.F + i]).float()
                    else:
                        pass
                elif mask[idx] == 1:
                    mask[idx] = torch.tensor(phi[self.F + i]).float()
            #mask[mask.nonzero()[i].item()]=phi[i, predicted_class]

        # Set to 0 importance of edges related to 0
        mask[mask == 1] = 0

        # Consider absolute contribution
        # Could also increase visibility (but need to also update phi accordingly)
        mask = torch.abs(mask)

        return mask

    def denoise_graph(self, weighted_edge_mask, node_explanations, node_idx, feat=None, label=None, threshold_num=10):
        """Cleaning a graph by thresholding its node values.

        Args:
            - weighted_edge_mask:  Edge mask, with importance given to each edge
            - node_explanations :  Shapley values for neighbours
            - node_idx          :  Index of node to highlight (TODO ?)
            - feat              :  An array of node features.
            - label             :  A list of node labels.
            - theshold_num      :  The maximum number of nodes to threshold.
        """
        # Disregard size of explanations
        node_explanations = np.abs(node_explanations)

        # Create graph of neighbourhood of node of interest
        G = nx.Graph()
        G.add_nodes_from(self.neighbours.detach().numpy())
        G.add_node(node_idx)
        G.nodes[node_idx]["self"] = 1
        if feat is not None:
            for node in G.nodes():
                G.nodes[node]["feat"] = feat[node].detach().numpy()
        if label is not None:
            for node in G.nodes():
                G.nodes[node]["label"] = label[node].item()

        # Find importance threshold required to retrieve 10 most import nei.
        threshold_num = min(len(self.neighbours), threshold_num)
        threshold = np.sort(
            node_explanations)[-threshold_num]

        # Keep edges that satisfy the threshold
        weighted_edge_list = [
            (self.data.edge_index[0, i].item(),
                            self.data.edge_index[1, i].item(), weighted_edge_mask[i].item())
            for i, _ in enumerate(weighted_edge_mask)
            if weighted_edge_mask[i] >= threshold
        ]
        G.add_weighted_edges_from(weighted_edge_list)

        #Keep nodes that satisfy the threshold
        del_nodes = []
        for i, node in enumerate(G.nodes()):
            if node != node_idx:
                if node_explanations[i] < threshold:
                    del_nodes.append(node)
        G.remove_nodes_from(del_nodes)

        # Remove disconnected components without node_idx
        if not nx.is_connected(G):
            for comp in nx.connected_components(G):
                if node in comp:
                    G = G.subgraph(list(comp))
        G = nx.Graph(G)  # unfreeze

        # Remove isolated nodes - except if this yields the empty graph
        if list(G.nodes()) != list(nx.isolates(G)):
            G.remove_nodes_from(list(nx.isolates(G)))

        return G

    def log_graph(self,
               writer,
               Gc,
               name,
               identify_self=True,
               nodecolor="label",
               epoch=0,
               fig_size=(4, 3),
               dpi=300,
               label_node_feat=False,
               edge_vmax=None,
               args=None):
        """
        Args:
            nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
                be one-hot'
        """
        cmap = plt.get_cmap("Set1")
        plt.switch_backend("agg")
        fig = plt.figure(figsize=fig_size, dpi=dpi)

        node_colors = []
        # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
        edge_colors = [w for (u, v, w) in Gc.edges.data("weight", default=1)]

        # maximum value for node color
        vmax = 8
        for i in Gc.nodes():
            if nodecolor == "feat" and "feat" in Gc.nodes[i]:
                num_classes = Gc.nodes[i]["feat"].size()[0]
                if num_classes >= 10:
                    cmap = plt.get_cmap("tab20")
                    vmax = 19
                elif num_classes >= 8:
                    cmap = plt.get_cmap("tab10")
                    vmax = 9
                break

        feat_labels = {}
        for i in Gc.nodes():
            if identify_self and "self" in Gc.nodes[i]:
                node_colors.append(0)
            elif nodecolor == "label" and "label" in Gc.nodes[i]:
                node_colors.append(Gc.nodes[i]["label"] + 1)
            elif nodecolor == "feat" and "feat" in Gc.nodes[i]:
                # print(Gc.nodes[i]['feat'])
                feat = Gc.nodes[i]["feat"].detach().numpy()
                # idx with pos val in 1D array
                feat_class = 0
                for j in range(len(feat)):
                    if feat[j] == 1:
                        feat_class = j
                        break
                node_colors.append(feat_class)
                feat_labels[i] = feat_class
            else:
                node_colors.append(1)
        if not label_node_feat:
            feat_labels = None

        plt.switch_backend("agg")
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        pos_layout = nx.kamada_kawai_layout(Gc, weight=None)

        if Gc.number_of_nodes() == 0 or Gc.number_of_edges() == 0:
            edge_vmax = 1
            edge_vmin = 0
        else:
            weights = [d for (u, v, d) in Gc.edges(data="weight", default=1)]
            if edge_vmax is None:
                edge_vmax = statistics.median_high(
                    [d for (u, v, d) in Gc.edges(data="weight", default=1)]
                )
            min_color = min([d for (u, v, d) in Gc.edges(data="weight", default=1)])
            # color range: gray to black
            edge_vmin = 2 * min_color - edge_vmax

        nx.draw(
            Gc,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            labels=feat_labels,
            node_color=node_colors,
            vmin=0,
            vmax=vmax,
            cmap=cmap,
            edge_color=edge_colors,
            edge_cmap=plt.get_cmap("Greys"),
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            width=1.0,
            node_size=50,
            alpha=0.8,
        )
        fig.axes[0].xaxis.set_visible(False)
        fig.canvas.draw()

        if args is None:
            save_path = os.path.join("log/", name + ".pdf")
        else:
            save_path = os.path.join(
                "log", name + gen_explainer_prefix(args) + "_" + str(epoch) + ".pdf"
            )
            print("log/" + name + gen_explainer_prefix(args) + "_" + str(epoch) + ".pdf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf")

        img = tensorboardX.utils.figure_to_image(fig)
        writer.add_image(name, img, epoch)

    def visualize_subgraph(self, model, node_idx, edge_index, edge_mask, num_hops, y=None,
                        threshold=None, **kwargs):
        """Visualizes the subgraph around :attr:`node_idx` given an edge mask
            :attr:`edge_mask`.

            Args:
                node_idx (int): The node id to explain.
                edge_index (LongTensor): The edge indices.
                edge_mask (Tensor): The edge mask.
                y (Tensor, optional): The ground-truth node-prediction labels used
                    as node colorings. (default: :obj:`None`)
                threshold (float, optional): Sets a threshold for visualizing
                    important edges. If set to :obj:`None`, will visualize all
                    edges with transparancy indicating the importance of edges.
                    (default: :obj:`None`)
                **kwargs (optional): Additional arguments passed to
                    :func:`nx.draw`.

            :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
            """

        assert edge_mask.size(0) == edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                    node_idx, num_hops, edge_index, relabel_nodes=True,
                    num_nodes=None, flow=self.__flow__(model))

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                    device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def __flow__(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'
