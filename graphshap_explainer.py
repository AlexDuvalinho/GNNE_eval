""" explainers.py

	Define the different explainers: GraphSHAP + benchmarks
"""
from sklearn.linear_model import LinearRegression
from models import LinearRegressionModel

from sklearn.metrics import r2_score
from copy import deepcopy
import warnings
import time
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import networkx as nx
import torch
import torch_geometric
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LassoLarsCV, LassoLars, Lasso, Ridge
from itertools import combinations
# GraphLIME
from plots import visualize_subgraph, k_hop_subgraph, denoise_graph, log_graph

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

			# Determine z': features and neighbours whose importance is investigated
			discarded_feat_idx = []
			# Consider only non-zero entries in the subgraph of v
			if args_feat == 'Null':
					feat_idx = self.data.x[self.neighbours, :].mean(axis=0).nonzero()
					self.F = feat_idx.size()[0]

			# Consider all features (+ use expectation like below)
			elif args_feat == 'All':
				self.F = self.data.x[node_index, :].shape[0]
				feat_idx = torch.unsqueeze(torch.arange(self.data.x.size(0)), 1)

			# Consider only features whose aggregated value is different from expected one
			else:
				# Stats dataset
				std = self.data.x.std(axis=0)
				mean = self.data.x.mean(axis=0)
				# Feature intermediate rep
				mean_subgraph = self.data.x[self.neighbours, :].mean(axis=0)
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
			args_K = 5

			# COALITIONS: sample z' - binary vector of dimension (num_samples, M)
			z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

			# Compute |z'| for each sample z': number of non-zero entries
			s = (z_ != 0).sum(dim=1)

			# Compute true prediction of model, for original instance
			#true_conf, true_pred = self.model(
					#	x=self.data.x.cuda(),
					#	edge_index=self.adj.cuda()).exp()[node_index].max(dim=0)
			if self.gpu:
				with torch.no_grad():
					true_pred, attention_weights = self.model(self.data.x.cuda(), self.adj.cuda())
					true_conf, true_pred = true_pred[0, node_index, :].max(dim=0)
			else:
				with torch.no_grad():
						true_pred, attention_weights = self.model(self.data.x, self.adj)
						true_conf, true_pred = true_pred[0, node_index, :].max(dim=0)

			# GRAPHSHAP KERNEL: define weights associated with each sample
			weights = self.shapley_kernel(s)
			# TODO: remove when tests are finished
			if max(weights) > 9 and info:
				print('!! Empty or/and full coalition is included !!')

			# H_V: Create dataset (z', f(hv(z'))=(z', f(z)), stored as (z_, fz)
			# Retrive z from z' and x_v, then compute f(z)
			fz = eval('self.' + args_hv)(node_index, num_samples, D, z_,
								feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx)

			# MULTICLASS / Predicted class only
			if not multiclass:
				fz = fz[:, true_pred]

			# g: Weighted Linear Regression to learn shapley values
			phi, base_value = eval('self.' + args_g)(z_, weights, fz, multiclass, info)
			if info:
				print('Base value', base_value, 'for class ', true_pred.item())

			# REGU
			if type(regu) == int and not multiclass:
				expl = np.array(true_conf - base_value)
				phi[:self.F] = (regu * expl / sum(phi[:self.F])) * phi[:self.F]
				phi[self.F:] = ((1-regu) * expl / sum(phi[self.F:])) * phi[self.F:]

			# PRINT some information
			if info:
				self.print_info(D, node_index, phi, feat_idx,
								true_pred, true_conf, multiclass)

			# VISUALISATION
			if vizu:
				self.vizu(edge_mask, node_index, phi, true_pred, hops, multiclass)

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
	def SmarterRegu(self, num_samples, args_K, regu):
		""" Coalition sampling that favour neighbours or features 

		"""
		if not regu:
			z_ = self.Smarter(num_samples, args_K, regu)
			return z_

		# Favour features - special coalitions don't study node's effect
		elif regu > 0.5:
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
					samp = i + 2*(num_samples - i)//3
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

		# Favour neighbour
		else:
			# Define empty and full coalitions
			z_ = torch.ones(num_samples, self.M)
			z_[1::2] = torch.zeros(num_samples//2, self.M)
			i = 2
			k = 1
			D = len(self.neighbours)
			# Loop until all samples are created
			while i < num_samples:
				# Look at each feat/nei individually if have enough sample
				# Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
				if i + 2 * D < num_samples and k == 1:
					z_[i:i+D, :] = torch.ones(D, self.M)
					z_[i:i+D, :].fill_diagonal_(0)
					z_[i+D:i+2*D, :] = torch.zeros(D, self.M)
					z_[i+D:i+2*D, :].fill_diagonal_(1)
					i += 2 * D
					k += 1

				else:
					# Split in two number of remaining samples
					# Half for specific coalitions with low k and rest random samples
					samp = i + 2*(num_samples - i)//3
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
				samp = i + 2*(num_samples - i)//3
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
				samp = i + 2*(num_samples - i)//3
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
	def shapley_kernel(self, s):
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
			if a == 0 or a == self.M:
				shap_kernel.append(1000)
			elif scipy.special.binom(self.M, a) == float('+inf'):
				shap_kernel.append(1)
			else:
				shap_kernel.append(
					(self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
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

	def compute_pred(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx):
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
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

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
			if self.M - self.F - len(ex_nei) >= min(self.F - len(ex_feat), args_K):
				A = np.delete(A, positions, axis=1)
			A = torch.tensor(A)

			# Change feature vector for node of interest
			X = deepcopy(self.data.x)
			X[node_index, ex_feat] = av_feat_values[ex_feat]
			if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
				X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

			# Special case - consider only nei. influence if too few feat included
			if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):

				# Look at the 2-hop neighbours included
				# Make sure that they are connected to v (with current nodes sampled nodes)
				included_nei = set(self.neighbours.detach().numpy()).difference(ex_nei)
				included_nei = included_nei.difference(one_hop_neighbours.detach().numpy())

				for incl_nei in included_nei:
					l = nx.shortest_path(G, source=node_index, target=incl_nei)
					for n in range(1, len(l)-1):
						A = torch.cat((A, torch.tensor(
							[[l[n-1]], [l[n]]])), dim=-1)
						X[l[n], :] = av_feat_values

			# Usual case - exclude features for the whole subgraph
			else:
				for val in ex_feat:
					X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

			# Transform new data (X, A) to original input form
			new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
			for i in range(A.shape[1]):
				new_adj[A[0, i], A[1, i]] = 1.0
			new_adj = new_adj.unsqueeze(0)
			del A

			# Apply model on (X,A) as input.
			if self.gpu:
				with torch.no_grad():
					true_pred, attention_weights = self.model(X.cuda(), new_adj.cuda())
					proba = true_pred[0, node_index, :]
			else:
				with torch.no_grad():
						true_pred, attention_weights = self.model(self.data.x, self.adj)
						proba = true_pred[0, node_index, :]
			# Softmax ? No exp(), log()

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz

	def basic_default_2hop(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx):
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
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

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
			if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
				X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
			X[node_index, ex_feat] = av_feat_values[ex_feat]
			for val in ex_feat:
				X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

			# Special case - consider only nei. influence if too few feat included
			if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):
				# Look at the 2-hop neighbours included
				included_nei = set(self.neighbours.detach().numpy()).difference(ex_nei)
				included_nei = included_nei.difference(one_hop_neighbours.detach().numpy())
				for incl_nei in included_nei:
					l = nx.shortest_path(G, source=node_index, target=incl_nei)
					for n in range(1, len(l)-1):
						A = torch.cat((A, torch.tensor(
							[[l[n-1]], [l[n]]])), dim=-1)
			
			# Transform new data (X, A) to original input form
			new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
			for i in range(A.shape[1]):
				new_adj[A[0, i], A[1, i]] = 1.0
			new_adj = new_adj.unsqueeze(0)

			if self.gpu:
				with torch.no_grad():
					true_pred, attention_weights = self.model(X.cuda(), new_adj.cuda())
					proba = true_pred[0, node_index, :]
			else:
				with torch.no_grad():
					true_pred, attention_weights = self.model(self.data.x, self.adj)
					proba = true_pred[0, node_index, :]

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz

	def basic_default(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx):
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
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

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
			if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
				X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

			for val in ex_feat:
				X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

			# Transform new data (X, A) to original input form
			new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
			for i in range(A.shape[1]):
				new_adj[A[0, i], A[1, i]] = 1.0
			new_adj = new_adj.unsqueeze(0)

			# Apply model on (X,A) as input.
			if self.gpu:
				with torch.no_grad():
					true_pred, attention_weights = self.model(X.cuda(), new_adj.cuda())
					proba = true_pred[0, node_index, :]
			else:
				with torch.no_grad():
						true_pred, attention_weights = self.model(self.data.x, self.adj)
						proba = true_pred[0, node_index, :]

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz

	def node_specific(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx):
		""" Construct z from z' and compute prediction f(z) for each sample z'
			In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

		Args: 
				Variables are defined exactly as defined in explainer function 

		Returns: 
				(tensor): f(z) - probability of belonging to each target classes, for all samples z
				Dimension (N * C) where N is num_samples and C num_classses. 
		"""
		G = torch_geometric.utils.to_networkx(self.data)

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
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

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
			# NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
			X = deepcopy(self.data.x)
			X[node_index, ex_feat] = av_feat_values[ex_feat]
			if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
				X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

			# Special case - consider only nei. influence if too few feat included
			if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):
				# Look at the 2-hop neighbours included
				included_nei = set(self.neighbours.detach().numpy()).difference(ex_nei)
				included_nei = included_nei.difference(one_hop_neighbours.detach().numpy())
				for incl_nei in included_nei:
					l = nx.shortest_path(G, source=node_index, target=incl_nei)
					for n in range(1, len(l)-1):
						A = torch.cat((A, torch.tensor(
							[[l[n-1]], [l[n]]])), dim=-1)
						X[l[n], :] = av_feat_values

			# Transform new data (X, A) to original input form
			new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
			for i in range(A.shape[1]):
				new_adj[A[0, i], A[1, i]] = 1.0
			new_adj = new_adj.unsqueeze(0)

			# Apply model on (X,A) as input.
			if self.gpu:
				with torch.no_grad():
					true_pred, attention_weights = self.model(X.cuda(), new_adj.cuda())
					proba = true_pred[0, node_index, :]
			else:
				with torch.no_grad():
					true_pred, attention_weights = self.model(self.data.x, self.adj)
					proba = true_pred[0, node_index, :]

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz

	def neutral(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx):
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
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

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
			new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
			for i in range(A.shape[1]):
				new_adj[A[0, i], A[1, i]] = 1.0
			new_adj = new_adj.unsqueeze(0)

			# Apply model on (X,A) as input.
			if self.gpu:
				with torch.no_grad():
					true_pred, attention_weights = self.model(X.cuda(), new_adj.cuda())
					proba = true_pred[0, node_index, :]
			else:
				with torch.no_grad():
						true_pred, attention_weights = self.model(self.data.x, self.adj)
						proba = true_pred[0, node_index, :]

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz

		################################

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

		# Define optimizer and loss function
		def weighted_mse_loss(input, target, weight):
			return (weight * (input - target) ** 2).mean()

		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(our_model.parameters(), lr=0.4)

		# Dataloader
		train = torch.utils.data.TensorDataset(z_, fz)
		train_loader = torch.utils.data.DataLoader(train, batch_size=1)

		# Repeat for several epochs
		for epoch in range(50):

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
			tmp = np.linalg.inv(tmp + np.diag(0.01 * np.random.randn(tmp.shape[1])))
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
		if D > 5:
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
							 self.adj,
							 mask,
							 hops,
							 y=self.data.y,
							 threshold=None)

		plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
											self.model.__class__.__name__,
											node_index),
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

		plt.savefig('results/GS_{}_{}_{}'.format(self.data.name,
										   self.model.__class__.__name__,
										   node_index),
												  bbox_inches='tight')

		# plt.show()
