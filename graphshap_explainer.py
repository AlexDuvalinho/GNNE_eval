import scipy.special
import numpy as np
from copy import deepcopy
import torch_geometric
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import os
from copy import copy
from math import sqrt
import networkx as nx
import tensorboardX
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from utils.io_utils import gen_explainer_prefix, gen_prefix

class GraphSHAP():

	def __init__(self, data, model, adj, writer, args_dataset):
		self.model = model
		self.data = data
		self.model.eval()
		self.M = None  # number of nonzero features - for each node index
		self.neighbours = None
		self.F = None
		self.adj = adj
		self.writer = writer
		self.args_dataset = args_dataset

	def explain(self, node_index=0, hops=2, num_samples=10, info=True):
		"""
		:param node_index: index of the node of interest
		:param hops: number k of k-hop neighbours to consider in the subgraph around node_index
		:param num_samples: number of samples we want to form GraphSHAP's new dataset 
		:return: shapley values for features/neighbours that influence node v's pred
		"""

		### Determine z' => features and neighbours whose importance is investigated

		# Create a variable to store node features
		x = self.data.x[node_index, :]

		# Number of non-zero entries for the feature vector x_v
		self.F = x[x != 0].shape[0]
		# Store indexes of these non zero feature values
		feat_idx = torch.nonzero(x)

		# Construct k hop subgraph of node of interest (denoted v)
		self.neighbours, _, _, edge_mask =\
                    torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                         num_hops=hops,
                                                         edge_index=self.data.edge_index)
		# Store the indexes of the neighbours of v (+ index of v itself)

		# Remove node v index from neighbours and store their number in D
		self.neighbours = self.neighbours[self.neighbours != node_index]
		D = self.neighbours.shape[0]

		# Total number of features + neighbours considered for node v
		self.M = self.F+D

		# Sample z' - binary vector of dimension (num_samples, M)
		# F node features first, then D neighbours
		z_ = torch.empty(num_samples, self.M).random_(2)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)

		# Compute true prediction of model, for original instance
		true_pred, attention_weights = self.model(self.data.x, self.adj)

		### Define weights associated with each sample using shapley kernel formula
		weights = self.shapley_kernel(s)

		###  Create dataset (z', f(z)), stored as (z_, fz)
		# Retrive z from z' and x_v, then compute f(z)
		fz = self.compute_pred(node_index, num_samples, D, z_, feat_idx)

		### OLS estimator for weighted linear regression
		phi = self.OLS(z_, weights, fz)  # dim (M*num_classes)

		### Print some information
		if info:
			self.print_info(D, node_index, phi, feat_idx)

		### Visualisation
			_, predicted_class = true_pred[0,node_index,:].max(dim=0)
			
			weighted_edge_mask = self.weighted_edge_mask(edge_mask, node_index, phi,
			          predicted_class, hops)
			
			G = self.denoise_graph(weighted_edge_mask, phi[self.F:,predicted_class], 
						node_index, feat=self.data.x, label=self.data.y, threshold_num=10)

			self.log_graph(
							self.writer,
							G,
                            "graph/{}_{}".format(self.args_dataset,
                                                    node_index),
							identify_self=True,
                        )

			# Vizu nodes and
			# ax, G = self.visualize_subgraph(self.model,
			# 						node_index,
			# 						self.data.edge_index,
			# 						weighted_edge_mask,
			# 						hops,
			# 						y=self.data.y,
			# 						threshold=None)

		# plt.savefig('demo.png', bbox_inches='tight')

		return phi

	def shapley_kernel(self, s):
		"""
		:param s: dimension of z' (number of features + neighbours included)
		:return: [scalar] value of shapley value 
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

	def compute_pred(self, node_index, num_samples, D, z_, feat_idx):
		"""
		Variables are exactly as defined in explainer function, where compute_pred is used
		This function aims to construct z (from z' and x_v) and then to compute f(z), 
		meaning the prediction of the new instances with our original model. 
		In fact, it builds the dataset (z', f(z)), required to train the weighted linear model.
		:return fz: probability of belonging to each target classes, for all samples z
		fz is of dimension N*C where N is num_samples and C num_classses. 
		"""
		# This implies retrieving z from z' - wrt sampled neighbours and node features
		# We start this process here by storing new node features for v and neigbours to
		# isolate
		X_v = torch.zeros([num_samples, self.data.num_features])
		excluded_nei = {}

		# Do it for each sample
		for i in range(num_samples):

			# Define new node features dataset (we only modify x_v for now)
			# Features where z_j == 1 are kept, others are set to 0
			for j in range(self.F):
				if z_[i, j].item() == 1:
					X_v[i, feat_idx[j].item()] = 1

			# Define new neighbourhood
			# Store index of neighbours that need to be shut down (not sampled, z_j=0)
			nodes_id = []
			for j in range(D):
				if z_[i, self.F+j] == 0:
					node_id = self.neighbours[j].item()
					nodes_id.append(node_id)
			# Dico with key = num_sample id, value = excluded neighbour index
			excluded_nei[i] = nodes_id

		# Init label f(z) for graphshap dataset - consider all classes
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

		# Create new matrix A and X - for each sample ≈ reform z from z'
		for key, value in excluded_nei.items():

			positions = []
			# For each excluded neighbour, retrieve the column index of each occurence
			# in the adj matrix - store in positions (list)
			for val in value:
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
			X[node_index, :] = X_v[key, :]
			# Transform new data (X, A) to original input form
			new_adj = torch.zeros(self.data.x.size(0), self.data.x.size(0))
			for i in range(self.data.edge_index.shape[1]):
				new_adj[self.data.edge_index[0, i], self.data.edge_index[1, i]] = 1.0
			new_adj = new_adj.unsqueeze(0)

			# Apply model on (X,A) as input.
			proba, _ = self.model(X, new_adj)
			proba = proba[0, node_index,:]

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz


	def OLS(self, z_, weights, fz):
		"""
		:param z_: z' - binary vector  
		:param weights: shapley kernel weights for z'
		:param fz: f(z) where z is a new instance - formed from z' and x
		:return: estimated coefficients of our weighted linear regression - on (z', f(z))
		phi is of dimension (M * num_classes)
		"""
		# OLS to estimate parameter of Weighted Linear Regression
		try:
			tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
		except np.linalg.LinAlgError:  # matrix not invertible
			tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
			tmp = np.linalg.inv(tmp + np.diag(np.random.randn(tmp.shape[1])))
		phi = np.dot(tmp, np.dot(
			np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
		return phi

	def print_info(self, D, node_index, phi, feat_idx):
		"""
		Displays some information about explanations - for a better comprehension and audit
		"""

		# Print some information
		print('Explanations include {} node features and {} neighbours for this node\
		for {} classes'.format(self.F, D, self.data.num_classes))

		# Compare with true prediction of the model - see what class should truly be explained
		true_pred, _ = self.model(self.data.x, self.adj)
		pred_value, true_pred = true_pred[0,node_index,:].max(dim=0)
		print('Prediction of orignal model is class {} while label is {}'.format(
			true_pred, self.data.y[node_index]))

		# Isolate explanations for predicted class - explain model choices
		pred_explanation = phi[:, true_pred]
		# print('Explanation for the class predicted by the model:', pred_explanation)

		# Look at repartition of weights among neighbours and node features
		# Motivation for regularisation
		sum_feat = sum_nei = 0
		for i in range(len(pred_explanation)):
			if i < self.F:
				sum_feat += np.abs(pred_explanation[i])
			else:
				sum_nei += np.abs(pred_explanation[i])
		print('Total weights for node features: ', sum_feat)
		print('Total weights for neighbours: ', sum_nei)

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
			print('Most influential features: ', [(item[0].item(), item[1].item()) for item in list(influential_feat.items())],
                            'and neighbours', [(item[0].item(), item[1].item()) for item in list(influential_nei.items())])

		# Most influential features splitted bewteen neighbours and features
		if self.F > 5:
			_, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation[:self.F])), 3)
			vals = [pred_explanation[idx] for idx in idxs]
			influential_feat = {}
			for idx, val in zip(idxs, vals):
				influential_feat[feat_idx[idx]] = val
			print('Most influential features: ', [
			      (item[0].item(), item[1].item()) for item in list(influential_feat.items())])

		# Most influential features splitted bewteen neighbours and features
		if D > 5:
			_, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation[self.F:])), 3)
			vals = [pred_explanation[self.F + idx] for idx in idxs]
			influential_nei = {}
			for idx, val in zip(idxs, vals):
				influential_nei[self.neighbours[idx]] = val
			print('Most influential neighbours: ', [
			      (item[0].item(), item[1].item()) for item in list(influential_nei.items())])

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
						mask[idx] = phi[self.F + i, predicted_class]
					else:
						pass
				elif mask[idx] == 1:
					mask[idx] = phi[self.F + i, predicted_class]
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
			if node!=node_idx:
				if node_explanations[i] < threshold:
					del_nodes.append(node)
		G.remove_nodes_from(del_nodes)
		
		# Remove isolated nodes
		G.remove_nodes_from(list(nx.isolates(G)))

		# Remove disconnected components without node_idx
		if not nx.is_connected(G):
			for comp in nx.connected_components(G):
				if node in comp:
					G = G.subgraph(list(comp))

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

		if Gc.number_of_nodes() == 0:
			raise Exception("empty graph")
		if Gc.number_of_edges() == 0:
			raise Exception("empty edge")
		# remove_nodes = []
		# for u in Gc.nodes():
		#    if Gc
		pos_layout = nx.kamada_kawai_layout(Gc, weight=None)
		#pos_layout = nx.spring_layout(Gc, weight=None)

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

