# The whole frame from GNN Explainer to get data and model 
""" explainer_main.py

	 Main user interface for the explainer module.
"""
import argparse
import os
import pickle
import shutil
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
from tensorboardX import SummaryWriter

import configs
import gengraph
import models
import utils.featgen as featgen
import utils.graph_utils as graph_utils
import utils.io_utils as io_utils
import utils.math_utils as math_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
from explainer import explain
from graphshap_explainer import GraphSHAP


def arg_parse():
	parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
	io_parser = parser.add_mutually_exclusive_group(required=False)
	io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
	benchmark_parser = io_parser.add_argument_group()
	benchmark_parser.add_argument(
		"--bmname", dest="bmname", help="Name of the benchmark dataset"
	)
	io_parser.add_argument("--pkl", dest="pkl_fname",
						   help="Name of the pkl data file")

	parser_utils.parse_optimizer(parser)

	parser.add_argument("--clean-log", action="store_true",
						help="If true, cleans the specified log directory before running.")
	parser.add_argument("--logdir", dest="logdir",
						help="Tensorboard log directory")
	parser.add_argument("--ckptdir", dest="ckptdir",
						help="Model checkpoint directory")
	parser.add_argument("--cuda", dest="cuda", help="CUDA.")
	parser.add_argument(
		"--gpu",
		dest="gpu",
		action="store_const",
		const=True,
		default=False,
		help="whether to use GPU.",
	)
	parser.add_argument(
		"--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
	)
	parser.add_argument(
		"--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
	)
	parser.add_argument(
		"--output-dim", dest="output_dim", type=int, help="Output dimension"
	)
	parser.add_argument(
		"--num-gc-layers",
		dest="num_gc_layers",
		type=int,
		help="Number of graph convolution layers before each pooling",
	)
	parser.add_argument(
		"--bn",
		dest="bn",
		action="store_const",
		const=True,
		default=False,
		help="Whether batch normalization is used",
	)
	parser.add_argument("--dropout", dest="dropout",
						type=float, help="Dropout rate.")
	parser.add_argument(
		"--nobias",
		dest="bias",
		action="store_const",
		const=False,
		default=True,
		help="Whether to add bias. Default to True.",
	)
	parser.add_argument(
		"--no-writer",
		dest="writer",
		action="store_const",
		const=False,
		default=True,
		help="Whether to add bias. Default to True.",
	)
	# Explainer
	parser.add_argument("--mask-act", dest="mask_act",
						type=str, help="sigmoid, ReLU.")
	parser.add_argument(
		"--mask-bias",
		dest="mask_bias",
		action="store_const",
		const=True,
		default=False,
		help="Whether to add bias. Default to True.",
	)
	parser.add_argument(
		"--explain-node", dest="explain_node", type=int, help="Node to explain."
	)
	parser.add_argument(
		"--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
	)
	parser.add_argument(
		"--graph-mode",
		dest="graph_mode",
		action="store_const",
		const=True,
		default=False,
		help="whether to run Explainer on Graph Classification task.",
	)
	parser.add_argument(
		"--multigraph-class",
		dest="multigraph_class",
		type=int,
		help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
	)
	parser.add_argument(
		"--multinode-class",
		dest="multinode_class",
		type=int,
		help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
	)
	parser.add_argument(
		"--align-steps",
		dest="align_steps",
		type=int,
		help="Number of iterations to find P, the alignment matrix.",
	)

	parser.add_argument(
		"--method", dest="method", type=str, help="Method. Possible values: base, att."
	)
	parser.add_argument(
		"--name-suffix", dest="name_suffix", help="suffix added to the output filename"
	)
	parser.add_argument(
		"--explainer-suffix",
		dest="explainer_suffix",
		help="suffix added to the explainer log",
	)
	parser.add_argument(
            "--hops",
            dest="hops",
           	type=int,
            help="k-hop subgraph considered for GraphSHAP",
        )
	parser.add_argument(
            "--num_samples",
            dest="num_samples",
			type=int,
            help="number of samples used to train GraphSHAP",
        )

	# TODO: Check argument usage
	parser.set_defaults(
		logdir="log",
		ckptdir="ckpt",
		dataset="syn2",
		opt="adam",
		opt_scheduler="none",
		#gpu="True",
		cuda="0",
		lr=0.1,
		clip=2.0,
		batch_size=20,
		num_epochs=100,
		hidden_dim=20,
		output_dim=20,
		num_gc_layers=3,
		dropout=0.0,
		method="base",
		name_suffix="",
		explainer_suffix="",
		align_steps=1000,
		explain_node=None,
		graph_idx=-1,
		mask_act="sigmoid",
		multigraph_class=-1,
		multinode_class=-1,
		hops=2,
		num_samples=100,
	)
	return parser.parse_args()


def preprocess_graph(G, labels, normalize_adj=False):
	""" Load an existing graph to be converted for the experiments.
	Args:
		G: Networkx graph to be loaded.
		labels: Associated node labels.
		normalize_adj: Should the method return a normalized adjacency matrix.
	Returns:
		A dictionary containing adjacency, node features and labels
	"""
	# Define adj matrix
	adj = np.array(nx.to_numpy_matrix(G))
	if normalize_adj:
		sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
		adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

	ajd = torch.tensor(adj, dtype=torch.int64)[0]
	edge_index = torch.tensor([[], []], dtype=torch.int64)
	for i, row in enumerate(adj):
		for j, entry in enumerate(row):
	 		if entry != 0:
 				edge_index = torch.cat((edge_index, torch.tensor([[torch.tensor(i, dtype=torch.int64)], [
                            torch.tensor(j, dtype=torch.int64)]],  dtype=torch.int64)), dim=1)

	# Define features
	existing_node = list(G.nodes)[-1]
	feat_dim = G.nodes[existing_node]["feat"].shape[0]
	f = torch.zeros(G.number_of_nodes(), feat_dim)
	for i, u in enumerate(G.nodes()):
		f[i, :] = torch.tensor(G.nodes[u]["feat"])

	# Define labels
	labels = torch.tensor(labels)

	return f, edge_index, labels


def transform_data(adj, x, labels):

	data = SimpleNamespace()

	adj_transfo = torch.tensor(adj, dtype=torch.int64)[0]
	data.edge_index = torch.tensor([[], []], dtype=torch.int64)
	for i, row in enumerate(adj_transfo):

		for j, entry in enumerate(row):
			if entry != 0:
				data.edge_index = torch.cat((data.edge_index, torch.tensor([[torch.tensor(i, dtype=torch.int64)], [
								torch.tensor(j, dtype=torch.int64)]],  dtype=torch.int64)), dim=1)

	# Define features
	feat_dim = x.size(2)
	data.x = torch.zeros(x.size(1), feat_dim)
	for i, u in enumerate(range(x.size(1))):
		data.x[i, :] = torch.tensor(x[0, i, :])

	# Define labels
	data.y = torch.tensor(labels)
	data.num_classes = max(labels) + 1
	data.num_features = x.size(2)
	data.num_nodes = x.size(1)

	return data


def extract_test_nodes(data, num_samples, train_indexes):
	"""
	:param data: dataset
	:param num_samples: number of test samples desired
	:param train_indexes: indexes of training samples
	:return: list of indexes representing nodes used as test samples
	"""

	test_indices = list( set(range(300, 700, 5)) - set(train_indexes) )
	node_indices = np.random.choice(test_indices, num_samples).tolist()

	return node_indices


def main():
	# Load a configuration
	prog_args = arg_parse()

	if prog_args.gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
		print("CUDA", prog_args.cuda)
	else:
		print("Using CPU")

	# Configure the logging directory
	if prog_args.writer:
		path = os.path.join(
			prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
		if os.path.isdir(path) and prog_args.clean_log:
		   print('Removing existing log dir: ', path)
		   if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y":
			   sys.exit(1)
		   shutil.rmtree(path)
		writer = SummaryWriter(path)
	else:
		writer = None

	# Load data and a model checkpoint
	ckpt = io_utils.load_ckpt(prog_args)
	cg_dict = ckpt["cg"]  # get computation graph
	input_dim = cg_dict["feat"].shape[2]
	num_classes = cg_dict["pred"].shape[2]
	print("Loaded model from {}".format(prog_args.ckptdir))
	print("input dim: ", input_dim, "; num classes: ", num_classes)

	# Determine explainer mode (node classif)
	graph_mode = (
		prog_args.graph_mode
		or prog_args.multigraph_class >= 0
		or prog_args.graph_idx >= 0
	)

	# build model
	print("Method: ", prog_args.method)
	if graph_mode:
		# Explain Graph prediction
		model = models.GcnEncoderGraph(
			input_dim=input_dim,
			hidden_dim=prog_args.hidden_dim,
			embedding_dim=prog_args.output_dim,
			label_dim=num_classes,
			num_layers=prog_args.num_gc_layers,
			bn=prog_args.bn,
			args=prog_args,
		)
	else:
		if prog_args.dataset == "ppi_essential":
			# class weight in CE loss for handling imbalanced label classes
			prog_args.loss_weight = torch.tensor(
				[1.0, 5.0], dtype=torch.float).cuda()
		# Explain Node prediction
		model = models.GcnEncoderNode(
			input_dim=input_dim,
			hidden_dim=prog_args.hidden_dim,
			embedding_dim=prog_args.output_dim,
			label_dim=num_classes,
			num_layers=prog_args.num_gc_layers,
			bn=prog_args.bn,
			args=prog_args,
		)
	if prog_args.gpu:
		model = model.cuda()

	# Load state_dict (obtained by model.state_dict() when saving checkpoint)
	model.load_state_dict(ckpt["model_state"])

	# Convertion data required to get correct model output for GraphSHAP
	adj = torch.tensor(cg_dict["adj"], dtype=torch.float)
	x = torch.tensor(cg_dict["feat"], requires_grad=True, dtype=torch.float)
	if prog_args.gpu:
		y_pred, att_adj = model(x.cuda(), adj.cuda())
	else:
		y_pred, att_adj = model(x, adj)

	# Transform their data into our format 
	data = transform_data(adj, x, cg_dict["label"][0].tolist())
	
	# Generate test nodes
	# Use only these specific nodes as they are the ones added manually, part of the defined shapes 
	# node_indices = extract_test_nodes(data, num_samples=10, cg_dict['train_idx'])
	k = 5 # number of nodes for the shape introduced (house, cycle)
	if prog_args.dataset == 'syn1':
		node_indices = list(range(400,450,5))
	elif prog_args.dataset=='syn2':
		node_indices = list(range(400,425,5)) + list(range(1100,1125,5))
	elif prog_args.dataset == 'syn4':
		node_indices = list(range(511,571,6))
		if prog_args.hops == 2:
			k = 4
	elif prog_args.dataset == 'syn5':
		node_indices = list(range(511, 601, 9))
		if prog_args.hops == 2:
			k = 8
		else: 
			k = 11

	# GraphSHAP explainer
	graphshap = GraphSHAP(data, model, adj, writer, prog_args.dataset, prog_args.gpu)

	# Run GNN Explainer and retrieve produced explanations
	gnne = explain.Explainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
        )
	
	# GraphSHAP - assess accuracy of explanations
	# Loop over test nodes
	accuracy = []
	feat_accuracy = []
	for node_idx in node_indices: 
		
		graphshap_explanations = graphshap.explain(node_idx,
                                             hops=prog_args.hops,
                                             num_samples=prog_args.num_samples,
											 info=True)

		# Predicted class
		pred_val, predicted_class = y_pred[0, node_idx, :].max(dim=0)

		# Keep only node explanations 
		graphshap_node_explanations = graphshap_explanations[graphshap.F:,
                                                  predicted_class]
		
		# Derive ground truth from graph structure 
		ground_truth = list(range(node_idx+1,node_idx+k))

		# Retrieve top k elements indices form graphshap_node_explanations
		if graphshap.neighbours.shape[0] > k: 
			i = 0
			val, indices = torch.topk(torch.tensor(
				graphshap_node_explanations), k)
			# could weight importance based on val 
			for node in graphshap.neighbours[indices]: 
				if node in ground_truth:
					i += 1
			# Sort of accruacy metric
			accuracy.append(i / len(indices)) 

			print('There are {} from targeted shape among most imp. nodes'.format(i))
		
		# Look at importance distribution among features
		# Identify most important features and check if it corresponds to truly imp ones
		if prog_args.dataset=='syn2':
			graphshap_feat_explanations = graphshap_explanations[:graphshap.F,
                                                    predicted_class]
			print('Feature importance graphshap', graphshap_feat_explanations)
			if np.argsort(graphshap_feat_explanations)[-1] == 0:
				feat_accuracy.append(1)
			else: 
				feat_accuracy.append(0)

	# Metric for graphshap
	final_accuracy = sum(accuracy)/len(accuracy)
	
	### GNNE 
	# Explain a set of nodes - accuracy on edges this time
	_, gnne_edge_accuracy, gnne_auc, gnne_node_accuracy =\
		gnne.explain_nodes_gnn_stats(
			node_indices, prog_args
		)

	# Tune k inside explain_nodes_gnn_stats (top k nodes/edges investigated)
	
	### GRAD benchmark
	#  MetricS to assess quality of predictionsx
	_, grad_edge_accuracy, grad_auc, grad_node_accuracy =\
            gnne.explain_nodes_gnn_stats(
                node_indices, prog_args, model="grad")

	### GAT
	# Nothing for now - implem a GAT on the side and look at weights coef

	### Results 
	print('Accuracy for GraphSHAP is {:.2f} vs {:.2f},{:.2f} for GNNE vs {:.2f},{:.2f} for GRAD'.format(
		final_accuracy, np.mean(gnne_edge_accuracy), np.mean(gnne_node_accuracy), 
		np.mean(grad_edge_accuracy), np.mean(grad_node_accuracy) ) 
		)
	if prog_args.dataset=='syn2':
		print('Most important feature was found in {:.2f}% of the case'.format(
			100*np.mean(feat_accuracy)))
	
if __name__ == "__main__":
	main()
