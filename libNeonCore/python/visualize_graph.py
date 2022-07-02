"""
Reads and plots the dot file
Dependencies: pydot, networkx and matplotlib
"""

import argparse
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Plot graph dot files")
parser.add_argument("input_file", type=str, help="Path to dot file")
args, _ = parser.parse_known_args()

G = nx.DiGraph(read_dot(args.input_file))

# Collect edge labels
edge_labels = {}
for edge in G.edges:
    edata = G[edge[0]][edge[1]]
    label = edata.get("label", None)
    if label:
        label = label.strip('"')
        edge_labels[(edge[0], edge[1])] = str(label)

# Collect vertex labels
vert_labels = {}
for node in G.nodes:
    vdata = G.nodes[node]
    label = vdata.get("label", None)
    if label:
        label = label.strip('"')
        vert_labels[node] = str(label)

# Draw
pos = nx.planar_layout(G)
if len(edge_labels) > 0:
    nx.draw_networkx_edge_labels(
        G, pos=pos, edge_labels=edge_labels, font_color="red",
    )

nx.draw(
    G,
    pos=pos,
    labels=vert_labels if len(vert_labels) > 0 else None,
    with_labels=True,
    edge_color="black",
    width=1,
    linewidths=1,
    node_color="gray",
    alpha=1.0,
)

# Show
plt.show()
