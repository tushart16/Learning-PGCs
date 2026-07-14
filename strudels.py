import math
import random
import numpy as np
from enum import Enum
from typing import List, Set, Tuple

class NodeType(Enum):
    LEAF = 1
    SUM = 2
    PROD = 3

class Edge:
    def __init__(self, child_node: 'Node'):
        self.child = child_node
        self.weight = 0.0
        self.aggregate_flow = 0

class Node:
    def __init__(self, node_type: NodeType):
        self.type = node_type
        # Leaf specifics
        self.var_idx = -1
        self.val = -1
        # Children
        self.sum_edges: List[Edge] = []
        self.prod_children: List['Node'] = []
        # State & Structural properties
        self.support = False
        self.context = False
        self.scope: Set[int] = set()

def get_topological_order(root_node: Node) -> List[Node]:
    """Dynamically builds a bottom-up topological list for fast traversals."""
    visited = set()
    ordered = []
    
    def dfs(n: Node):
        if id(n) in visited: return
        visited.add(id(n))
        
        if n.type == NodeType.SUM:
            for edge in n.sum_edges: dfs(edge.child)
        elif n.type == NodeType.PROD:
            for child in n.prod_children: dfs(child)
            
        ordered.append(n)
        
    dfs(root_node)
    return ordered

def calculate_scopes(pc_nodes: List[Node]) -> None:
    """Bottom-up calculation of variables present in each sub-circuit."""
    for node in pc_nodes:
        if node.type == NodeType.LEAF:
            node.scope = {node.var_idx}
        elif node.type == NodeType.SUM:
            node.scope = set()
            for e in node.sum_edges: node.scope.update(e.child.scope)
        elif node.type == NodeType.PROD:
            node.scope = set()
            for c in node.prod_children: node.scope.update(c.scope)

def compute_circuit_flows(pc_nodes: List[Node], data_sample: List[int]) -> None:
    """Runs Pass 1 (Support) and Pass 2 (Flows) instantly."""
    # Pass 1: Bottom-Up Support
    for node in pc_nodes:
        if node.type == NodeType.LEAF:
            node.support = (data_sample[node.var_idx] == node.val)
        elif node.type == NodeType.SUM:
            node.support = any(edge.child.support for edge in node.sum_edges)
        elif node.type == NodeType.PROD:
            node.support = all(child.support for child in node.prod_children)

    # Pass 2: Top-Down Flow
    for node in pc_nodes: node.context = False
    
    root = pc_nodes[-1]
    root.context = root.support
    
    for node in reversed(pc_nodes):
        if not node.context: continue
            
        if node.type == NodeType.SUM:
            for edge in node.sum_edges:
                if edge.child.support:
                    edge.child.context = True
                    edge.aggregate_flow += 1
                    break # Determinism!
        elif node.type == NodeType.PROD:
            for child in node.prod_children:
                child.context = True

def update_parameters_mle(pc_nodes: List[Node]) -> None:
    """Closed-form parameter update based on aggregated traffic."""
    for node in pc_nodes:
        if node.type == NodeType.SUM:
            total_flow = sum(e.aggregate_flow for e in node.sum_edges)
            if total_flow > 0:
                for edge in node.sum_edges:
                    edge.weight = edge.aggregate_flow / total_flow
                    edge.aggregate_flow = 0 # Reset for next batch

def copy_and_condition(node: Node, split_var: int, split_val: int, memo: dict) -> Node:
    """
    Implements Algorithm 9 (Conjoin) and Algorithm 10 (PartialCopy).
    Recursively duplicates a sub-circuit while forcing a variable to a specific value.
    If it detects a logical contradiction, it returns None, destroying the invalid branch.
    """
    if id(node) in memo:
        return memo[id(node)]

    if node.type == NodeType.LEAF:
        # CONTRADICTION DETECTED
        if node.var_idx == split_var and node.val != split_val:
            return None 
            
        new_n = Node(NodeType.LEAF)
        new_n.var_idx, new_n.val = node.var_idx, node.val
        memo[id(node)] = new_n
        return new_n

    elif node.type == NodeType.SUM:
        new_n = Node(NodeType.SUM)
        memo[id(node)] = new_n
        for edge in node.sum_edges:
            new_child = copy_and_condition(edge.child, split_var, split_val, memo)
            if new_child is not None:
                new_edge = Edge(new_child)
                new_edge.weight = edge.weight # Inherit weights, MLE will tune them
                new_n.sum_edges.append(new_edge)
                
        if not new_n.sum_edges:
            return None # Entire sum node became a contradiction
        return new_n

    elif node.type == NodeType.PROD:
        new_n = Node(NodeType.PROD)
        memo[id(node)] = new_n
        for child in node.prod_children:
            new_child = copy_and_condition(child, split_var, split_val, memo)
            if new_child is None:
                return None # AND gate with a contradiction destroys the gate
            new_n.prod_children.append(new_child)
        return new_n

def split_operation(parent: Node, target_edge: Edge, split_var: int):
    """Algorithm 4: Replaces an edge with two mutually exclusive conditioned clones."""
    # 1. Generate the mutually exclusive clones
    copy_0 = copy_and_condition(target_edge.child, split_var, 0, {})
    copy_1 = copy_and_condition(target_edge.child, split_var, 1, {})

    # 2. Sever the old connection
    parent.sum_edges.remove(target_edge)

    # 3. Attach the new deterministic branches
    if copy_0 is not None: parent.sum_edges.append(Edge(copy_0))
    if copy_1 is not None: parent.sum_edges.append(Edge(copy_1))

def initialize_factorized_circuit(num_vars: int) -> Node:
    """Builds a basic, fully-factorized PC as the starting point for growth."""
    var_sums = []
    for i in range(num_vars):
        l0, l1 = Node(NodeType.LEAF), Node(NodeType.LEAF)
        l0.var_idx, l0.val = i, 0
        l1.var_idx, l1.val = i, 1
        
        s = Node(NodeType.SUM)
        s.sum_edges.extend([Edge(l0), Edge(l1)])
        var_sums.append(s)
        
    root = Node(NodeType.PROD)
    root.prod_children = var_sums
    
    # STRUDEL technically requires a Root SUM node to split on.
    master_root = Node(NodeType.SUM)
    master_root.sum_edges.append(Edge(root))
    return master_root

def eFLOW_vRAND_heuristic(pc_nodes: List[Node]) -> Tuple[Node, Edge, int]:
    """
    Finds the highest traffic edge (eFLOW) and picks a variable to split (vRAND).
    """
    best_parent, best_edge, max_flow = None, None, -1

    for node in pc_nodes:
        if node.type == NodeType.SUM:
            for edge in node.sum_edges:
                # We can only split if the child is a Product node and has variables left to split
                if edge.child.type == NodeType.PROD and len(edge.child.scope) > 1:
                    if edge.aggregate_flow > max_flow:
                        max_flow = edge.aggregate_flow
                        best_parent = node
                        best_edge = edge

    if best_edge is None:
        return None, None, None
        
    split_var = random.choice(list(best_edge.child.scope))
    return best_parent, best_edge, split_var

def evaluate_log_likelihood(pc_nodes: List[Node], dataset: List[List[int]]) -> float:
    """Standard bottom-up continuous evaluation to check PC accuracy."""
    total_ll = 0.0
    for sample in dataset:
        node_vals = {}
        for node in pc_nodes:
            if node.type == NodeType.LEAF:
                node_vals[id(node)] = 1.0 if sample[node.var_idx] == node.val else 0.0
            elif node.type == NodeType.SUM:
                node_vals[id(node)] = sum(e.weight * node_vals[id(e.child)] for e in node.sum_edges)
            elif node.type == NodeType.PROD:
                node_vals[id(node)] = math.prod(node_vals[id(c)] for c in node.prod_children)
        
        prob = node_vals[id(pc_nodes[-1])]
        total_ll += math.log(prob + 1e-15)
    return total_ll / len(dataset)

def strudel_fit(dataset: List[List[int]], num_vars: int, max_iterations: int = 10):
    
    # 1. Initialize
    root = initialize_factorized_circuit(num_vars)
    
    for iteration in range(max_iterations):
        # Flatten graph into arrays for fast traversal
        pc_nodes = get_topological_order(root)
        calculate_scopes(pc_nodes)
        
        # 2. Gather traffic (Algorithm 1)
        for sample in dataset:
            compute_circuit_flows(pc_nodes, sample)
            
        # 3. Find the bottleneck edge using EFLOW
        parent, target_edge, split_var = eFLOW_vRAND_heuristic(pc_nodes)
        
        # 4. Lock in parameters (Equation 3)
        update_parameters_mle(pc_nodes)
        
        ll = evaluate_log_likelihood(pc_nodes, dataset)
        print(f"Iteration {iteration:02d} | Nodes: {len(pc_nodes):03d} | Log-Likelihood: {ll:.4f}")
        
        # 5. Grow the circuit (Algorithm 4)
        if parent is None:
            print("Circuit fully saturated. Stopping.")
            break
            
        split_operation(parent, target_edge, split_var)