from typing import Tuple, List
import numpy as np
import networkx as nx
import logging

def detect_cycles(adjacency_matrix: np.ndarray, node_names: List[str]) -> Tuple[bool, List[List[str]]]:
    """
    检测因果图中是否存在环路。
    
    Args:
        adjacency_matrix: 邻接矩阵 (n x n)
        node_names: 节点名称列表
        
    Returns:
        (has_cycle, cycles): 是否有环路，以及所有环路的列表
        
    技术细节：
        - 使用networkx构建有向图
        - 调用is_directed_acyclic_graph检测环路
        - 如果存在环路，使用simple_cycles找出所有环路
    """
    try:
        if adjacency_matrix.size == 0:
            return False, []
        
        # 构建有向图：只保留值为1的边（确定的有向边）
        # 创建一个副本，将非0值转换为1（简化处理）
        # 由于causal-learn的邻接矩阵会重复输出-1值，所以需要先转换为1
        adj_binary = (adjacency_matrix == 1).astype(int)
        
        # 转置矩阵，因为networkx的约定是adj[i][j]=1表示i->j
        # 而causal-learn的约定是adj[i][j]=1表示j->i
        adj_for_nx = adj_binary.T
        
        # 创建有向图
        G = nx.from_numpy_array(adj_for_nx, create_using=nx.DiGraph)
        
        # 重命名节点为实际变量名
        mapping = {i: node_names[i] for i in range(len(node_names))}
        G = nx.relabel_nodes(G, mapping)
        
        # 检测是否为有向无环图(DAG)
        is_acyclic = nx.is_directed_acyclic_graph(G)
        
        if is_acyclic:
            logging.info("因果图检查：无环路，符合DAG要求。")
            return False, []
        else:
            # 找出所有环路
            cycles = list(nx.simple_cycles(G))
            logging.warning(f"因果图检查：检测到 {len(cycles)} 个环路！")
            for i, cycle in enumerate(cycles):
                logging.warning(f"  环路 {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
            return True, cycles
            
    except Exception as e:
        logging.error(f"环路检测时发生错误: {e}", exc_info=True)
        return False, []
