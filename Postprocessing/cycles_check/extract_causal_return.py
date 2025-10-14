import numpy as np
from typing import Tuple, List
import logging

def extract_adjacency_matrix(analysis_result: dict) -> Tuple[np.ndarray, List[str]]:
    """
    从MCP的因果分析结果中提取邻接矩阵和节点名称。
    
    Args:
        analysis_result: MCP返回的因果分析结果字典
        
    Returns:
        (adjacency_matrix, node_names): 邻接矩阵和节点名称列表
        
    注意：
        - 邻接矩阵格式：matrix[i][j] = 1 表示存在边 j -> i
        - 这与causal-learn的格式一致
    """
    try:
        # 从raw_results中提取邻接矩阵
        raw_results = analysis_result.get("raw_results", {})
        adjacency_list = raw_results.get("adjacency_matrix", [])
        
        # 转换为numpy数组
        adjacency_matrix = np.array(adjacency_list)
        
        # 从data中提取节点名称
        data_nodes = analysis_result.get("data", {}).get("nodes", [])
        node_names = [node['id'] for node in data_nodes]
        
        logging.info(f"提取邻接矩阵成功: 形状 {adjacency_matrix.shape}, 节点数 {len(node_names)}")
        return adjacency_matrix, node_names
        
    except Exception as e:
        logging.error(f"提取邻接矩阵时发生错误: {e}", exc_info=True)
        # 返回空矩阵
        return np.array([]), []
