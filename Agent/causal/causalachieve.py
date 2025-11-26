import numpy as np
import pandas as pd
import io
import logging

# causal-learn:PC algorithm 
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Endpoint import Endpoint

# CDMIR: OLC algorithm 
# 使用延迟导入 + 优雅降级，避免未安装时启动报错
_CDMIR_AVAILABLE = False
try:
    from cdmir.discovery.funtional_based.one_component.olc import olc
    _CDMIR_AVAILABLE = True
except ImportError:
    olc = None  # 占位，防止后续代码引用报错

logger = logging.getLogger(__name__)


def is_cdmir_available() -> bool:
    """检查 CDMIR 库是否可用"""
    return _CDMIR_AVAILABLE

def _format_edges(causallearn_edges):
    """将 Causal-learn 的边对象转换为 vis-network 兼容的格式。"""
    formatted_edges = []
    for edge in causallearn_edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        end1 = edge.get_endpoint1()
        end2 = edge.get_endpoint2()
        
        vis_edge = {
            'from': node1.get_name(),
            'to': node2.get_name(),
            # 使用边的字符串表示作为标签，方便调试
            'label': str(edge)
        }

        # 根据端点类型为 vis.js 设置箭头
        arrows = []
        if end2 == Endpoint.ARROW:
            arrows.append('to')
        if end1 == Endpoint.ARROW:
            arrows.append('from')
        
        if arrows:
            vis_edge['arrows'] = ','.join(arrows)

        # 如果边包含圆圈（代表不确定性），则使用虚线表示
        if end1 == Endpoint.CIRCLE or end2 == Endpoint.CIRCLE:
            vis_edge['dashes'] = True

        formatted_edges.append(vis_edge)
    return formatted_edges


def run_pc_analysis(csv_data_string: str) -> dict:
    """
    对CSV格式的字符串数据运行PC因果发现算法。
    
    Returns:
        一个包含分析结果的字典，包括节点、边和邻接矩阵，
        用于在前端动态生成图表。
    """
    try:
        logger.info("开始从字符串加载数据...")
        string_io = io.StringIO(csv_data_string)
        df = pd.read_csv(string_io)
        
        if df.empty or len(df.columns) < 2:
            msg = "错误：CSV数据为空或列数少于2，无法进行因果分析。"
            logger.error(msg)
            return {"success": False, "message": msg}
        
        logger.info(f"数据加载成功，包含 {len(df)} 行和 {len(df.columns)} 列。")
        data = df.to_numpy()
        node_names = df.columns.tolist()

        logger.info("正在运行PC算法...")
        cg = pc(data=data, alpha=0.05, indep_test=fisherz, node_names=node_names)
        logger.info("PC算法完成。")

        # 提取结果
        edges = cg.G.get_graph_edges()
        
        # 准备前端需要的数据格式
        nodes_for_vis = [{'id': name, 'label': name} for name in node_names]
        edges_for_vis = _format_edges(edges)
        
        logger.info(f"格式化后的边: {edges_for_vis}")

        return {
            "success": True,
            "message": "因果分析成功完成。",
            "data": {
                "nodes": nodes_for_vis,
                "edges": edges_for_vis,
            },
            "raw_results": {
                "edges": [str(edge).strip() for edge in edges],
                "adjacency_matrix": cg.G.graph.tolist()
            },
            "analyzed_filename": None
        }

    except Exception as e:
        error_message = f"执行因果分析时发生错误: {e}"
        logger.error(error_message, exc_info=True)
        return {"success": False, "message": error_message}


def _format_olc_edges(adjacency_matrix: np.ndarray, coefficient_matrix: np.ndarray,
                       node_names: list) -> list:
    """
    将 OLC 算法的邻接矩阵转换为 vis-network 兼容的边格式。

    参数:
        adjacency_matrix: OLC 返回的邻接矩阵
            - 0: 无边
            - 1: 有向边 (row → column)
            - 2: 无向边（双向）
        coefficient_matrix: OLC 返回的系数矩阵，表示因果效应强度
        node_names: 所有节点名称（包括观测变量 + 潜变量）

    返回:
        vis-network 格式的边列表
    """
    formatted_edges = []
    n = adjacency_matrix.shape[0]

    # 用于记录已处理的无向边，避免重复添加
    processed_undirected = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            edge_type = adjacency_matrix[i, j]

            if edge_type == 0:
                # 无边，跳过
                continue

            elif edge_type == 1:
                # 有向边: i → j
                coef = coefficient_matrix[j, i]  # 注意：系数矩阵是 [j, i] 存储 i→j 的系数
                vis_edge = {
                    'from': node_names[i],
                    'to': node_names[j],
                    'arrows': 'to',
                    'label': f'{coef:.3f}' if coef != 0 else ''
                }
                formatted_edges.append(vis_edge)

            elif edge_type == 2:
                # 无向边（双向）: i -- j
                # 为避免重复，只在 i < j 时添加
                edge_key = (min(i, j), max(i, j))
                if edge_key not in processed_undirected:
                    processed_undirected.add(edge_key)
                    vis_edge = {
                        'from': node_names[i],
                        'to': node_names[j],
                        'arrows': '',  # 无箭头表示无向
                        'dashes': True,  # 用虚线表示不确定性
                        'label': '无向'
                    }
                    formatted_edges.append(vis_edge)

    return formatted_edges


def run_olc_analysis(csv_data_string: str, alpha: float = 0.05, beta: float = 0.01) -> dict:
    """
    对 CSV 格式的字符串数据运行 OLC (One-Component Latent) 因果发现算法。

    OLC 算法特点:
        - 能够检测隐藏混杂因子（潜变量）
        - 适用于存在未观测变量影响多个观测变量的场景
        - 基于四阶累积量进行潜变量检测

    参数:
        csv_data_string: CSV 格式的字符串数据
        alpha: 主显著性水平，用于边定向（默认 0.05）
        beta: 次显著性水平，用于潜变量检测（默认 0.01，更严格）

    Returns:
        一个包含分析结果的字典。
        包括节点、边和邻接矩阵，用于在前端动态生成图表。
    """
    # 检查 CDMIR 是否可用
    if not _CDMIR_AVAILABLE:
        msg = "OLC 算法不可用：CDMIR 库未安装。请运行 'pip install git+https://github.com/DMIRLAB-Group/CDMIR.git' 安装。"
        logger.warning(msg)
        return {"success": False, "message": msg, "algorithm": "olc"}

    try:
        logger.info("OLC: 开始从字符串加载数据...")
        string_io = io.StringIO(csv_data_string)
        df = pd.read_csv(string_io)

        if df.empty or len(df.columns) < 2:
            msg = "错误：CSV数据为空或列数少于2，无法进行因果分析。"
            logger.error(msg)
            return {"success": False, "message": msg}

        logger.info(f"OLC: 数据加载成功，包含 {len(df)} 行和 {len(df.columns)} 列。")

        # OLC 需要 numpy 数组作为输入
        data = df.to_numpy()
        observed_node_names = df.columns.tolist()
        n_observed = len(observed_node_names)

        logger.info("OLC: 正在运行 OLC 算法...")
        # OLC 返回两个矩阵：邻接矩阵和系数矩阵
        adjacency_matrix, coefficient_matrix = olc(
            data=data,
            alpha=alpha,
            beta=beta,
            verbose=True
        )
        logger.info("OLC: 算法完成。")

        # 检测是否发现了潜变量
        # OLC 输出的矩阵维度 = n_observed + n_latent
        n_total = adjacency_matrix.shape[0]
        n_latent = n_total - n_observed

        # 构建完整的节点名称列表（观测变量 + 潜变量）
        all_node_names = observed_node_names.copy()
        for i in range(n_latent):
            latent_name = f"L{i+1}"  # 潜变量命名为 L1, L2, ...
            all_node_names.append(latent_name)

        logger.info(f"OLC: 检测到 {n_latent} 个潜变量")

        # 准备前端需要的数据格式
        nodes_for_vis = []
        for i, name in enumerate(all_node_names):
            node = {'id': name, 'label': name}
            # 为潜变量添加特殊标记，方便前端区分样式
            if i >= n_observed:
                node['group'] = 'latent'  # 标记为潜变量组
                node['shape'] = 'diamond'  # 可选：用菱形表示潜变量
            else:
                node['group'] = 'observed'
            nodes_for_vis.append(node)

        # 转换边格式
        edges_for_vis = _format_olc_edges(adjacency_matrix, coefficient_matrix, all_node_names)

        logger.info(f"OLC: 格式化后的边数量: {len(edges_for_vis)}")

        return {
            "success": True,
            "message": f"OLC 因果分析成功完成。检测到 {n_latent} 个潜变量。",
            "data": {
                "nodes": nodes_for_vis,
                "edges": edges_for_vis,
            },
            "raw_results": {
                "adjacency_matrix": adjacency_matrix.tolist(),
                "coefficient_matrix": coefficient_matrix.tolist(),
                "n_observed": n_observed,
                "n_latent": n_latent,
                "observed_names": observed_node_names,
                "latent_names": [f"L{i+1}" for i in range(n_latent)]
            },
            "analyzed_filename": None
        }

    except Exception as e:
        error_message = f"OLC: 执行因果分析时发生错误: {e}"
        logger.error(error_message, exc_info=True)
        return {"success": False, "message": error_message}
# 用于独立测试
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 指定要测试的CSV文件路径（相对于项目根目录）
    test_csv_path = '杂项/test/1.csv'
    
    print(f"开始独立测试 run_pc_analysis，使用文件: {test_csv_path}")
    
    try:
        # 读取指定的CSV文件内容
        with open(test_csv_path, 'r', encoding='utf-8') as f:
            csv_string = f.read()
        
        # 运行分析函数
        results = run_pc_analysis(csv_string)
        
        # 导入json库并打印格式化后的结果
        import json
        print("\n分析结果 (JSON)")
        # 使用 indent=2 美化输出, ensure_ascii=False 以正确显示中文（如果未来有）
        print(json.dumps(results, indent=2, ensure_ascii=False))

    except FileNotFoundError:
        logger.error(f"测试失败：找不到文件 '{test_csv_path}'。请确保文件路径是相对于您运行脚本的目录。")
    except Exception as e:
        logger.error(f"测试过程中发生未知错误: {e}", exc_info=True)
