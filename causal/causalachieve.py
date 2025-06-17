import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Endpoint import Endpoint
import io
import logging

# 获取 causalachieve.py 的 logger
logger = logging.getLogger(__name__)

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

# 用于独立测试
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 指定要测试的CSV文件路径（相对于项目根目录）
    test_csv_path = 'MCPdemo/文档/1.csv'
    
    print(f"--- 开始独立测试 run_pc_analysis，使用文件: {test_csv_path} ---")
    
    try:
        # 读取指定的CSV文件内容
        with open(test_csv_path, 'r', encoding='utf-8') as f:
            csv_string = f.read()
        
        # 运行分析函数
        results = run_pc_analysis(csv_string)
        
        # 导入json库并打印格式化后的结果
        import json
        print("\n--- 分析结果 (JSON) ---")
        # 使用 indent=2 美化输出, ensure_ascii=False 以正确显示中文（如果未来有）
        print(json.dumps(results, indent=2, ensure_ascii=False))

    except FileNotFoundError:
        logger.error(f"测试失败：找不到文件 '{test_csv_path}'。请确保文件路径是相对于您运行脚本的目录。")
    except Exception as e:
        logger.error(f"测试过程中发生未知错误: {e}", exc_info=True)
