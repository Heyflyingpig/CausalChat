def metadata_summary(analysis_parameters: dict, visualizations: dict) -> dict:
    """
    根据状态中的数据，构建一份报告需要用到的元数据摘要。
    
    Args:
        analysis_parameters (dict): 包含列级分析结果的字典，来自 get_data_summary 函数
        visualizations (dict): 包含所有可视化图表 Base64 编码的字典
        
    Returns:
        dict: 元数据字典，包含每个图表的描述信息和占位符
        
    Example:
        输入 visualizations = {
            "histograms": {"age": "base64_str_1", "income": "base64_str_2"},
            "barcharts": {"city": "base64_str_3"},
            "correlation_heatmap": "base64_str_4"
        }
        
        输入 analysis_parameters = {
            "column_profiles": {
                "age": {
                    "inferred_type": "continuous",
                    "causal_suitability": "good",
                    "stats": {"mean": 33.0, "min": 22, "max": 50}
                },
                "city": {
                    "inferred_type": "categorical",
                    "unique_count": 4
                }
            }
        }
        
        返回 meta_data = {
            "age": {
                "graph_type": "histograms",
                "inferred_type": "continuous",
                "stats": {"mean": 33.0, "min": 22, "max": 50},
                "description": "age 的分布直方图.",
                "placeholder": "{{CHART:histogram_age}}"
            },
            "city": {
                "graph_type": "barcharts",
                "inferred_type": "categorical",
                "description": "city 的类别分布条形图.",
                "placeholder": "{{CHART:barchart_city}}"
            },
            "correlation_heatmap": {
                "graph_type": "correlation_heatmap",
                "description": "数据集中的变量之间的相关性热力图.",
                "placeholder": "{{CHART:correlation_heatmap}}"
            }
        }
    """
    meta_data = {}
    if "histograms" in visualizations:
        for item_name in visualizations["histograms"].keys():
            analysis_profile = analysis_parameters["column_profiles"].get(item_name, {})
            meta_data[item_name] = {
                "graph_type": "histograms",
                "inferred_type": analysis_profile.get("inferred_type", ""),
                "stats": analysis_profile.get("stats", {}),
            }
            meta_data[item_name]["description"] = f" {item_name} 的分布直方图."
            meta_data[item_name]["placeholder"] = f"[[CHART:histogram_{item_name}]]"
    
    if"barcharts" in visualizations:
        for item_name in visualizations["barcharts"].keys():
            analysis_profile = analysis_parameters["column_profiles"].get(item_name, {})
            meta_data[item_name] = {
                "graph_type": "barcharts",
                "inferred_type": analysis_profile.get("inferred_type", ""),
                "stats": analysis_profile.get("stats", {}),
            }
            meta_data[item_name]["description"] = f" {item_name} 的类别分布条形图."
            meta_data[item_name]["placeholder"] = f"[[CHART:barchart_{item_name}]]"
    
    if "correlation_heatmap" in visualizations:
        meta_data["correlation_heatmap"] = {
            "graph_type": "correlation_heatmap",
            "description": "数据集中的变量之间的相关性热力图.",
            "placeholder": "[[CHART:correlation_heatmap]]"
        }
    return meta_data

def metadata_mapping(meta_data: dict, visualizations: dict) -> dict:
    """
    构建占位符到 Base64 编码的映射表，供报告生成时替换使用。
    
    Args:
        meta_data (dict): 元数据字典，来自 metadata_summary 函数
        visualizations (dict): 包含所有可视化图表 Base64 编码的字典
        
    Returns:
        dict: 占位符映射表，键为占位符字符串，值为对应的 Base64 编码
        
    Example:
        输入 visualizations = {
            "histograms": {"age": "base64_str_1", "income": "base64_str_2"},
            "barcharts": {"city": "base64_str_3"},
            "correlation_heatmap": "base64_str_4"
        }
        
        输入 meta_data = {
            "age": {
                "graph_type": "histograms",
                "placeholder": "{{CHART:histogram_age}}"
            },
            "city": {
                "graph_type": "barcharts",
                "placeholder": "{{CHART:barchart_city}}"
            },
            "correlation_heatmap": {
                "graph_type": "correlation_heatmap",
                "placeholder": "{{CHART:correlation_heatmap}}"
            }
        }
        
        返回 mapping_data = {
            "{{CHART:histogram_age}}": "base64_str_1",
            "{{CHART:barchart_city}}": "base64_str_3",
            "{{CHART:correlation_heatmap}}": "base64_str_4"
        }
    """
    mapping_data = {}
    
    for item_name, item_info in meta_data.items():
        type_name = item_info["graph_type"]
        mapping_value = item_info["placeholder"]
        
        # 特殊处理相关性热力图
        if type_name == "correlation_heatmap":
            visual_data = visualizations["correlation_heatmap"]  # 直接访问
        else:
            visual_data = visualizations[type_name][item_name]  # 嵌套访问
        
        mapping_data[mapping_value] = visual_data
    
    return mapping_data

def replace_placeholders(text: str, visualization_mapping: dict) -> str:
    """
    将文本中的占位符替换为 HTML 图片标签

    Args:
        text: 包含占位符的文本
        visualization_mapping: 占位符到 base64 的映射

    Returns:
        替换后的文本
    """
    import logging

    if not text or not visualization_mapping:
        return text

    result = text
    replaced_count = 0

    # 遍历映射，逐个替换
    for placeholder, base64_str in visualization_mapping.items():
        if placeholder in result:
            # 构建 HTML 图片标签
            html_img = f'<img src="data:image/png;base64,{base64_str}" alt="{placeholder}" style="max-width:100%; height:auto; display:block; margin:20px 0;" />'

            # 替换占位符
            result = result.replace(placeholder, html_img)
            replaced_count += 1

    logging.info(f"成功替换了 {replaced_count}/{len(visualization_mapping)} 个占位符")
    return result