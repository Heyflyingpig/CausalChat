def validate_analysis(analysis_params: dict, target: str = None, treatment: str = None) -> tuple[bool, list[str]]:
    """
    基于增强的数据摘要进行因果分析就绪性验证。
    
    Args:
        analysis_params: get_data_summary返回的完整数据摘要
        target: 目标变量名
        treatment: 处理变量名
        
    Returns:
        tuple[bool, list[str]]: (是否通过验证, 问题列表)
    """
    issues = []
    columns = analysis_params.get("columns", [])
    column_profiles = analysis_params.get("column_profiles", {})
    quality_assessment = analysis_params.get("quality_assessment", {})
    
    # 1. 基本参数检查
    if not target or not treatment:
        issues.append("缺少目标变量(target)或处理变量(treatment)的指定")
    else:
        if target not in columns:
            issues.append(f"目标变量 '{target}' 不存在于数据列中")
        if treatment not in columns:
            issues.append(f"处理变量 '{treatment}' 不存在于数据列中")
        if target == treatment:
            issues.append("目标变量与处理变量不能相同")

        # 2. 变量质量检查
        for var_name, var_type in [("目标变量", target), ("处理变量", treatment)]:
            if var_type in column_profiles:
                profile = column_profiles[var_type]
                
                # 检查是否为不适用的列类型
                if profile.get('causal_suitability') == 'unsuitable':
                    issues.append(f"{var_name} '{var_type}' 完全缺失，无法用于分析")
                elif profile.get('causal_suitability') == 'poor':
                    if profile.get('possible_id', False):
                        issues.append(f"{var_name} '{var_type}' 疑似ID列，不适合直接用于因果分析")
                    elif profile.get('inferred_type') == 'high_cardinality_categorical':
                        issues.append(f"{var_name} '{var_type}' 为高基数分类变量，建议先进行编码或分组处理")
                
                # 检查缺失率
                if profile.get('missing_ratio', 0) > 0.3:
                    issues.append(f"{var_name} '{var_type}' 缺失率过高({profile['missing_ratio']:.1%})，建议先处理缺失值")
                
                # 检查是否为常数列
                if profile.get('is_constant', False):
                    issues.append(f"{var_name} '{var_type}' 为常数列或唯一值过少，无法进行有效分析")

    # 3. 数据规模检查
    n_rows = analysis_params.get('n_rows', 0)
    if n_rows < 50:
        issues.append(f"样本量过小(n={n_rows})，建议至少有50个观测值以确保分析的稳健性")
    
    # 4. 整体数据质量检查
    total_missing_ratio = quality_assessment.get('total_missing_ratio', 0)
    if total_missing_ratio > 0.2:
        issues.append(f"整体缺失率过高({total_missing_ratio:.1%})，建议先进行数据清洗")
    
    # 5. 常数列检查
    constant_columns = quality_assessment.get('constant_columns', [])
    if len(constant_columns) > 0:
        issues.append(f"存在常数列: {', '.join(constant_columns[:3])}{'等' if len(constant_columns) > 3 else ''}，建议移除")
    
    # 6. 特殊格式建议（不算阻断性错误，但给出提示）
    datetime_columns = [col for col, profile in column_profiles.items() 
                       if profile.get('inferred_type') == 'datetime']
    if datetime_columns and len(datetime_columns) > 0:
        # 这不算错误，但给出提示
        pass  # 可以在后续添加时间序列因果分析的特殊提示
    
    return (len(issues) == 0, issues)