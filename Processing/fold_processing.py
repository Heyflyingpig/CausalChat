from os import name
import pandas as pd

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    从Pandas DataFrame中提取一个结构化的摘要，包含因果推断所需的详细统计信息。
    
    本函数专为因果推断分析设计，不仅提供基础的数据统计信息，还包含变量类型推断、
    因果分析适用性评估、数据质量检查等高级功能，为后续的因果分析提供决策依据。
    
    Args:
        df (pd.DataFrame): 需要分析的pandas DataFrame对象
        
    Returns:
        dict: 包含完整数据摘要的字典，具有以下结构：
        
        {   
            "n_rows": int,
            "n_cols": int,
            "columns": List[str],
            "data_types": {
                "col_name1": "type",
                "col_name2": "type",
                "col_name3": "type",
                ...
            },
            "column_profiles": {
                "col_name1": {
                    "missing_count": int,
                    "missing_ratio": float,
                    "unique_count": int,
                    "is_constant": bool,
                    "inferred_type": str,
                    "causal_suitability": str,
                    "issues": List[str],
                    "value_counts": Dict[str, int],
                    "stats": Dict[str, float],
                    "possible_id": bool,
                    "notes": str,
                },
            },
            "quality_assessment": {
                "total_missing_ratio": float,
                "constant_columns": List[str],
                "high_missing_columns": List[str],
                "suitable_for_causal": {
                    "excellent": List[str],
                }
            }
        }
        基础数据信息:
            n_rows (int): 数据行数
            n_cols (int): 数据列数  
            columns (List[str]): 所有列名列表
            
        数据类型映射:
            data_types (Dict[str, str]): 列名到推断数据类型的映射
                可能的类型值包括:
                - "Binary": 二元数值变量 (0/1, True/False等)
                - "Continuous": 连续数值变量
                - "Categorical (from Numeric)": 数值型分类变量 (<10个唯一值)
                - "Binary Categorical": 二元分类变量 (Male/Female等)
                - "Categorical": 普通分类变量 (<20个唯一值)
                - "High Cardinality Categorical": 高基数分类变量 (≥20个唯一值)
                - "DateTime": 日期时间变量
                - "Empty": 完全缺失的列
                
        详细的列级分析:
            column_profiles (Dict[str, dict]): 每列的详细分析结果，包含:
                
                基础统计信息:
                    missing_count (int): 缺失值数量
                    missing_ratio (float): 缺失率 (0.0-1.0)
                    unique_count (int): 唯一值数量
                    is_constant (bool): 是否为常数列
                    
                类型推断结果:
                    inferred_type (str): 推断的精确数据类型
                        可能值: "binary", "continuous", "categorical_numeric", 
                               "binary_categorical", "categorical", 
                               "high_cardinality_categorical", "datetime", "empty"
                    
                因果分析适用性评估:
                    causal_suitability (str): 在因果分析中的适用性评级
                        - "excellent": 非常适合 (二元变量，理想的treatment候选)
                        - "good": 适合 (连续变量、普通分类变量)
                        - "moderate": 中等适合 (时间变量等，需特殊处理)
                        - "poor": 不太适合 (高基数分类、ID列等)
                        - "unsuitable": 不适合 (完全缺失等)
                    
                条件性字段 (根据数据类型而定):
                    value_counts (Dict): 值分布字典 (仅二元/分类变量，显示前10个)
                    stats (Dict): 描述性统计信息 (仅连续变量)
                        包含: mean, std, min, max, q25, q50, q75
                    possible_id (bool): 是否可能是标识符列
                    notes (str): 特殊说明和建议 (日期时间变量)
                    
                数据质量问题清单:
                    issues (List[str]): 发现的数据质量问题
                        可能包含: "缺失率过高", "常数列", "高基数分类", "疑似ID列"
                        
        整体数据质量评估:
            quality_assessment (dict): 包含以下子项:
                total_missing_ratio (float): 整个数据集的缺失率
                constant_columns (List[str]): 常数列名列表
                high_missing_columns (List[str]): 高缺失率列名列表 (>30%)
                suitable_for_causal (Dict[str, List[str]]): 按适用性分组的列名
                    包含: excellent, good, moderate, poor, unsuitable 五个级别
                    
    """

    
    summary = {}
    summary['n_rows'] = len(df)
    summary['n_cols'] = len(df.columns)
    summary['columns'] = df.columns.tolist()
    
    # 详细的列级分析
    column_profiles = {}
    data_types = {}
    
    for col in df.columns:
        series = df[col]
        
        # 临时储存列的详细信息
        col_profile = {}
        
        # 基础统计，计算缺失值
        # series.isna() 返回布尔掩码，标识缺失值
        col_profile['missing_count'] = int(series.isna().sum())
        col_profile['missing_ratio'] = float(series.isna().mean())
        col_profile['unique_count'] = int(series.nunique(dropna=True))
        
        # 如果唯一值数目小于等于1，则认为该列是常数列
        col_profile['is_constant'] = col_profile['unique_count'] <= 1
        
        # 数据类型推断和格式检测
        # 返回非空值的series
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            # 完全缺失的列
            data_types[col] = 'Empty'
            col_profile['inferred_type'] = 'empty'
            col_profile['causal_suitability'] = 'unsuitable'
            col_profile['issues'] = ['完全缺失']
            
        elif pd.api.types.is_numeric_dtype(series):
            # 数值类型进一步分析
            # 如果唯一值数目为2，则认为该列是二元变量
            if col_profile['unique_count'] == 2:
                data_types[col] = 'Binary'
                col_profile['inferred_type'] = 'binary'
                col_profile['causal_suitability'] = 'excellent'  # 二元变量适合作为treatment
                col_profile['value_counts'] = series.value_counts().to_dict()
            
            # 如果唯一值数目小于10，则认为该列是分类变量
            elif col_profile['unique_count'] < 10:
                data_types[col] = 'Categorical (from Numeric)'
                col_profile['inferred_type'] = 'categorical_numeric'
                col_profile['causal_suitability'] = 'good'
                col_profile['value_counts'] = series.value_counts().head(10).to_dict()
            
            # 如果唯一值数目大于10，则认为该列是连续变量
            else:
                data_types[col] = 'Continuous'
                col_profile['inferred_type'] = 'continuous'
                col_profile['causal_suitability'] = 'good'
                # 连续变量的分布信息
                col_profile['stats'] = {
                    'mean': float(non_null_series.mean()),
                    'std': float(non_null_series.std()),
                    'min': float(non_null_series.min()),
                    'max': float(non_null_series.max()),
                    # 计算四分位数（识别离散程度）
                    'q25': float(non_null_series.quantile(0.25)),
                    'q50': float(non_null_series.quantile(0.5)),
                    'q75': float(non_null_series.quantile(0.75))
                }
                
                # 检测是否可能是整数ID（连续但实际是标识符）
                if (non_null_series % 1 == 0).all() and col_profile['unique_count'] / len(non_null_series) > 0.95:
                    col_profile['possible_id'] = True
                    col_profile['causal_suitability'] = 'poor'  # ID列不适合直接用于因果分析
                    
        else:
            # 非数值类型转化为字符串类型进一步分析
            str_series = non_null_series.astype(str)
            
            # 检测是否为日期时间格式
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'^\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                r'^\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            
            is_date_like = False
            
            for pattern in date_patterns:
                if str_series.str.match(pattern).sum() > len(str_series) * 0.8:
                    is_date_like = True
                    break
                    
            # 尝试解析为日期时间
            if not is_date_like:
                try:
                    pd.to_datetime(str_series.head(min(100, len(str_series))))
                    is_date_like = True
                except:
                    pass
            
            if is_date_like:
                data_types[col] = 'DateTime'
                col_profile['inferred_type'] = 'datetime'
                col_profile['causal_suitability'] = 'moderate'  # 时间变量需要特殊处理
                col_profile['notes'] = '可能需要提取时间特征（年、月、季度等）用于因果分析'
            else:
                # 普通分类变量
                # 二元分类
                if col_profile['unique_count'] == 2:
                    data_types[col] = 'Binary Categorical'
                    col_profile['inferred_type'] = 'binary_categorical'
                    col_profile['causal_suitability'] = 'excellent'
                # 小基数分类
                elif col_profile['unique_count'] < 20:
                    data_types[col] = 'Categorical'
                    col_profile['inferred_type'] = 'categorical'
                    col_profile['causal_suitability'] = 'good'
                # 高基数分类
                else:
                    data_types[col] = 'High Cardinality Categorical'
                    col_profile['inferred_type'] = 'high_cardinality_categorical'
                    col_profile['causal_suitability'] = 'poor'  # 高基数分类变量需要特殊处理
                
                # 解析出唯一值以及其出现次数    
                col_profile['value_counts'] = series.value_counts().head(10).to_dict()
        
        
        # 数据质量问题检测
        issues = []
        if col_profile['missing_ratio'] > 0.3:
            issues.append(f"缺失率过高({col_profile['missing_ratio']:.1%})")
        if col_profile['is_constant']:
            issues.append("常数列或唯一值过少")
        if col_profile['inferred_type'] == 'high_cardinality_categorical':
            issues.append("分类变量基数过高，可能需要编码或分组")
        if col_profile.get('possible_id', False):
            issues.append("疑似ID列，不适合直接用于因果分析")
            
        col_profile['issues'] = issues
        column_profiles[col] = col_profile
    
    summary['data_types'] = data_types
    summary['column_profiles'] = column_profiles
    
    # 整体数据质量评估
    quality_assessment = {
        # 整体缺失率
        'total_missing_ratio': float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])),
        
        # 常数列
        'constant_columns': [col for col, col_profile in column_profiles.items() if col_profile['is_constant']],
        
        # 高缺失率列
        'high_missing_columns': [col for col, col_profile in column_profiles.items() if col_profile['missing_ratio'] > 0.3],
        # 因果分析适宜性
        'suitable_for_causal': {
            'excellent': [col for col, col_profile in column_profiles.items() if col_profile.get('causal_suitability') == 'excellent'],
            'good': [col for col, col_profile in column_profiles.items() if col_profile.get('causal_suitability') == 'good'],
            'moderate': [col for col, col_profile in column_profiles.items() if col_profile.get('causal_suitability') == 'moderate'],
            'poor': [col for col, col_profile in column_profiles.items() if col_profile.get('causal_suitability') == 'poor'],
            'unsuitable': [col for col, col_profile in column_profiles.items() if col_profile.get('causal_suitability') == 'unsuitable']
        }
    }
 
    summary['quality_assessment'] = quality_assessment

    
    return summary

if __name__ == "__main__":
    # 测试代码可以在这里添加
    pass