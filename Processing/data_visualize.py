import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any

# 设置绘图风格，避免中文乱码问题
# 使用 matplotlib 的通用字体回退机制，兼容 Docker 环境
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def _convert_plot_to_base64(fig: plt.Figure) -> str:
    """
    将 matplotlib 图表对象转换为 Base64 编码的字符串。
    这是一个内部辅助函数，用于将生成的图表序列化，方便在状态(state)中传递。

    Args:
        fig (plt.Figure): matplotlib 的图表对象。

    Returns:
        str: PNG 格式图表的 Base64 编码字符串。
    """
    # 创建一个内存中的二进制缓冲区
    buf = io.BytesIO()
    
    # 将图表以PNG格式保存到缓冲区
    # bbox_inches='tight' 会自动裁剪图表周围的空白
    fig.savefig(buf, format='png', bbox_inches='tight')
    
    # 关闭图表对象，释放内存
    plt.close(fig)
    
    # 将缓冲区的指针移到开头
    buf.seek(0)
    
    # 读取缓冲区内容并进行 Base64 编码，然后解码为 UTF-8 字符串
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 关闭缓冲区
    buf.close()
    
    return image_base64

def plot_numerical_distribution(df: pd.DataFrame, column: str) -> str:
    """
    为指定的数值型列生成分布直方图。

    Args:
        df (pd.DataFrame): 数据源。
        column (str): 需要可视化的数值型列名。

    Returns:
        str: 图表的 Base64 编码字符串。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # 使用 seaborn 的 histplot 绘制直方图，kde=True 会同时绘制核密度估计曲线
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of '{column}'", fontsize=14)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    fig.tight_layout()
    return _convert_plot_to_base64(fig)

def plot_categorical_distribution(df: pd.DataFrame, column: str) -> str:
    """
    为指定的类别型列生成计数条形图。

    Args:
        df (pd.DataFrame): 数据源。
        column (str): 需要可视化的类别型列名。

    Returns:
        str: 图表的 Base64 编码字符串。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # 使用 seaborn 的 countplot 绘制计数条形图
    # order 参数可以按计数的降序排列条形
    order = df[column].value_counts().index
    sns.countplot(y=df[column], ax=ax, order=order)
    ax.set_title(f"Categorical Distribution of '{column}'", fontsize=14)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    fig.tight_layout()
    return _convert_plot_to_base64(fig)

def plot_correlation_heatmap(df: pd.DataFrame, numerical_columns: List[str]) -> str:
    """
    为指定的数值型列生成相关性热力图。

    Args:
        df (pd.DataFrame): 数据源。
        numerical_columns (List[str]): 需要计算相关性的数值型列名列表。

    Returns:
        str: 图表的 Base64 编码字符串。
    """
    # 仅当数值列多于一个时，计算相关性才有意义
    if len(numerical_columns) < 2:
        return ""
        
    fig, ax = plt.subplots(figsize=(12, 10))
    # 计算相关性矩阵
    correlation_matrix = df[numerical_columns].corr()
    # 使用 seaborn 的 heatmap 绘制热力图
    # annot=True 会在格子上显示数值
    # cmap='coolwarm' 是一个常用的颜色映射方案
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Variables", fontsize=16)
    fig.tight_layout()
    return _convert_plot_to_base64(fig)

def generate_visualizations(df: pd.DataFrame, analysis_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据数据摘要信息，自动为 DataFrame 生成一套标准的可视化图表。
    这是本模块的主调用函数。

    Args:
        df (pd.DataFrame): 需要可视化的数据。
        analysis_parameters (Dict[str, Any]): 从 get_data_summary 函数获得的详细数据摘要。

    Returns:
        Dict[str, Any]: 一个包含所有图表 Base64 字符串的字典，结构如下:
        {
            "histograms": {"col_name_1": "base64_str", ...},
            "barcharts": {"col_name_2": "base64_str", ...},
            "correlation_heatmap": "base64_str"
        }
    """
    visualizations = {
        "histograms": {},
        "barcharts": {},
        "correlation_heatmap": None
    }
    
    column_profiles = analysis_parameters.get("column_profiles", {})
    numerical_cols = []

    for col, profile in column_profiles.items():
        inferred_type = profile.get("inferred_type")
        # 根据推断的类型，调用不同的绘图函数
        if inferred_type == "continuous":
            visualizations["histograms"][col] = plot_numerical_distribution(df, col)
            numerical_cols.append(col)
        elif inferred_type in ["binary", "categorical_numeric", "binary_categorical", "categorical"]:
            # 对于唯一值过多的类别变量，不进行绘图，避免图表混乱
            if profile.get("unique_count", 0) < 50:
                visualizations["barcharts"][col] = plot_categorical_distribution(df, col)

    # 为所有识别出的数值型变量生成一张相关性热力图
    visualizations["correlation_heatmap"] = plot_correlation_heatmap(df, numerical_cols)


    return visualizations

if __name__ == '__main__':

    data = {
        'age': [25, 30, 35, 40, 25, 30, 45, 50, 22, 33],
        'city': ['北京', '伦敦', '巴黎', '北京', '伦敦', '东京', '巴黎', '伦敦', '北京', '东京'],
        'gender': ['男', '女', '女', '男', '男', '女', '男', '女', '男', '女'],
        'income': [50000, 60000, 75000, 80000, 52000, 65000, 90000, 120000, 48000, 70000],
        'subscribed': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    }
    verify_df = pd.DataFrame(data)

    # 2. 模拟一个 analysis_parameters 字典，这通常由 get_data_summary 生成
    mock_analysis_parameters = {
        'column_profiles': {
            'age': {'inferred_type': 'continuous'},
            'city': {'inferred_type': 'categorical', 'unique_count': 4},
            'gender': {'inferred_type': 'binary_categorical', 'unique_count': 2},
            'income': {'inferred_type': 'continuous'},
            'subscribed': {'inferred_type': 'binary', 'unique_count': 2}
        }
    }

    # 3. 调用主函数生成可视化结果
    visualizations = generate_visualizations(verify_df, mock_analysis_parameters)

    # 4. 打印验证结果
    print("--- 可视化模块简单验证 ---")
    print("已生成图表类型:", list(visualizations.keys()))

    if visualizations.get("histograms"):
        print("\n直方图:")
        for col, b64_str in visualizations["histograms"].items():
            print(f"  - 列 '{col}': Base64 编码长度 {len(b64_str)}")

    if visualizations.get("barcharts"):
        print("\n条形图:")
        for col, b64_str in visualizations["barcharts"].items():
            print(f"  - 列 '{col}': Base64 编码长度 {len(b64_str)}")

    if visualizations.get("correlation_heatmap"):
        print("\n相关性热力图:")
        # 检查字符串是否为空
        if visualizations['correlation_heatmap']:
            print(f"  - Base64 编码长度 {len(visualizations['correlation_heatmap'])}")
        else:
            print("  - 未生成 (数值列不足2个)")

    print("\n--- 验证完成 ---")
    # 如果需要，可以将其中一个 base64 字符串写入文件来查看图片
    with open("test_heatmap.html", "w") as f:
        f.write(f'<img src="data:image/png;base64,{visualizations["histograms"]["age"]}" />')
