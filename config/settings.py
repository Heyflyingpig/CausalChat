import os
import logging
import sys
from pathlib import Path

# 计算项目根目录
# __file__ -> D:/.../CausalChat/config/settings.py
# os.path.dirname(__file__) -> D:/.../CausalChat/config
# os.path.dirname(os.path.dirname(__file__)) -> D:/.../CausalChat (项目根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 文件（如果存在）
# 这个库会自动读取项目根目录的 .env 文件，并将其中的变量加载到环境变量中
try:
    from dotenv import load_dotenv
    env_path = Path(BASE_DIR) / '.env'
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logging.info(f"从 {env_path} 加载环境变量")
    else:
        logging.info("未找到 .env 文件，直接使用系统环境变量")
except ImportError:
    # 如果没有安装 python-dotenv，只使用系统环境变量（Docker环境下正常）
    logging.info("未安装 python-dotenv，使用系统环境变量")

class AppConfig:
    """
    一个用于加载、管理和验证应用配置的类。
    
    配置来源（统一使用环境变量）：
    - Docker环境：通过 docker-compose 的 environment 或 env_file 传递
    - 本地开发：通过项目根目录的 .env 文件（由 python-dotenv 自动加载）
    
    """
    def __init__(self):
        """
        初始化配置类。
        所有配置都从环境变量加载（本地开发通过.env文件自动加载到环境变量）
        """
        
        # 应用必需的配置
        self.SECRET_KEY = self._get_config("SECRET_KEY")

        # AI 模型配置
        self.API_KEY = self._get_config("API_KEY")
        self.BASE_URL = self._get_config("BASE_URL")
        self.MODEL = self._get_config("MODEL")

        # 数据库配置
        self.MYSQL_HOST = self._get_config("MYSQL_HOST")
        self.MYSQL_USER = self._get_config("MYSQL_USER")
        self.MYSQL_PASSWORD = self._get_config("MYSQL_PASSWORD")
        self.MYSQL_DATABASE = self._get_config("MYSQL_DATABASE")
        
        # LangSmith 
        # 可选配置不强制要求，缺失时使用默认值
        self.LANGCHAIN_API_KEY = self._get_config("LANGCHAIN_API_KEY", required=False)
        self.LANGCHAIN_PROJECT = self._get_config(
            "LANGCHAIN_PROJECT", 
            required=False, 
            default="CausalChat-Default-Project"
        )

        # 初始化完成后，自动设置 LangSmith
        self._setup_langsmith()

    def _get_config(self, key, required=True, default=None):
        """
        从环境变量获取配置项。
        
        参数：
            key (str): 配置项名称（环境变量名）
            required (bool): 是否为必需项，默认True
            default: 默认值（仅当required=False时有效）
        
        返回：
            配置值（字符串）或默认值
        
        异常：
            ValueError: 当必需项缺失时抛出
        
        """
        # 从环境变量获取配置值
        value = os.environ.get(key)
        
        # 如果获取到值，直接返回
        if value:
            return value
        
        # 如果是必需项且未找到，抛出异常
        if required:
            error_msg = (
                f"配置错误: 缺少必需的环境变量 '{key}'。\n"
                f"请确保：\n"
                f"  - Docker环境：在项目根目录的 .env 文件中设置 {key}=...\n"
                f"  - 本地开发：在项目根目录的 .env 文件中设置 {key}=...\n"
                f"  - 或直接设置系统环境变量\n"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # 可选项未找到，返回默认值
        return default
        
    def _setup_langsmith(self):
        """
        根据配置设置 LangSmith 追踪的环境变量。
        
        LangSmith 是可选功能，用于追踪和调试 LangChain 应用。
        如果未配置 LANGCHAIN_API_KEY，应用仍可正常运行，只是不会有追踪功能。
        """
        if self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            logging.info(f"LangSmith 追踪已启用，项目名: '{self.LANGCHAIN_PROJECT}'")
        else:
            logging.warning("未找到 'LANGCHAIN_API_KEY' 环境变量。LangSmith 追踪将不会启用。")

#  单例模式：创建全局唯一的配置实例 
# 在应用启动时，尝试加载配置。
# 如果失败，settings 将为 None，依赖此配置的服务将无法启动。
settings = None
try:
    settings = AppConfig()
    logging.info("应用配置已从环境变量成功加载。")
except (FileNotFoundError, ValueError) as e:
    logging.critical(f"配置加载失败，应用无法启动: {e}")

    raise 