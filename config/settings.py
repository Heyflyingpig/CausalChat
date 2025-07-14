import os
import json
import logging
import sys

# 计算项目根目录
# __file__ -> D:/.../CausalChat/config/settings.py
# os.path.dirname(__file__) -> D:/.../CausalChat/config
# os.path.dirname(os.path.dirname(__file__)) -> D:/.../CausalChat (项目根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")

class AppConfig:
    """
    一个用于加载、管理和验证应用配置的类。
    它会从 secrets.json 文件中读取敏感信息，并将其作为类的属性。
    同时，它也负责设置 LangSmith 等服务的环境变量。
    """
    def __init__(self, secrets_path=SECRETS_PATH):
        """
        初始化配置类。
        如果配置文件不存在或缺少关键信息，将抛出异常。
        """
        self.secrets_data = self._load_secrets(secrets_path)
        
        # --- 应用必需的配置 ---
        self.SECRET_KEY = self._get_required_key("SECRET_KEY")

        # --- AI 模型配置 ---
        self.API_KEY = self._get_required_key("API_KEY")
        self.BASE_URL = self._get_required_key("BASE_URL")
        self.MODEL = self._get_required_key("MODEL")

        # --- 数据库配置 ---
        self.MYSQL_HOST = self._get_required_key("MYSQL_HOST")
        self.MYSQL_USER = self._get_required_key("MYSQL_USER")
        self.MYSQL_PASSWORD = self._get_required_key("MYSQL_PASSWORD")
        self.MYSQL_DATABASE = self._get_required_key("MYSQL_DATABASE")
        
        # --- LangSmith (可选配置) ---
        self.LANGCHAIN_API_KEY = self.secrets_data.get("LANGCHAIN_API_KEY")
        self.LANGCHAIN_PROJECT = self.secrets_data.get("LANGCHAIN_PROJECT", "CausalChat-Default-Project")

        # 初始化完成后，自动设置 LangSmith
        self._setup_langsmith()

    def _load_secrets(self, path):
        """加载 JSON 格式的密钥文件。"""
        try:
            if not os.path.exists(path):
                logging.error(f"敏感信息配置文件 {path} 不存在。程序无法继续。")
                raise FileNotFoundError(f"必需的配置文件 {path} 未找到。")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"解析敏感信息配置文件 {path} 失败。请检查文件格式。")
            raise
        except Exception as e:
            logging.error(f"加载配置文件时发生未知错误: {e}")
            raise

    def _get_required_key(self, key):
        """获取必需的配置项，如果不存在则抛出异常。"""
        if key not in self.secrets_data:
            logging.error(f"关键配置 '{key}' 未在 {SECRETS_PATH} 中找到。")
            raise ValueError(f"配置错误: {SECRETS_PATH} 中缺少 '{key}'")
        return self.secrets_data[key]
        
    def _setup_langsmith(self):
        """根据配置设置 LangSmith 追踪的环境变量。"""
        if self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            logging.info(f"LangSmith 追踪已启用，项目名: '{self.LANGCHAIN_PROJECT}'")
        else:
            logging.warning(f"未在 {SECRETS_PATH} 中找到 'LANGCHAIN_API_KEY'。LangSmith 追踪将不会启用。")

# --- 单例模式：创建全局唯一的配置实例 ---
# 在应用启动时，尝试加载配置。
# 如果失败，settings 将为 None，依赖此配置的服务将无法启动。
settings = None
try:
    settings = AppConfig()
    logging.info(f"应用配置已从 {SECRETS_PATH} 成功加载。")
except (FileNotFoundError, ValueError) as e:
    logging.critical(f"配置文件加载失败，应用无法启动: {e}")
    # 让主程序决定如何处理这个致命错误
    # 在这里不调用 sys.exit()，只重新抛出异常
    raise 