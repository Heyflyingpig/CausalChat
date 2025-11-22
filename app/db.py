"""
app.db - 数据库访问模块

"""
import mysql.connector
from mysql.connector import errorcode
import logging
from config.settings import settings

def get_db_connection():
    """创建并返回一个 MySQL 数据库连接。"""
    try:
        # 使用新的配置对象
        conn = mysql.connector.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE # 连接到指定的数据库
        )
        # logging.debug(f"成功连接到 MySQL 数据库 '{settings.MYSQL_DATABASE}'。")
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logging.error(f"MySQL 连接错误: 用户 '{settings.MYSQL_USER}' 或密码错误。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            logging.error(f"MySQL 连接错误: 数据库 '{settings.MYSQL_DATABASE}' 不存在。")
        else:
            logging.error(f"MySQL 连接错误: {err}")
        raise # 重新抛出异常，让调用者知道连接失败

def check_database_readiness():
    """检查数据库是否已准备就绪，包括所需的表和基本连接性。
    这个函数不会创建或修改数据库结构，只是检查现有配置是否正确。
    如果数据库未初始化，请先运行 database_init.py 脚本。
    """
    try:
        logging.info(f"检查数据库连接和表结构就绪状态...")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 检查新的、优化的表是否存在
            required_tables = ['users', 'sessions', 'chat_messages', 'chat_attachments', 'uploaded_files', 'archived_sessions','checkpoints','checkpoint_writes']
            cursor.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{settings.MYSQL_DATABASE}'
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                error_msg = f"数据库表缺失: {list(missing_tables)}。请先运行 'python database_init.py' 初始化数据库。"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 简单的连接性测试
            cursor.execute("SELECT 1")
            test_result = cursor.fetchone()
            if not test_result or test_result[0] != 1:
                raise RuntimeError("数据库连接测试失败")
            
            logging.info(f"数据库 '{settings.MYSQL_DATABASE}' 就绪检查通过。所有必需表已存在。")
            return True
            
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_BAD_DB_ERROR:
            error_msg = f"数据库 '{settings.MYSQL_DATABASE}' 不存在。请先运行 'python database_init.py' 创建和初始化数据库。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        elif e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            error_msg = f"无法访问数据库。请检查用户 '{settings.MYSQL_USER}' 的权限配置。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logging.error(f"数据库就绪性检查失败: {e}")
            raise
    except Exception as e:
        logging.error(f"数据库就绪性检查过程中发生未知错误: {e}")
        raise