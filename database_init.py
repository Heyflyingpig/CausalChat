# database_init.py - 独立的数据库初始化脚本
"""
数据库初始化脚本
用于创建和初始化 CausalChat 项目所需的 MySQL 数据库和表结构
"""

import mysql.connector
from mysql.connector import errorcode
import json
import os
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('database_init.log', encoding='utf-8')
    ]
)

class DatabaseInitializer:
    def __init__(self, secrets_path="secrets.json"):
        """
        初始化数据库初始化器
        
        Args:
            secrets_path (str): secrets.json 文件路径
        """
        self.secrets_path = secrets_path
        self.mysql_config = {}
        self.load_database_config()
    
    def load_database_config(self):
        """从 secrets.json 加载数据库配置"""
        required_keys = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
        
        try:
            if not os.path.exists(self.secrets_path):
                raise FileNotFoundError(f"配置文件 {self.secrets_path} 不存在")
            
            with open(self.secrets_path, "r", encoding="utf-8") as f:
                secrets_data = json.load(f)
            
            # 检查必需的数据库配置
            for key in required_keys:
                if key not in secrets_data:
                    raise ValueError(f"配置错误: {self.secrets_path} 中缺少 '{key}'")
            
            self.mysql_config = {
                'host': secrets_data["MYSQL_HOST"],
                'user': secrets_data["MYSQL_USER"],
                'password': secrets_data["MYSQL_PASSWORD"],
                'database': secrets_data["MYSQL_DATABASE"]
            }
            
            logging.info(f"数据库配置已从 {self.secrets_path} 成功加载")
            
        except json.JSONDecodeError as e:
            logging.error(f"解析配置文件 {self.secrets_path} 失败: {e}")
            raise
        except Exception as e:
            logging.error(f"加载数据库配置时发生错误: {e}")
            raise
    
    def create_database_if_not_exists(self):
        """创建数据库（如果不存在）"""
        try:
            # 连接到 MySQL 服务器（不指定数据库）
            logging.info(f"尝试连接到 MySQL 服务器: host={self.mysql_config['host']}, user={self.mysql_config['user']}")
            
            conn_server = mysql.connector.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user'],
                password=self.mysql_config['password']
            )
            
            cursor_server = conn_server.cursor()
            
            # 创建数据库
            database_name = self.mysql_config['database']
            logging.info(f"尝试创建数据库 '{database_name}' (如果不存在)...")
            
            cursor_server.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            
            cursor_server.close()
            conn_server.close()
            
            logging.info(f"数据库 '{database_name}' 已确保存在")
            return True
            
        except mysql.connector.Error as e:
            logging.error(f"创建数据库失败: {e}")
            if e.errno == errorcode.ER_DBACCESS_DENIED_ERROR or e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logging.error(f"用户 '{self.mysql_config['user']}' 没有创建数据库的权限")
            raise
        except Exception as e:
            logging.error(f"创建数据库时发生未知错误: {e}")
            raise
    
    def get_db_connection(self):
        """获取数据库连接"""
        try:
            conn = mysql.connector.connect(**self.mysql_config)
            return conn
        except mysql.connector.Error as e:
            logging.error(f"连接数据库失败: {e}")
            raise
    
    def create_tables(self):
        """创建所有必需的表"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                logging.info(f"开始创建数据库表于 '{self.mysql_config['database']}'")
                
                # 创建用户表
                self._create_users_table(cursor)
                
                # 创建聊天记录表
                self._create_chat_messages_table(cursor)
                
                # 创建上传文件表
                self._create_uploaded_files_table(cursor)
                
                # 创建索引
                self._create_indexes(cursor)
                
                conn.commit()
                logging.info("所有数据库表和索引已成功创建")
                
        except mysql.connector.Error as e:
            logging.error(f"创建表时发生数据库错误: {e}")
            raise
        except Exception as e:
            logging.error(f"创建表时发生未知错误: {e}")
            raise
    
    def _create_users_table(self, cursor):
        """创建用户表"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("用户表 'users' 已检查/创建")
    
    def _create_chat_messages_table(self, cursor):
        """创建聊天记录表"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                user_id INT NOT NULL,
                user_msg TEXT,
                ai_msg TEXT,
                ai_msg_structured JSON DEFAULT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("聊天记录表 'chat_messages' 已检查/创建")
    
    def _create_uploaded_files_table(self, cursor):
        """创建上传文件表"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                mime_type VARCHAR(100) NOT NULL,
                file_content LONGBLOB NOT NULL,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("上传文件表 'uploaded_files' 已检查/创建")
    
    def _create_indexes(self, cursor):
        """创建索引"""
        indexes = [
            {
                'name': 'idx_chat_messages_session_userid_time',
                'table': 'chat_messages',
                'columns': '(session_id, user_id, timestamp)'
            },
            {
                'name': 'idx_chat_messages_userid_session_time',
                'table': 'chat_messages', 
                'columns': '(user_id, session_id, timestamp DESC)'
            },
            {
                'name': 'idx_uploaded_files_user_id',
                'table': 'uploaded_files',
                'columns': '(user_id)'
            }
        ]
        
        for index in indexes:
            try:
                cursor.execute(f"""
                    CREATE INDEX {index['name']} 
                    ON {index['table']} {index['columns']}
                """)
                logging.info(f"已创建索引 {index['name']}")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DUP_KEYNAME:
                    logging.info(f"索引 {index['name']} 已存在")
                else:
                    logging.warning(f"创建索引 {index['name']} 时发生错误: {err}")
    
    def check_database_connection(self):
        """检查数据库连接是否正常"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result[0] == 1:
                    logging.info("数据库连接检查通过")
                    return True
                else:
                    logging.error("数据库连接检查失败")
                    return False
        except Exception as e:
            logging.error(f"数据库连接检查失败: {e}")
            return False
    
    def initialize_all(self):
        """执行完整的数据库初始化流程"""
        try:
            logging.info("开始数据库初始化流程...")
            
            # 1. 创建数据库
            self.create_database_if_not_exists()
            
            # 2. 创建表和索引
            self.create_tables()
            
            # 3. 检查连接
            if self.check_database_connection():
                logging.info("数据库初始化完成！")
                return True
            else:
                logging.error("数据库初始化完成但连接检查失败")
                return False
                
        except Exception as e:
            logging.error(f"数据库初始化失败: {e}")
            return False


def main():
    """主函数 - 命令行入口"""
    print("CausalChat 数据库初始化工具")
    print("=" * 40)
    
    try:
        # 创建初始化器实例
        db_init = DatabaseInitializer()
        
        # 执行初始化
        success = db_init.initialize_all()
        
        if success:
            print("\n✅ 数据库初始化成功完成！")
            print("现在可以启动主应用程序了。")
        else:
            print("\n❌ 数据库初始化失败！")
            print("请检查日志文件 database_init.log 获取详细信息。")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断了初始化过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 初始化过程中发生错误: {e}")
        print("请检查日志文件 database_init.log 获取详细信息。")
        sys.exit(1)


if __name__ == "__main__":
    main() 