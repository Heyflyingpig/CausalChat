# database_init.py - 优化的数据库初始化脚本
"""
优化的数据库初始化脚本
采用会话表分离、数据分层存储的设计，适合大规模数据场景
"""

import mysql.connector
from mysql.connector import errorcode
import json
import os
import logging
import sys
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('database_init.log', encoding='utf-8')
    ]
)

class OptimizedDatabaseInitializer:
    def __init__(self):
        """
        初始化优化的数据库初始化器
        
        配置加载方式：统一从环境变量加载
        - Docker环境：docker-compose传递的环境变量
        - 本地开发：.env文件（由python-dotenv加载到环境变量）
        """
        self.mysql_config = {}
        self.load_database_config()
    
    def load_database_config(self):
        """
        从环境变量加载数据库配置
        
        优先加载.env文件到环境变量（如果存在）
        然后统一从环境变量读取配置
        """
        # 先尝试加载.env文件（本地开发模式）
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            
            # 查找项目根目录的.env文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            env_path = Path(project_root) / '.env'
            
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                logging.info(f"从 {env_path} 加载环境变量（本地开发模式）")
        except ImportError:
            logging.info("未安装 python-dotenv，使用系统环境变量（Docker模式）")
        
        # 从环境变量读取配置
        required_keys = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
        
        self.mysql_config = {
            'host': os.environ.get('MYSQL_HOST'),
            'user': os.environ.get('MYSQL_USER'),
            'password': os.environ.get('MYSQL_PASSWORD'),
            'database': os.environ.get('MYSQL_DATABASE')
        }
        
        # 检查必需配置是否完整
        missing_keys = [k for k, v in zip(required_keys, self.mysql_config.values()) if not v]
        
        if missing_keys:
            error_msg = (
                f"配置错误: 缺少环境变量 {missing_keys}\n"
                f"请确保：\n"
                f"  - Docker环境：.env文件包含所有配置\n"
                f"  - 本地开发：项目根目录存在.env文件\n"
                f"当前配置：{self.mysql_config}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info(f"数据库配置已从环境变量成功加载: host={self.mysql_config['host']}, database={self.mysql_config['database']}")
    
    def create_database_if_not_exists(self):
        """创建数据库（如果不存在）"""
        try:
            # 使用root用户连接MySQL服务器（创建数据库需要权限）
            # Docker环境下，使用环境变量中的MYSQL_ROOT_PASSWORD
            root_password = os.environ.get('MYSQL_ROOT_PASSWORD') or self.mysql_config['password']
            
            logging.info(f"尝试连接到 MySQL 服务器: host={self.mysql_config['host']}, user=root")
            
            conn_server = mysql.connector.connect(
                host=self.mysql_config['host'],
                user='root',  # 使用root用户创建数据库
                password=root_password
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
                logging.info(f"开始创建优化的数据库表于 '{self.mysql_config['database']}'")
                
                # 创建用户表
                self._create_users_table(cursor)
                
                # 创建会话表（
                self._create_sessions_table(cursor)
                
                # 创建优化的聊天记录表
                self._create_optimized_chat_messages_table(cursor)
                
                # 创建聊天附件表（新增，用于存储大型结构化数据）
                self._create_chat_attachments_table(cursor)
                
                # 创建上传文件表
                self._create_uploaded_files_table(cursor)
                
                # 创建归档表（可选，用于数据生命周期管理）
                self._create_archived_sessions_table(cursor)
                
                # 创建优化的索引
                self._create_optimized_indexes(cursor)
                
                conn.commit()
                logging.info("所有优化的数据库表和索引已成功创建")
                
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login_at TIMESTAMP DEFAULT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                INDEX idx_username (username),
                INDEX idx_active_users (is_active, last_login_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("用户表 'users' 已检查/创建")
    
    def _create_sessions_table(self, cursor):
        """创建会话表，用于管理会话元数据"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(36) PRIMARY KEY COMMENT 'UUID格式的会话ID',
                user_id INT NOT NULL,
                title VARCHAR(500) DEFAULT NULL COMMENT '会话标题，可以是第一条消息的摘要',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                message_count INT DEFAULT 0 COMMENT '该会话的消息总数',
                is_archived BOOLEAN DEFAULT FALSE COMMENT '是否已归档',
                archived_at TIMESTAMP DEFAULT NULL,
                
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                
                INDEX idx_user_activity (user_id, last_activity_at DESC),
                INDEX idx_user_active (user_id, is_archived, last_activity_at DESC),
                INDEX idx_archive_cleanup (is_archived, archived_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("会话表 'sessions' 已检查/创建")
    
    def _create_optimized_chat_messages_table(self, cursor):
        """创建优化的聊天记录表"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGINT AUTO_INCREMENT,
                session_id VARCHAR(36) NOT NULL,
                user_id INT NOT NULL,
                message_type ENUM('user', 'ai') NOT NULL COMMENT '消息类型：用户或AI',
                content TEXT NOT NULL COMMENT '消息内容（纯文本或简单JSON）',
                has_attachment BOOLEAN DEFAULT FALSE COMMENT '是否有大型附件数据',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- 分区表的主键必须包含分区键(created_at)
                PRIMARY KEY (id, created_at),
                

                -- FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                -- FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                
                INDEX idx_session_time (session_id, created_at),
                INDEX idx_user_session (user_id, session_id, created_at),
                INDEX idx_message_type (message_type, created_at),
                INDEX idx_attachment_flag (has_attachment)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            PARTITION BY RANGE (UNIX_TIMESTAMP(created_at)) (
                PARTITION p_2024 VALUES LESS THAN (UNIX_TIMESTAMP('2025-01-01')),
                PARTITION p_2025_q1 VALUES LESS THAN (UNIX_TIMESTAMP('2025-04-01')),
                PARTITION p_2025_q2 VALUES LESS THAN (UNIX_TIMESTAMP('2025-07-01')),
                PARTITION p_2025_q3 VALUES LESS THAN (UNIX_TIMESTAMP('2025-10-01')),
                PARTITION p_2025_q4 VALUES LESS THAN (UNIX_TIMESTAMP('2026-01-01')),
                PARTITION p_future VALUES LESS THAN MAXVALUE
            )
        """)
        logging.info("优化的聊天记录表 'chat_messages' 已检查/创建（包含分区，复合主键，无外键约束）")
    
    def _create_chat_attachments_table(self, cursor):
        """创建聊天附件表 - 用于存储大型结构化数据"""
        # 注意：由于chat_messages是分区表，不能创建外键约束
        # 注意，attachment的附件多了visualization，具体请看almebic
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_attachments (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                message_id BIGINT NOT NULL,
                attachment_type ENUM('causal_graph', 'analysis_result', 'file_content', 'other') NOT NULL,
                content LONGTEXT NOT NULL COMMENT '大型JSON数据或其他结构化内容',
                content_size INT GENERATED ALWAYS AS (LENGTH(content)) STORED COMMENT '内容大小，用于监控',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
                
                INDEX idx_message_attachment (message_id, attachment_type),
                INDEX idx_type_size (attachment_type, content_size),
                INDEX idx_size_cleanup (content_size, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("聊天附件表 'chat_attachments' 已检查/创建（无外键约束）")
    
    def _create_uploaded_files_table(self, cursor):
        """创建优化的上传文件表"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL COMMENT '用户上传时的原始文件名',
                mime_type VARCHAR(100) NOT NULL,
                file_size BIGINT NOT NULL COMMENT '文件大小（字节）',
                file_hash VARCHAR(64) NOT NULL COMMENT 'SHA-256哈希，用于去重',
                file_content LONGBLOB NOT NULL,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                access_count INT DEFAULT 0 COMMENT '访问次数',
                
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                
                UNIQUE KEY unique_user_hash (user_id, file_hash) COMMENT '同一用户不能上传相同内容的文件',
                INDEX idx_user_files (user_id, upload_timestamp DESC),
                INDEX idx_filename_search (user_id, filename),
                INDEX idx_size_cleanup (file_size, last_accessed_at),
                INDEX idx_hash_dedup (file_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("优化的上传文件表 'uploaded_files' 已检查/创建")
    
    def _create_archived_sessions_table(self, cursor):
        """创建归档会话表 - 用于数据生命周期管理"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_sessions (
                id VARCHAR(36) PRIMARY KEY,
                user_id INT NOT NULL,
                original_session_data JSON NOT NULL COMMENT '原始会话的元数据',
                message_count INT NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                archive_reason ENUM('user_request', 'auto_cleanup', 'admin_action') DEFAULT 'auto_cleanup',
                compressed_data LONGBLOB COMMENT '压缩后的会话数据',
                
                INDEX idx_user_archived (user_id, archived_at),
                INDEX idx_cleanup_schedule (archive_reason, archived_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logging.info("归档会话表 'archived_sessions' 已检查/创建")
    
    def _create_optimized_indexes(self, cursor):
        """创建优化的索引"""
        # 复合索引已在表创建时定义，这里添加一些额外的性能索引
        additional_indexes = [
            {
                'name': 'idx_active_sessions_by_user',
                'table': 'sessions',
                'columns': '(user_id, is_archived, last_activity_at DESC)'
            },
            {
                'name': 'idx_recent_messages',
                'table': 'chat_messages', 
                'columns': '(created_at DESC, user_id, session_id)'
            },
            {
                'name': 'idx_large_attachments',
                'table': 'chat_attachments',
                'columns': '(content_size DESC, created_at)'
            }
        ]
        
        for index in additional_indexes:
            try:
                cursor.execute(f"""
                    CREATE INDEX {index['name']} 
                    ON {index['table']} {index['columns']}
                """)
                logging.info(f"已创建额外索引 {index['name']}")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DUP_KEYNAME:
                    logging.info(f"索引 {index['name']} 已存在")
                else:
                    logging.warning(f"创建索引 {index['name']} 时发生错误: {err}")
    
    def create_maintenance_procedures(self):
        """创建数据库维护存储过程"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 会话清理存储过程
                cursor.execute("""
                    DROP PROCEDURE IF EXISTS ArchiveOldSessions
                """)
                
                cursor.execute("""
                    CREATE PROCEDURE ArchiveOldSessions(IN days_old INT)
                    BEGIN
                        DECLARE done INT DEFAULT FALSE;
                        DECLARE session_id VARCHAR(36);
                        DECLARE session_cursor CURSOR FOR 
                            SELECT s.id FROM sessions s 
                            WHERE s.last_activity_at < DATE_SUB(NOW(), INTERVAL IFNULL(days_old, 90) DAY)
                            AND s.is_archived = FALSE;
                        DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
                        
                        START TRANSACTION;
                        
                        OPEN session_cursor;
                        read_loop: LOOP
                            FETCH session_cursor INTO session_id;
                            IF done THEN
                                LEAVE read_loop;
                            END IF;
                            
                            -- 标记会话为已归档
                            UPDATE sessions SET is_archived = TRUE, archived_at = NOW() 
                            WHERE id = session_id;
                            
                        END LOOP;
                        CLOSE session_cursor;
                        
                        COMMIT;
                        
                        SELECT ROW_COUNT() as archived_sessions_count;
                    END
                """)
                
                # 数据完整性检查存储过程（补偿没有外键约束）
                cursor.execute("""
                    DROP PROCEDURE IF EXISTS CheckDataIntegrity
                """)
                
                cursor.execute("""
                    CREATE PROCEDURE CheckDataIntegrity()
                    BEGIN
                        DECLARE orphaned_messages INT DEFAULT 0;
                        DECLARE orphaned_attachments INT DEFAULT 0;
                        
                        -- 检查孤立的聊天消息（session_id不存在）
                        SELECT COUNT(*) INTO orphaned_messages
                        FROM chat_messages cm
                        LEFT JOIN sessions s ON cm.session_id = s.id
                        WHERE s.id IS NULL;
                        
                        -- 检查孤立的附件（message_id不存在）
                        SELECT COUNT(*) INTO orphaned_attachments
                        FROM chat_attachments ca
                        LEFT JOIN chat_messages cm ON ca.message_id = cm.id
                        WHERE cm.id IS NULL;
                        
                        -- 返回检查结果
                        SELECT 
                            orphaned_messages as orphaned_messages_count,
                            orphaned_attachments as orphaned_attachments_count,
                            CASE 
                                WHEN orphaned_messages = 0 AND orphaned_attachments = 0 
                                THEN 'PASS' 
                                ELSE 'FAIL' 
                            END as integrity_status;
                    END
                """)
                
                logging.info("数据库维护存储过程和完整性检查已创建")
                conn.commit()
                
        except mysql.connector.Error as e:
            logging.error(f"创建维护存储过程时发生错误: {e}")
            raise
    
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
    
    def initialize_all(self, create_procedures=True):
        """执行完整的优化数据库初始化流程"""
        try:
            logging.info("开始优化的数据库初始化流程...")
            
            # 1. 创建数据库
            self.create_database_if_not_exists()
            
            # 2. 创建表和索引
            self.create_tables()
            
            # 3. 创建维护存储过程
            if create_procedures:
                self.create_maintenance_procedures()
            
            # 4. 检查连接
            if self.check_database_connection():
                logging.info("优化的数据库初始化完成！")
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
    
    try:
        # 创建初始化器实例
        db_init = OptimizedDatabaseInitializer()
        
        # 询问是否创建维护存储过程
        create_procedures = True
        user_input = input("\n是否创建数据库维护存储过程？(Y/n): ").strip().lower()
        if user_input in ['n', 'no']:
            create_procedures = False
        
        # 执行初始化
        success = db_init.initialize_all(create_procedures=create_procedures)
        
        if success:
            print("\n数据库初始化成功完成！")
            print("现在可以启动主应用程序了。")
        else:
            print("\n数据库初始化失败！")
            print("请检查日志文件 database_init.log 获取详细信息。")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断了初始化过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n初始化过程中发生错误: {e}")
        print("请检查日志文件 database_init.log 获取详细信息。")
        sys.exit(1)


if __name__ == "__main__":
    main() 