from flask import Flask
from config.settings import settings
import logging
import sys
import os
from app.db import check_database_readiness
from app.auth.routes import auth_bp
from app.chat.routes import chat_bp
from app.files.routes import files_bp
from app.agent.routes import agent_bp
from app.main.routes import main_bp


def create_app():
    app = Flask(__name__, static_folder="static")
    app.secret_key = settings.SECRET_KEY

    try:
        check_database_readiness()
    except Exception as e:
        logging.critical(f"数据库检查失败，应用无法启动: {e}")
        print(f"数据库检查失败: {e}")
        raise e
        sys.exit(1)

    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(main_bp)
    return app