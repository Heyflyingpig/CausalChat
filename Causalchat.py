# app.py (Flask后端)
from flask import Flask, jsonify, request, send_from_directory
import os
import logging
import json 
import asyncio
import atexit
import sys
from flask import session
import threading
from app.agent import core as agent_core
from Agent.causal_agent.graph import create_graph
from app import create_app
try:
    from config.settings import settings
except (ValueError, FileNotFoundError) as e:
    logging.critical(f"无法加载应用配置，程序终止。错误: {e}")
    sys.exit(1)
# 在 Windows 上，默认的 asyncio 事件循环 (SelectorEventLoop) 不支持子进程。
# MCP 客户端需要通过子进程启动服务器，因此我们必须切换到 ProactorEventLoop。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 再次获取根logger并显式设置级别，以防被其他库覆盖
logging.getLogger().setLevel(logging.INFO)

app = create_app()

# 主程序入口
if __name__ == '__main__':
    # 注册应用退出时的清理函数
    atexit.register(agent_core.shutdown_mcp_connection)

    # 1. 初始化 LLM
    if not agent_core.initialize_llm():
        logging.critical("LLM 初始化失败，应用无法启动。")
        sys.exit(1)

    # 2. 初始化 RAG 系统
    if not agent_core.initialize_rag_system():
        # RAG 不是致命错误，允许在没有知识库的情况下继续运行
        logging.warning("RAG 系统初始化失败。应用将以无知识库模式运行。")
    
    # 3. 启动后台事件循环并等待 MCP 就绪
    mcp_ready_event = threading.Event()
    
    ## 后台线程
    background_event_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=agent_core.start_event_loop, 
        args=(background_event_loop, mcp_ready_event),
        daemon=True
    )
    loop_thread.start()
    
    logging.info("主线程正在等待 MCP 初始化...")
    is_ready = mcp_ready_event.wait(timeout=30.0)

    if not is_ready:
        logging.critical("MCP 服务在30秒内未能完成初始化。应用即将退出。")
        if agent_core.background_loop and agent_core.background_loop.is_running():
            asyncio.run_coroutine_threadsafe(agent_core.mcp_process_stack.aclose(), agent_core.background_loop)
        sys.exit(1)

    if not agent_core.mcp_session:
        logging.critical("MCP 初始化完成但会话无效。应用即将退出。")
        sys.exit(1)

    logging.info("正在根据 LLM 和 MCP 会话创建 Agent Graph...")
    
    agent_core.agent_graph = create_graph(agent_core.llm, agent_core.mcp_session)
    logging.info("Agent Graph 创建成功。")
    logging.info("MCP 连接就绪，启动 Flask 应用服务器...")
    # 启动 Flask 应用
    # use_reloader=False 是必须的，因为重载器会启动一个子进程，
    # 这会干扰我们已经手动管理的 MCP 子进程和事件循环。
    # 
    # Docker环境注意：
    # - host='0.0.0.0' 监听所有网络接口，允许容器外部访问
    # - host='127.0.0.1' 只允许容器内部访问，Docker端口映射会失效
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

    