'''
agent.routes - agent路由
'''
from flask import Blueprint, request, jsonify, session
import logging
import asyncio
from app.agent import core as agent_core
from app.chat.services import save_chat
from app.agent.core import ai_call_stream
import json

agent_bp = Blueprint('agent', __name__, url_prefix='/api')

@agent_bp.route('/send_stream', methods=['POST'])
def handle_message_stream():
    """
    SSE流式端点：实时推送agent节点执行事件
    """
    from flask import session, stream_with_context, Response
    
    # 认证检查
    if 'user_id' not in session or 'username' not in session:
        return jsonify({'success': False, 'error': '用户未登录或会话已过期'}), 401
    
    user_id = session['user_id']
    username = session['username']
    
    # 获取请求参数
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id')
    
    if not session_id:
        logging.error(f"用户 {username} (ID: {user_id}) 发送流式消息时缺少 session_id")
        return jsonify({'success': False, 'error': '请求无效，缺少会话ID'}), 400
    
    logging.info(f"[流式] 用户 {username} (ID: {user_id}) 在会话 {session_id} 中发送消息: {user_input[:50]}...")
    
    def generate():
        """
        生成器函数，用于SSE流式传输
        """
        try:
            # 在后台事件循环中运行异步生成器
            loop = agent_core.background_loop
            queue = asyncio.Queue()
            
            async def producer():
                try:
                    async for event_data in ai_call_stream(user_input, user_id, username, session_id):
                        # 放入队列
                        await queue.put(event_data)
                except Exception as e:
                    logging.error(f"[SSE] 生成器错误: {e}", exc_info=True)
                    error_event = f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                    await queue.put(error_event)
                finally:
                    await queue.put(None)  # 结束标记
            
            # 启动生产者
            future = asyncio.run_coroutine_threadsafe(producer(), loop)
            
            # 从队列中读取并yield
            while True:
                # 使用 run_coroutine_threadsafe 获取队列中的数据
                get_future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
                event_data = get_future.result(timeout=600)  # 5分钟超时
                
                if event_data is None:  # 结束标记
                    break
                    
                yield event_data
                
                # 如果是final_result或interrupt，保存聊天记录
                try:
                    if event_data.startswith("data: "):
                        event_json = json.loads(event_data[6:])  # 去掉 "data: " 前缀
                        
                        if event_json.get("type") == "final_result":
                            # 保存聊天记录
                            response_data = event_json.get("data", {})
                            save_chat(user_id, session_id, user_input, response_data)
                            logging.info(f"[SSE] 已保存聊天记录到数据库")
                        
                        elif event_json.get("type") == "interrupt":
                            # interrupt场景也保存
                            response_data = {
                                "type": "human_input_required",
                                "summary": event_json.get("message", "")
                            }
                            save_chat(user_id, session_id, user_input, response_data)
                            logging.info(f"[SSE] 已保存interrupt聊天记录")
                except Exception as e:
                    logging.warning(f"[SSE] 保存聊天记录时出错: {e}")
            
            # 等待生产者完成
            future.result(timeout=5)
            
        except Exception as e:
            logging.error(f"[SSE] 流式传输错误: {e}", exc_info=True)
            error_event = f"data: {json.dumps({'type': 'error', 'message': f'流式传输错误: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield error_event
    
    # 返回SSE响应
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )