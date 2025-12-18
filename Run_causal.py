import webview
import sys # 用于获取命令行参数 (可选)

if __name__ == '__main__':
 
    # 确保端口号与您 Gunicorn 服务绑定的端口一致 (例如 5000)
    remote_app_url = "http://127.0.0.1:5001"

   
    print(f"Attempting to load: {remote_app_url}")

    try:
        # 创建一个 webview 窗口来加载远程 URL
        window = webview.create_window(
            'CausalAgent',  # 窗口标题
            remote_app_url,
            width=1200,           # 窗口宽度
            height=800,           # 窗口高度
            resizable=True
        )
        webview.start(debug=True) # debug=True 会在某些后端提供开发者工具
    except Exception as e:
        print(f"An error occurred: {e}")
        # 对于 pywebview 的常见问题，可以提示用户检查依赖
        if "WebView2 runtime" in str(e) or "WebKitGTK" in str(e) or "QtWebEngine" in str(e):
            print("Please ensure you have the necessary WebView backend installed for your system.")
            print("See pywebview documentation for details: https://pywebview.flowrl.com/guide/installation.html")
