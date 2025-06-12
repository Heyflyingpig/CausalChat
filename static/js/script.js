// --- 新增：认证相关变量和函数 ---
const authOverlay = document.getElementById('authOverlay');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const loginError = document.getElementById('loginError');
const registerError = document.getElementById('registerError'); // 错误
const userAvatar = document.getElementById('userAvatar'); //头像
const userInfoPopup = document.getElementById('userInfoPopup');
const userInfoContent = document.getElementById('userInfoContent');
const historyList = document.getElementById('historyList'); // 获取 historyList 元素
const settingPopup = document.getElementById('settingPopup'); // 获取设置
const settingOptions = document.getElementById('settingOptions'); // 新增：获取设置选项容器
const settingContentDisplay = document.getElementById('settingContentDisplay'); // 获取内容显示区域
const backToSettingsButton = document.getElementById('backToSettingsButton'); // 获取返回按钮
const csvUploaderInput = document.getElementById('csvUploader'); // 获取CSV上传器
const uploadCsvButton = document.getElementById('uploadCsvButton'); // 获取上传按钮
const chatArea = document.getElementById('chatArea');
// --- 新增：全局变量存储当前会话的用户名 ---
let currentUsername = null;

// 切换登录和注册表单
function toggleAuthForms() {
    loginError.textContent = ''; // 清除错误信息
    registerError.textContent = '';
    if (loginForm.style.display === 'none') {
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
    } else {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
    }
}

// 密码哈希函数 (使用 Web Crypto API - 异步)
async function hashPassword(password) {
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data); // 使用 SHA-256
    const hashArray = Array.from(new Uint8Array(hashBuffer)); // 转换为字节数组
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join(''); // 转换为十六进制字符串
    return hashHex;
}

// 处理注册
async function handleRegister() {
    const username = document.getElementById('registerUsername').value.trim();
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    registerError.textContent = ''; // 清空之前的错误

    if (!username || !password || !confirmPassword) {
        registerError.textContent = '所有字段均为必填项。';
        return;
    }
    if (password.length < 6) { // 添加密码长度检查
         registerError.textContent = '密码至少需要6位。';
         return;
    }
    if (password !== confirmPassword) {
        registerError.textContent = '两次输入的密码不匹配。';
        return;
    }

    try {
        const hashedPassword = await hashPassword(password); // 哈希密码

        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ username: username, password: hashedPassword }) // 发送哈希后的密码
        });
        const data = await response.json();

        if (data.success) {
            alert('注册成功！请登录。');
            toggleAuthForms(); // 切换回登录表单
             // 清空注册表单
            document.getElementById('registerUsername').value = '';
            document.getElementById('registerPassword').value = '';
            document.getElementById('confirmPassword').value = '';
        } else {
            registerError.textContent = data.error || '注册失败，请稍后再试。';
        }

    } catch (error) {
        console.error("Register error:", error);
        registerError.textContent = '注册过程中发生错误。';
    }
}

// 处理登录 - **修改**
async function handleLogin() {
    const usernameInput = document.getElementById('loginUsername'); // 获取输入元素
    const passwordInput = document.getElementById('loginPassword'); // 获取输入元素
    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    loginError.textContent = '';

    if (!username || !password) {
        loginError.textContent = '请输入用户名和密码。';
        return;
    }

    try {
        const hashedPassword = await hashPassword(password); // 哈希密码

        const response = await fetch('/api/login', {
             method: 'POST',
             headers: {'Content-Type': 'application/json'},
             body: JSON.stringify({ username: username, password: hashedPassword })
        });
        const data = await response.json();

        if (data.success && data.username) { // 确保返回了 username
            // 登录成功
            currentUsername = data.username; // **修改**: 设置全局变量
            document.body.classList.add('logged-in'); // 添加标记类
            authOverlay.classList.remove('active'); // 隐藏登录/注册层
            updateUserInfo(); // 更新用户信息显示
            loadHistory(); // 登录成功后加载历史记录
             // 清空登录表单
            usernameInput.value = '';
            passwordInput.value = '';
        } else {
            loginError.textContent = data.error || '登录失败，请检查用户名和密码。';
            currentUsername = null; // **修改**: 确保登录失败时全局变量为空
        }
    } catch (error) {
        console.error("Login error:", error);
        loginError.textContent = '登录过程中发生错误。';
        currentUsername = null; // **修改**: 确保出错时全局变量为空
    }
}

// 处理退出登录 - **修改**
async function handleLogout() {
    const username = currentUsername; // **修改**: 使用全局变量获取当前用户 (主要用于日志)
    if (!username) return; // 如果没有当前用户，直接返回

    console.log(`用户 ${username} 正在请求退出登录`);

    try {
         // --- 新增：调用后端登出接口 ---
        const response = await fetch('/api/logout', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            console.log("后端登出成功");
            currentUsername = null; // **修改**: 清除全局变量
            document.body.classList.remove('logged-in'); // 移除标记类
            authOverlay.classList.add('active'); // 显示登录/注册层
            loginForm.style.display = 'block'; // 确保显示的是登录表单
            registerForm.style.display = 'none';
            closeUserInfoPopup(); // 关闭用户信息弹窗
            document.getElementById('chatArea').innerHTML = ''; // 清空聊天区域
            historyList.innerHTML = ''; // 清空历史列表
            updateUserInfo(); // 清空头像等信息
            console.log("用户已退出登录，UI已更新");
        } else {
            showError("退出登录失败，请稍后再试。");
            console.error("后端登出失败:", data.error);
        }
    } catch (error) {
         showError("退出登录时发生网络错误。");
         console.error("退出登录错误:", error);
    }
}

async function checkLoginStatus() {
    console.log("检查后端认证状态...");
    try {
        const response = await fetch('/api/check_auth'); // 调用新接口
        const data = await response.json();

        if (data.isLoggedIn && data.username) {
            console.log(`用户 ${data.username} 已通过后端验证，加载主界面`);
            currentUsername = data.username; // **修改**: 设置全局变量
            document.body.classList.add('logged-in');
            authOverlay.classList.remove('active');
            updateUserInfo(); // 更新用户信息显示 (稍后修改此函数)
            loadHistory(); // 加载历史记录 (稍后修改此函数)
        } else {
            console.log("后端验证：用户未登录，显示登录界面");
            currentUsername = null; // **修改**: 确保全局变量为空
            document.body.classList.remove('logged-in');
            authOverlay.classList.add('active');
            loginForm.style.display = 'block';
            registerForm.style.display = 'none';
            historyList.innerHTML = ''; // 清空可能存在的旧历史记录
            updateUserInfo(); // 清空头像等 (稍后修改此函数)
        }
    } catch (error) {
        console.error("检查认证状态时出错:", error);
        // 网络错误等，也显示登录界面
        currentUsername = null;
        document.body.classList.remove('logged-in');
        authOverlay.classList.add('active');
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
        historyList.innerHTML = '<p style="padding: 10px; color: red;">无法连接服务器检查状态</p>';
        showError('无法连接服务器检查登录状态。'); // 可以显示错误提示
    }
}

// 更新用户界面信息（例如头像区域） - 
function updateUserInfo() {
    if (currentUsername) { // : 使用全局变量
        userAvatar.textContent = currentUsername.charAt(0).toUpperCase(); // 显示用户名首字母
        userInfoContent.textContent = `账号: ${currentUsername}`; // 设置弹窗内容
    } else {
        userAvatar.textContent = ''; // 未登录则清空
        userInfoContent.textContent = ''; // 清空弹窗内容
    }
}

// 显示用户信息弹窗 - **修改**
function showUserInfoPopup() {
     if (!currentUsername) return; // **修改**: 使用全局变量
    userInfoPopup.classList.add('active');
}

// 关闭用户信息弹窗
function closeUserInfoPopup() {
    userInfoPopup.classList.remove('active');
}

// 设置
function showSettingPopup() {
    // 这里可以加一个登录检查，如果需要的话
    if (!currentUsername) {
         showError("请先登录！");
         return;
     }
    console.log("打开设置弹窗");
    // 重置到初始状态
    settingOptions.style.display = 'block';
    settingContentDisplay.style.display = 'none'; // 确保内容区隐藏
    settingContentDisplay.innerHTML = ''; // 清空旧内容
    backToSettingsButton.style.display = 'none'; // 新增：确保返回按钮隐藏
    settingPopup.classList.add('active'); // 添加 active 类来显示弹窗（并触发动画）
}

//隐藏设置
function hideSettingPopup() {
    console.log("关闭设置弹窗");
    settingPopup.classList.remove('active'); 
    setTimeout(() => {
         settingOptions.style.display = 'block';
         settingContentDisplay.style.display = 'none';
         settingContentDisplay.innerHTML = '';
         backToSettingsButton.style.display = 'none';
     }, 300); // 300ms 匹配 CSS 过渡时间
}

//设置按钮点击处理
async function handleSettingOption(optionId) {
    console.log(`点击了设置选项: ${optionId}`);

    settingOptions.style.display = 'none'; // 修正：隐藏选项列表容器
    settingContentDisplay.style.display = 'block'; // 显示内容区域
    backToSettingsButton.style.display = 'inline-block'; // 显示返回按钮

    settingContentDisplay.innerHTML = '<p>正在加载内容...</p>'; // 显示加载提示

    if (optionId === 'checkUpdate') {
    settingContentDisplay.innerHTML = '<p>版本已经更新到最新</p>'; // 显示提示信息
    return; // 直接结束函数，不执行后续的 fetch
}
    try {
        const response = await fetch(`/api/setting?topic=${encodeURIComponent(optionId)}`); // 不需要区分选项，直接调用
        const data = await response.json();

        if (data.success && data.messages) {
            console.log("成功获取设置内容");
            
            settingContentDisplay.textContent = data.messages;

        } else {
            console.error("获取设置内容失败:", data.error);
            // 显示错误信息
            settingContentDisplay.innerHTML = `<p style="color: red;">加载内容失败: ${data.error || '未知错误'}</p>`;
        }
    } catch (error) {
        console.error("处理设置选项时出错:", error);
        // 显示网络或请求错误信息
        settingContentDisplay.innerHTML = `<p style="color: red;">加载内容时出错: ${error.message}</p>`;
    }
}

// 返回设置列表
function showSettingOptions() {
    console.log("返回设置选项列表");
    settingOptions.style.display = 'block'; // 显示选项列表
    settingContentDisplay.style.display = 'none'; // 隐藏内容区域
    backToSettingsButton.style.display = 'none'; // 隐藏返回按钮
    settingContentDisplay.innerHTML = ''; // 清空内容，避免下次直接显示旧内容
}

// 初始化
document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!message) return;
    if (!currentUsername) {
        showError("请先登录！");
        return;
    }

    // 1. Add user's message to the chat
    addMessage('user', message);
    input.value = ''; // Clear the input field

    // 2. Add a loading placeholder for the AI response and get a reference to its content element
    const aiMessageContent = addMessage('ai', '', true);

    try {
        const res = await fetch('/api/send', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, username: currentUsername })
        });

        // Clear the loading state
        aiMessageContent.innerHTML = '';
        aiMessageContent.classList.remove('loading');

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({ error: '无法解析服务器错误响应' }));
            throw new Error(errorData.error || '服务器返回了一个错误');
        }

        const data = await res.json();
        
        // Process the successful response
        if (data.success) {
            const response = data.response;

            if (response.type === 'causal_graph' && response.data) {
                // Handle causal graph response
                const summaryDiv = document.createElement('div');
                summaryDiv.innerHTML = marked.parse(response.summary || '已生成因果图，见下：');
                aiMessageContent.appendChild(summaryDiv);
                
                const graphContainerId = `graph-${Date.now()}`;
                const graphDiv = document.createElement('div');
                graphDiv.id = graphContainerId;
                graphDiv.className = 'causal-graph-container';
                aiMessageContent.appendChild(graphDiv);

                renderCausalGraph(graphContainerId, response.data);

            } else {
                // Handle plain text response
                aiMessageContent.innerHTML = marked.parse(response.summary || '抱歉，我无法回答这个问题。');
            }
        } else {
            aiMessageContent.textContent = `错误: ${data.error || '未知错误'}`;
        }

    } catch (error) {
        console.error('发送消息失败:', error);
        if (aiMessageContent) {
            aiMessageContent.innerHTML = ''; // Ensure loading content is gone
            aiMessageContent.classList.remove('loading');
            aiMessageContent.textContent = `请求失败: ${error.message}`;
        }
    } finally {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

// 显示错误
function showError(msg) {
    console.error('错误:', msg);
    alert('发生错误: ' + msg);
}

// 创建新会话
function newChat() {
     const loggedInUser = currentUsername;
     if (!loggedInUser) {
         showError("请先登录！");
         return;
     }

    if (confirm("确定要开始一个新的会话吗？\n（当前聊天记录会保存，可在历史中找回）")) {
        fetch('/api/new_chat', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('chatArea').innerHTML = '';
                    console.log("新会话已创建，重新加载历史记录...");
                    loadHistory();
                } else {
                    showError(data.error);
                }
            }).catch(error => showError('创建新会话时出错: ' + error));
    }
}

// 新增侧边栏切换功能
function toggleSidebar() {
    if (!currentUsername) return;

    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('mainContainer');
    const body = document.body;

    sidebar.classList.toggle('active');
    main.classList.toggle('sidebar-active');
    body.classList.toggle('sidebar-active');
}

// 加载历史记录
function loadHistory() {
     const loggedInUser = currentUsername;
     if (!loggedInUser) {
         console.log("用户未登录，不加载历史记录。");
         historyList.innerHTML = '<p style="padding: 10px; color: #888;">请先登录以查看历史记录。</p>';
         return;
     }

    console.log(`为用户 ${loggedInUser} 加载历史记录...`);
    fetch(`/api/sessions?user=${encodeURIComponent(loggedInUser)}`)
        .then(response => response.json())
        .then(data => {
            historyList.innerHTML = '';

            if (data.error) {
                 console.error("加载历史记录失败:", data.error);
                 historyList.innerHTML = `<p style="padding: 10px; color: red;">加载历史记录失败: ${data.error}</p>`;
                 return;
            }
            if (!Array.isArray(data)) {
                console.error("加载历史记录失败: 响应格式不正确");
                historyList.innerHTML = `<p style="padding: 10px; color: red;">加载历史记录失败: 格式错误</p>`;
                return;
            }
             if (data.length === 0) {
                 console.log(`用户 ${loggedInUser} 没有历史会话记录。`);
                 historyList.innerHTML = '<p style="padding: 10px; color: #888;">没有历史会话记录。</p>';
                 return;
             }

            console.log(`收到 ${data.length} 条历史记录项`);
            data.forEach(([sessionId, sessionData]) => {
                const item = document.createElement('div');
                item.className = 'history-item';
                let formattedTime = "时间未知";
                try {
                    formattedTime = new Date(sessionData.last_time).toLocaleString('zh-CN', { dateStyle: 'short', timeStyle: 'short' });
                } catch (e) {
                    console.warn(`无法解析会话 ${sessionId} 的时间: ${sessionData?.last_time}`);
                }
                 const previewText = sessionData?.preview || "无预览";

                item.innerHTML = `
                    <div class="session-info">
                        <div class="session-time">${formattedTime}</div>
                    </div>
                    <div class="preview-text">${previewText}</div>
                `;
                item.onclick = () => loadSession(sessionId, loggedInUser);
                historyList.appendChild(item);
            });
        }).catch(error => {
             console.error("加载历史记录时发生网络错误:", error);
             historyList.innerHTML = `<p style="padding: 10px; color: red;">加载历史记录时出错: ${error}</p>`;
        });
}

// 页面加载完成时
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM 已加载，检查登录状态...");
    checkLoginStatus();

    if (csvUploaderInput) {
        csvUploaderInput.addEventListener('change', handleCsvFileSelect);
    }
});


// 加载特定会话内容
async function loadSession(sessionId, username) {
     if (!username || username !== currentUsername) {
         showError("无法加载会话：用户状态异常或权限不足。");
         console.warn(`尝试加载会话 ${sessionId} 但参数用户 ${username} 与当前登录用户 ${currentUsername} 不匹配。`);
         return;
     }
    console.log(`用户 ${username} 尝试加载会话: ${sessionId}`);
    try {
        const response = await fetch(`/api/load_session?session=${sessionId}&user=${encodeURIComponent(username)}`);
        const data = await response.json();

        if (data.success) {
            const chatArea = document.getElementById('chatArea');
            chatArea.innerHTML = '';

            data.messages.forEach(message => {
                addMessage(message.sender, message.text);
            });

            chatArea.scrollTop = chatArea.scrollHeight;
        
            const sidebar = document.getElementById('sidebar');
            if (sidebar.classList.contains('active')) {
                 toggleSidebar();
            }
            console.log(`会话 ${sessionId} 加载成功`);
        } else {
            showError(data.error || `无法加载会话 ${sessionId}`);
        }
    } catch (error) {
        showError('加载会话时发生网络错误: ' + error);
        console.error("加载会话错误:", error);
    }
}

function triggerCsvUpload() {
    if (!currentUsername) {
        showError("请先登录才能上传文件！");
        return;
    }
    if (csvUploaderInput) {
        csvUploaderInput.click();
    }
}

// 处理文件选择和上传
async function handleCsvFileSelect(event) {
    if (!currentUsername) {
        showError("用户未登录，无法上传文件。");
        return;
    }

    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const fileName = file.name.toLowerCase();
    const allowedExtensions = ['.csv'];
    const allowedMimeTypes = ['text/csv', 'application/vnd.ms-excel', ''];


    if (!allowedExtensions.some(ext => fileName.endsWith(ext)) || !allowedMimeTypes.includes(file.type)) {
        showError('请选择一个有效的 CSV 文件 (.csv)。');
        event.target.value = null;
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('username', currentUsername);

    if (uploadCsvButton) {
        uploadCsvButton.textContent = '上传中...';
        uploadCsvButton.disabled = true;
    }

    try {
        const response = await fetch('/api/upload_file', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            alert(data.message || '文件上传成功！');
        } else {
            showError(data.error || '文件上传失败。');
        }
    } catch (error) {
        console.error("CSV Upload error:", error);
        showError('上传文件时发生网络错误。');
    } finally {
        if (uploadCsvButton) {
            uploadCsvButton.textContent = '上传';
            uploadCsvButton.disabled = false;
        }
        event.target.value = null;
    }
}

// --- 新增：渲染因果图表的函数 ---
function renderCausalGraph(containerId, graphData) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`图表容器 #${containerId} 未找到。`);
        return;
    }

    const nodes = new vis.DataSet(graphData.nodes);
    const edges = new vis.DataSet(graphData.edges);

    const data = {
        nodes: nodes,
        edges: edges,
    };

    const options = {
        layout: {
            hierarchical: {
                enabled: false,
            },
        },
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -50,
                centralGravity: 0.01,
                springConstant: 0.08,
                springLength: 100,
                damping: 0.4,
                avoidOverlap: 0.5
            }
        },
        nodes: {
            shape: 'box',
            size: 16,
            font: {
                size: 14,
                color: '#333'
            },
            borderWidth: 1,
            shadow: true,
        },
        edges: {
            width: 2,
            shadow: true,
            smooth: {
                enabled: true,
                type: 'dynamic'
            }
        },
        interaction: {
            dragNodes: true,
            dragView: true,
            zoomView: true,
        },
    };

    new vis.Network(container, data, options);
}

function addMessage(sender, text, isLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    // Create the main content block
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isLoading) {
        contentDiv.classList.add('loading');
        // Add three bouncing dots for loading animation
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dot.className = 'loading-dot';
            contentDiv.appendChild(dot);
        }
    } else {
        if (sender === 'user') {
            // For user messages, just use plain text to prevent XSS
            contentDiv.textContent = text;
        } else {
            // For AI messages, parse Markdown
            contentDiv.innerHTML = marked.parse(text);
        }
    }

    messageDiv.appendChild(contentDiv);
    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    // Return the content element so it can be updated (e.g., from loading to final content)
    return contentDiv;
}