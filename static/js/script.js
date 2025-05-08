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
            // localStorage.setItem('loggedInUser', data.username); // <-- 移除
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
    // const username = localStorage.getItem('loggedInUser'); // <-- 移除
    const username = currentUsername; // **修改**: 使用全局变量获取当前用户 (主要用于日志)
    if (!username) return; // 如果没有当前用户，直接返回

    console.log(`用户 ${username} 正在请求退出登录`);

    try {
         // --- 新增：调用后端登出接口 ---
        const response = await fetch('/api/logout', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            console.log("后端登出成功");
            // localStorage.removeItem('loggedInUser'); // <-- 移除
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

// 检查登录状态 (页面加载时调用) - **修改**
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

// 更新用户界面信息（例如头像区域） - **修改**
function updateUserInfo() {
    // const loggedInUser = localStorage.getItem('loggedInUser'); // <-- 移除
    if (currentUsername) { // **修改**: 使用全局变量
        userAvatar.textContent = currentUsername.charAt(0).toUpperCase(); // 显示用户名首字母
        userInfoContent.textContent = `账号: ${currentUsername}`; // 设置弹窗内容
    } else {
        userAvatar.textContent = ''; // 未登录则清空
        userInfoContent.textContent = ''; // 清空弹窗内容
    }
}

// 显示用户信息弹窗 - **修改**
function showUserInfoPopup() {
    // if (!localStorage.getItem('loggedInUser')) return; // <-- 移除
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
        // showError(`加载内容时出错: ${error.message}`); // 也可以用 alert
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
// --- 修改现有函数以包含登录检查和用户名 ---

// 初始化 (不变)
document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// API交互，使用异步调用，防止页面阻塞 - **修改**
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    // const loggedInUser = localStorage.getItem('loggedInUser'); // <-- 移除
    const loggedInUser = currentUsername; // **修改**: 使用全局变量

    if (!message) return;

     // 检查是否登录
     if (!loggedInUser) {
         showError("请先登录！");
         return;
     }

    addMessage(message, 'user');
    input.value = '';//清空输入框内容。

    try {
        const response = await fetch('/api/send', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            // **修改：** 发送消息和用户名 (这里逻辑不变，因为之前已从变量获取)
            body: JSON.stringify({ message: message, username: loggedInUser })
        });

        const data = await response.json();
        if (data.success) {
            addMessage(data.response, 'ai');
            // 可以在这里考虑是否需要更新历史列表预览，目前不刷新
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('与服务器通信时出错: '+ error);
    }
}

// 界面更新
function addMessage(text, sender) {
    const chatArea = document.getElementById('chatArea'); // 获取聊天区域
    const div = document.createElement('div');//使用 在内存中创建一个新的、空白的 div 元素
    div.className = `message ${sender}-message`; // 类名保持不变
     // 它会包含 'message' 类和根据发送者是 'user' 还是 'ai' 动态添加的 'user-message' 或 'ai-message' 类。这对应了 CSS 中的样式规则。
     if (sender === 'ai') {
    div.innerHTML = marked.parse(text);
} else {
    // 用户消息仍然使用 textContent 以防止 XSS
    div.textContent = text;
}
    chatArea.appendChild(div);
    // 修改滚动逻辑，滚动聊天区域而不是整个 body
    chatArea.scrollTop = chatArea.scrollHeight; 
    // `chatArea.scrollHeight` 是 `chatArea` 内部内容的总高度（即使内容超出了可见区域）。
     // `chatArea.scrollTop` 是 `chatArea` 向上滚动的距离。将其设置为 `scrollHeight` 意味着滚动到最底部，使用户能看到最新的消息。
    
}

// 显示错误 (不变)
function showError(msg) {
    // 考虑将错误显示在更友好的地方，而不是 alert
    console.error('错误:', msg);
    alert('发生错误: ' + msg);
}

// 创建新会话 - **修改**
function newChat() {
     // const loggedInUser = localStorage.getItem('loggedInUser'); // <-- 移除
     const loggedInUser = currentUsername; // **修改**: 使用全局变量
     if (!loggedInUser) {
         showError("请先登录！");
         return;
     }

    if (confirm("确定要开始一个新的会话吗？\n（当前聊天记录会保存，可在历史中找回）")) {
        // 调用 /api/new_chat (这个接口本身不需要用户名)
        fetch('/api/new_chat', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('chatArea').innerHTML = ''; // 清空聊天区域
                    console.log("新会话已创建，重新加载历史记录...");
                    loadHistory(); // 重新加载历史
                } else {
                    showError(data.error);
                }
            }).catch(error => showError('创建新会话时出错: ' + error));
    }
}

// API 切换 - **修改** (虽然API切换本身不需要用户，但检查登录状态需要)
document.getElementById('apiSelect').addEventListener('change', async (e) => {
     // const loggedInUser = localStorage.getItem('loggedInUser'); // <-- 移除
     const loggedInUser = currentUsername; // **修改**: 使用全局变量检查登录
     if (!loggedInUser) {
         showError("请先登录再切换 API！"); // 提示用户
         // 可以选择将下拉列表重置回之前的值
         // e.target.value = previousApiValue; // (需要保存之前的API值)
         return;
     }

    try {
        await fetch('/api/switch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ api: e.target.value })
        });
         console.log(`API 已切换为 ${e.target.value}`);
         // previousApiValue = e.target.value; // 保存当前值
    } catch (error) {
         showError('切换 API 时出错: ' + error);
         // e.target.value = previousApiValue; // 出错时恢复
    }
});

// 新增侧边栏切换功能 - **修改**
function toggleSidebar() {
    // if (!localStorage.getItem('loggedInUser')) return; // <-- 移除
    if (!currentUsername) return; // **修改**: 使用全局变量

    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('mainContainer');
    const body = document.body;

    sidebar.classList.toggle('active');
    main.classList.toggle('sidebar-active');
    body.classList.toggle('sidebar-active');
}

// 加载历史记录 (修改为加载当前用户的历史) - **修改**
function loadHistory() {
     // const loggedInUser = localStorage.getItem('loggedInUser'); // <-- 移除
     const loggedInUser = currentUsername; // **修改**: 使用全局变量
     if (!loggedInUser) {
         console.log("用户未登录，不加载历史记录。");
         historyList.innerHTML = '<p style="padding: 10px; color: #888;">请先登录以查看历史记录。</p>';
         return;
     }

    console.log(`为用户 ${loggedInUser} 加载历史记录...`);
    // **修改：** 在请求中加入用户名 (这里逻辑不变，因为之前已从变量获取)
    fetch(`/api/sessions?user=${encodeURIComponent(loggedInUser)}`)
        .then(response => response.json())
        .then(data => {
            historyList.innerHTML = ''; // 清空

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
                     // 尝试解析时间，如果 sessionData 或 last_time 不存在，会自然出错
                    formattedTime = new Date(sessionData.last_time).toLocaleString('zh-CN', { dateStyle: 'short', timeStyle: 'short' });
                } catch (e) {
                    console.warn(`无法解析会话 ${sessionId} 的时间: ${sessionData?.last_time}`);
                }
                 // 增加预览文本的健壮性
                 const previewText = sessionData?.preview || "无预览";

                item.innerHTML = `
                    <div class="session-info">
                        <div class="session-time">${formattedTime}</div>
                    </div>
                    <div class="preview-text">${previewText}</div>
                `;
                // **修改：** 加载会话时也需要用户名
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
    checkLoginStatus(); // 检查登录状态 (这个函数现在会调用 /api/check_auth)
});

// 加载特定会话内容 (修改为需要用户名) - **修改**
async function loadSession(sessionId, username) { // 参数 username 仍然需要
     // const loggedInUser = currentUsername; // 可以用全局变量再次确认，但参数传递更直接
     if (!username || username !== currentUsername) { // **修改**: 做个检查，确保是当前登录用户在操作
         showError("无法加载会话：用户状态异常或权限不足。");
         console.warn(`尝试加载会话 ${sessionId} 但参数用户 ${username} 与当前登录用户 ${currentUsername} 不匹配。`);
         return;
     }
    console.log(`用户 ${username} 尝试加载会话: ${sessionId}`);
    try {
        // **修改：** 在请求中加入用户名 (这里逻辑不变)
        const response = await fetch(`/api/load_session?session=${sessionId}&user=${encodeURIComponent(username)}`);
        const data = await response.json();

        if (data.success) {
            const chatArea = document.getElementById('chatArea');
            chatArea.innerHTML = ''; // 清空当前聊天区域

            // 遍历返回的消息并添加到聊天区
            data.messages.forEach(message => {
                addMessage(message.text, message.sender);
            });

            // 确保滚动到底部
            chatArea.scrollTop = chatArea.scrollHeight;
            // (可选) 加载成功后关闭侧边栏
            // 这里有个小问题：如果侧边栏本来就没开，toggle 会把它打开
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
