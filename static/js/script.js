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
const fileList = document.getElementById('fileList'); // --- 新增: 获取 fileList 元素
const settingPopup = document.getElementById('settingPopup'); // 获取设置
const settingOptions = document.getElementById('settingOptions'); // 新增：获取设置选项容器
const settingContentDisplay = document.getElementById('settingContentDisplay'); // 获取内容显示区域
const backToSettingsButton = document.getElementById('backToSettingsButton'); // 获取返回按钮
const csvUploaderInput = document.getElementById('csvUploader'); // 获取CSV上传器
const uploadCsvButton = document.getElementById('uploadCsvButton'); // 获取上传按钮
const chatArea = document.getElementById('chatArea');
// --- 新增：全局变量存储当前会话的用户名 ---
let currentUsername = null;
let currentSessionId = null; // <--- 新增：全局变量跟踪当前会话ID
let chatEventListenersAttached = false; // 新增：跟踪事件监听器是否已附加

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
            
            // 登录成功后，绑定聊天界面的事件
            setupChatEventListeners();

            updateUserInfo(); // 更新用户信息显示
            loadHistory(); // --- 新增：先加载历史记录
            loadFiles(); // --- 新增: 加载文件列表 ---
            newChat(); // --- 修改：然后准备一个新对话界面
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
            chatEventListenersAttached = false; // --- 新增：重置监听器标志 ---
            document.body.classList.remove('logged-in'); // 移除标记类
            authOverlay.classList.add('active'); // 显示登录/注册层
            loginForm.style.display = 'block'; // 确保显示的是登录表单
            registerForm.style.display = 'none';
            closeUserInfoPopup(); // 关闭用户信息弹窗
            document.getElementById('chatArea').innerHTML = ''; // 清空聊天区域
            historyList.innerHTML = ''; // 清空历史列表
            fileList.innerHTML = ''; // --- 新增: 清空文件列表 ---
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
            
            // 状态检查通过后，绑定聊天界面的事件
            setupChatEventListeners();

            updateUserInfo(); // 更新用户信息显示 (稍后修改此函数)
            loadHistory(); // 加载历史记录 (稍后修改此函数)
            loadFiles(); // --- 新增: 加载文件列表 ---
        } else {
            console.log("后端验证：用户未登录，显示登录界面");
            currentUsername = null; // **修改**: 确保全局变量为空
            document.body.classList.remove('logged-in');
            authOverlay.classList.add('active');
            loginForm.style.display = 'block';
            registerForm.style.display = 'none';
            historyList.innerHTML = ''; // 清空可能存在的旧历史记录
            fileList.innerHTML = ''; // --- 新增: 清空文件列表 ---
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

function setupGlobalEventListeners() {
    // 登录/注册表单的切换
    document.getElementById('switchToRegister').addEventListener('click', (e) => {
        e.preventDefault();
        toggleAuthForms();
    });
    document.getElementById('switchToLogin').addEventListener('click', (e) => {
        e.preventDefault();
        toggleAuthForms();
    });

    // 登录和注册按钮
    document.getElementById('loginButton').addEventListener('click', handleLogin);
    document.getElementById('registerButton').addEventListener('click', handleRegister);
    
    // 用户信息弹窗
    document.getElementById('logoutButton').addEventListener('click', handleLogout);
    document.getElementById('closePopupButton').addEventListener('click', closeUserInfoPopup);
    
    // 设置弹窗
    document.getElementById('hideSettingPopupButton').addEventListener('click', hideSettingPopup);
    document.getElementById('backToSettingsButton').addEventListener('click', showSettingOptions);
    
    // 设置选项
    document.querySelectorAll('.setting-option').forEach(option => {
        option.addEventListener('click', (e) => {
            const optionId = e.currentTarget.getAttribute('data-option');
            handleSettingOption(optionId);
        });
    });
}

function setupChatEventListeners() {
    // --- 新增：防止重复绑定 ---
    if (chatEventListenersAttached) {
        return;
    }
    // 聊天输入和发送
    const sendButton = document.getElementById('sendButton');
    const userInput = document.getElementById('userInput');
    if (sendButton) sendButton.addEventListener('click', sendMessage);
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // 侧边栏和头部交互
    const menuIcon = document.getElementById('menuIcon');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const newChatButton = document.getElementById('newChatButton');
    const settingButton = document.getElementById('settingButton');
    const userAvatar = document.getElementById('userAvatar');
    const uploadCsvButton = document.getElementById('uploadCsvButton');
    
    if (menuIcon) menuIcon.addEventListener('click', toggleSidebar);
    if (sidebarToggle) sidebarToggle.addEventListener('click', toggleSidebar);
    if (newChatButton) newChatButton.addEventListener('click', newChat);
    if (settingButton) settingButton.addEventListener('click', showSettingPopup);
    if (userAvatar) userAvatar.addEventListener('click', showUserInfoPopup);
    if (uploadCsvButton) uploadCsvButton.addEventListener('click', triggerCsvUpload);

    chatEventListenersAttached = true; // --- 新增：设置标志 ---
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
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM 已加载，检查登录状态...");
    checkLoginStatus();
    // 设置全局监听器，这些元素始终存在
    setupGlobalEventListeners();

    // 注意：聊天界面的事件监听器已移至 setupChatEventListeners 函数中
    // 在登录成功后调用

    if (csvUploaderInput) {
        csvUploaderInput.addEventListener('change', handleCsvFileSelect);
    }
});

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    const isNewSession = !currentSessionId; // --- 新增：在开始时检查这是否是一个新会话 ---

    if (!message) {
        return;
    }

    if (!currentUsername) {
        showError("请先登录再发送消息！");
        return;
    }

    // 如果是新对话，先在后端创建会话
    if (!currentSessionId) {
        console.log("检测到新对话，正在后端创建会话...");
        try {
            const response = await fetch('/api/new_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ username: currentUsername })
            });
            const data = await response.json();
            if (data.success) {
                currentSessionId = data.new_session_id; // 更新全局ID
                console.log(`新会话已创建: ${currentSessionId}`);
                // --- 移除：不在这里加载历史，因为标题还没更新 ---
            } else {
                showError(data.error || "创建新对话失败。");
                return; // 创建失败则中止发送
            }
        } catch (error) {
            showError("创建新对话时发生网络错误。");
            console.error("创建新对话错误:", error);
            return; // 创建失败则中止发送
        }
    }

    addMessage('user', message);
    userInput.value = ''; // 清空输入框

    const loadingBubble = addMessage('ai', '', true);

    try {
        const response = await fetch('/api/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // --- 核心修改：在请求体中包含 session_id ---
            body: JSON.stringify({ 
                message: message, 
                username: currentUsername, // username 仍可用于日志记录，但后端不再依赖它进行认证
                session_id: currentSessionId 
            })
        });

        // 移除加载动画
        if (loadingBubble && loadingBubble.parentNode) {
            loadingBubble.parentNode.removeChild(loadingBubble);
        }

        const data = await response.json();

        if (data.success) {
            // The new addMessage function can handle both structured and plain text responses
            addMessage('ai', data.response);
            // --- 新增：在消息成功保存后（标题已更新），再加载历史记录 ---
            if (isNewSession) {
                loadHistory();
            }
        } else {
            showError(data.error || '从服务器获取响应失败。');
        }
    } catch (error) {
        // Also remove loading message on error
        if (loadingBubble && loadingBubble.parentNode) {
            loadingBubble.parentNode.removeChild(loadingBubble);
        }
        console.error("发送消息时出错:", error);
        showError('发送消息时发生网络错误。');
    }
}

// 显示错误
function showError(msg) {
    console.error('错误:', msg);
    alert('发生错误: ' + msg);
}

// 创建新会话
function newChat() {
    // 增加用户检查
    if (!currentUsername) {
        showError("请先登录！");
        return;
    }

    console.log("正在准备新聊天界面...");

    // 注意：后端会话将在用户发送第一条消息时创建。
    // --- 修复：立即在后端创建新会话 ---
    handleNewChatRequest();
}

async function handleNewChatRequest() {
    if (!currentUsername) {
        showError("请先登录！");
        return;
    }

    console.log("正在为新聊天创建会话...");
    chatArea.innerHTML = '<div class="loading-spinner"></div>'; // 显示加载动画

    try {
        const response = await fetch('/api/new_chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });

        const data = await response.json();
        chatArea.innerHTML = ''; // 清除加载动画

        if (data.success) {
            currentSessionId = data.new_session_id;
            console.log(`新会话已创建: ${currentSessionId}`);
            
            addMessage('ai', '你好！这是一个新的对话。你想聊些什么？你可以上传因果的数据文件，我将对该文件进行分析');
            document.getElementById('userInput').focus();

            await loadHistory(); // 重新加载历史列表以显示新会话
        } else {
            showError(data.error || "创建新对话失败。");
        }
    } catch (error) {
        chatArea.innerHTML = ''; // 确保出错时也移除加载动画
        showError("创建新对话时发生网络错误。");
        console.error("创建新对话错误:", error);
    }
}

// 更新会话标题
 async function updateSessionTitle(sessionId, title) {
    if (!title) {
        console.log("标题为空，取消更新。");
        return false; // 如果标题为空则不执行任何操作
    }
    try {
        // --- 修改：使用POST方法，并将数据放在body中 ---
        const response = await fetch('/api/change_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: title, session_id: sessionId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log(`会话 ${sessionId} 标题已更新为 "${title}"`);
            return true; // --- 修改：仅返回成功状态 ---
        } else {
            console.error("更新会话标题失败:", data.error);
            showError(`更新标题失败: ${data.error || '未知错误'}`);
            return false; // --- 修改：仅返回失败状态 ---
        }
    } catch (error) {
        console.error("更新会话标题时发生网络错误:", error);
        showError("更新标题时发生网络错误。");
        return false; // --- 修改：仅返回失败状态 ---
    }
}

// --- 新增：为标题添加可编辑监听器的辅助函数 ---
function addEditableListener(element, sessionId, title) {
    const handler = (e) => {
        e.stopPropagation();
        element.removeEventListener('click', handler); // 关键：在触发时移除自身，防止累积
        makeTitleEditable(element, sessionId, title); // 调用编辑器函数
    };
    element.addEventListener('click', handler);
}

// 侧边栏切换功能
function toggleSidebar() {
    if (!currentUsername) return;

    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('mainContainer');
    const body = document.body;

    sidebar.classList.toggle('active');
    main.classList.toggle('sidebar-active');
    body.classList.toggle('sidebar-active');
}

// --- 会话标题可编辑 ---
function makeTitleEditable(previewDiv, sessionId, oldTitle) {
    const parent = previewDiv.parentElement;
    if (!parent) {
        console.error("无法编辑标题：元素已脱离文档。");
        return;
    }
    
    // 创建一个新的输入框元素
    const input = document.createElement('input');
    input.type = 'text';
    input.value = oldTitle;
    input.className = 'title-edit-input';

    // 阻止点击事件冒泡
    input.addEventListener('click', (e) => e.stopPropagation());

    // 用输入框替换掉原来的标题div
    parent.replaceChild(input, previewDiv);
    input.focus();
    input.select();

    const finishEditing = async () => {
        const newTitle = input.value.trim();
        
        // 如果标题有效且被修改过
        if (newTitle && newTitle !== oldTitle) {
            input.disabled = true;
            input.value = '正在保存...';
            
            const success = await updateSessionTitle(sessionId, newTitle);
            
            if (success) {
                // 如果成功，更新原始div的文本
                previewDiv.textContent = newTitle;
                // 将更新后的div替换回输入框
                parent.replaceChild(previewDiv, input);
                // 关键：重新添加监听器，以便再次编辑
                addEditableListener(previewDiv, sessionId, newTitle);
            } else {
                // 如果失败，则重新加载整个历史记录以恢复状态
                await loadHistory();
            }
        } else {
            // 如果标题为空或未更改，则直接将原始div替换回来
            parent.replaceChild(previewDiv, input);
            // 关键：同样需要重新添加监听器
            addEditableListener(previewDiv, sessionId, oldTitle);
        }
    };

    // 监听输入框的事件
    input.addEventListener('blur', finishEditing);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            input.blur(); // 触发 blur 事件来保存
        } else if (e.key === 'Escape') {
            input.value = oldTitle; // 恢复旧值
            input.blur(); // 触发 blur 事件来取消编辑
        }
    });
}

// 加载历史记录
async function loadHistory() {
    if (!currentUsername) {
        historyList.innerHTML = '<p class="history-empty-message">请先登录以查看历史记录。</p>';
        return;
    }
    console.log(`为用户 ${currentUsername} 加载历史会话...`);

    try {
        const response = await fetch(`/api/sessions`);
        if (!response.ok) {
            if (response.status === 401) {
                // 如果是401未授权，可能是会话过期，可以提示用户重新登录
                 historyList.innerHTML = '<p class="history-empty-message">会话已过期，请重新登录。</p>';
                 handleLogout(); // 可以选择直接触发登出流程
                 return;
            }
            throw new Error(`服务器错误: ${response.status}`);
        }
        const sessions = await response.json();
        
        historyList.innerHTML = ''; // 清空旧列表

        if (Object.keys(sessions).length === 0) {
            historyList.innerHTML = '<p class="history-empty-message">还没有任何对话记录。</p>';
        } else {
            // --- 新增：用于跟踪当前打开的滑动项 ---
            let currentlyOpenItem = null;

            sessions.forEach(session => {
                const session_id = session[0];
                const info = session[1];

                // --- 核心修改：创建新的DOM结构以支持滑动 ---
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.setAttribute('data-session-id', session_id);

                // 删除按钮容器
                const swipeActions = document.createElement('div');
                swipeActions.className = 'swipe-actions';

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'delete-btn';
                deleteBtn.textContent = '删除';
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    handleDeleteSession(session_id, historyItem);
                };
                swipeActions.appendChild(deleteBtn);

                // 可见内容容器
                const itemContent = document.createElement('div');
                itemContent.className = 'history-item-content';
                
                const sessionInfo = document.createElement('div');
                sessionInfo.className = 'session-info';
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'session-time';
                timeDiv.textContent = info.last_time;
                
                const previewDiv = document.createElement('div');
                previewDiv.className = 'preview-text';
                previewDiv.textContent = info.preview;
                previewDiv.title = '点击修改标题';
                
                // 关键修改：使用辅助函数来添加监听器
                addEditableListener(previewDiv, session_id, info.preview);
                
                sessionInfo.appendChild(timeDiv);
                itemContent.appendChild(sessionInfo); // 将 info 放入 content
                itemContent.appendChild(previewDiv); // 将 preview 放入 content

                historyItem.appendChild(swipeActions); // 先添加操作按钮
                historyItem.appendChild(itemContent);  // 再添加可见内容

                historyList.appendChild(historyItem);

                // --- 新增：滑动逻辑 ---
                let isDragging = false;
                let startX = 0;
                let currentX = 0;
                let hasMoved = false; // --- 关键：用于区分拖拽和点击的标志 ---
                const threshold = -70; // 滑动阈值

                const closeCurrentlyOpen = () => {
                    if (currentlyOpenItem && currentlyOpenItem !== itemContent) {
                        currentlyOpenItem.style.transform = 'translateX(0px)';
                    }
                    currentlyOpenItem = null;
                };

                const onDragStart = (e) => {
                    // 如果点击的是输入框或标题，则不开始拖动
                    if (e.target.tagName.toLowerCase() === 'input' || e.target.classList.contains('preview-text')) {
                        return;
                    }
                    closeCurrentlyOpen();
                    hasMoved = false; // --- 关键：每次开始拖拽时，重置标志 ---
                    isDragging = true;
                    startX = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
                    itemContent.style.transition = 'none'; // 拖动时禁用平滑过渡
                };

                const onDragMove = (e) => {
                    if (!isDragging) return;

                    const moveX = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
                    // --- 关键：如果移动超过一个微小距离，就确认为拖拽 ---
                    if (!hasMoved && Math.abs(moveX - startX) > 5) {
                        hasMoved = true;
                    }

                    e.preventDefault(); // 防止页面滚动
                    currentX = moveX;
                    let diff = currentX - startX;
                    if (diff > 0) diff = 0; // 只允许向左滑
                    if (diff < threshold * 1.5) diff = threshold * 1.5;

                    itemContent.style.transform = `translateX(${diff}px)`;
                };

                const onDragEnd = () => {
                    if (!isDragging) return;
                    isDragging = false;
                    // 重新启用平滑过渡，让"吸附"动画生效
                    itemContent.style.transition = 'transform 0.3s ease';
                    
                    // --- 核心修改：使用 getComputedStyle 来可靠地获取当前位置 ---
                    const computedStyle = window.getComputedStyle(itemContent);
                    // DOMMatrix 可以安全地解析 transform 属性，无论是 'translateX' 还是 'matrix'
                    const transformMatrix = new DOMMatrix(computedStyle.transform);
                    const finalX = transformMatrix.m41; // m41 是 X 轴的平移量

                    // 如果滑动距离小于阈值的一半，则认为用户是想打开会话
                    if (finalX < threshold / 2) {
                        itemContent.style.transform = `translateX(${threshold}px)`;
                        currentlyOpenItem = itemContent; // 更新当前打开的项
                    } else {
                        itemContent.style.transform = 'translateX(0px)';
                    }
                };
                
                // 加载会话的点击事件
                itemContent.addEventListener('click', (e) => {
                    // --- 关键：如果是拖拽，则不执行点击逻辑 ---
                    if (hasMoved) {
                        e.stopPropagation();
                        return;
                    }

                    if (e.target.tagName.toLowerCase() !== 'input' && !e.target.classList.contains('preview-text')) {
                        const currentTransform = window.getComputedStyle(itemContent).transform;
                        if(currentTransform === 'none' || currentTransform === 'matrix(1, 0, 0, 1, 0, 0)'){
                             loadSession(session_id);
                        } else {
                            // 如果是打开状态，则关闭它
                            itemContent.style.transform = 'translateX(0px)';
                            currentlyOpenItem = null;
                        }
                    }
                });

                // 绑定鼠标和触摸事件
                itemContent.addEventListener('mousedown', onDragStart);
                itemContent.addEventListener('mousemove', onDragMove);
                itemContent.addEventListener('mouseup', onDragEnd);
                itemContent.addEventListener('mouseleave', onDragEnd); // 鼠标离开也结束拖动

                itemContent.addEventListener('touchstart', onDragStart);
                itemContent.addEventListener('touchmove', onDragMove);
                itemContent.addEventListener('touchend', onDragEnd);
            });
        }
    } catch (error) {
        console.error("加载历史记录失败:", error);
        historyList.innerHTML = `<p class="history-empty-message">加载历史记录失败: ${error.message}</p>`;
    }
}

// --- 新增：处理会话删除的函数 ---
async function handleDeleteSession(sessionId, element) {
    if (!confirm("确定要永久删除此会话及其所有消息吗？此操作无法撤销。")) {
        return;
    }

    try {
        const response = await fetch('/api/delete_session', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId })
        });
        const data = await response.json();

        if (data.success) {
            console.log("会话已在后端删除");
            // 平滑的删除动画
            element.style.height = `${element.offsetHeight}px`; // 固定高度
            element.style.opacity = '0';
            element.style.transform = 'translateX(-100%)';
            element.style.marginBottom = `-${element.offsetHeight}px`;
            
            setTimeout(() => {
                element.remove();
                // 检查是否列表已空
                if (historyList.children.length === 0) {
                     historyList.innerHTML = '<p class="history-empty-message">还没有任何对话记录。</p>';
                }
            }, 300); // 匹配CSS过渡时间
        } else {
            showError(data.error || "删除失败，请重试。");
            // 如果删除失败，则关闭滑动状态
            const content = element.querySelector('.history-item-content');
            if (content) {
                content.style.transform = 'translateX(0px)';
            }
        }
    } catch (error) {
        showError("删除会话时发生网络错误。");
        console.error("删除会话错误:", error);
    }
}

// 加载特定会话内容
async function loadSession(sessionId) {
    if (!currentUsername) {
        showError("登录状态异常，请刷新页面。");
        return;
    }

    console.log(`用户 ${currentUsername} 正在加载会话: ${sessionId}`);
    chatArea.innerHTML = '<div class="loading-spinner"></div>'; // 显示加载动画

    try {
        const response = await fetch(`/api/load_session?session=${sessionId}&user=${currentUsername}`);
        const data = await response.json();

        chatArea.innerHTML = ''; // 清除加载动画

        if (data.success) {
            currentSessionId = sessionId; // <--- 核心修改：更新全局会话ID
            data.messages.forEach(msg => {
                addMessage(msg.sender, msg.text);
            });
            // 确保加载会话后事件监听器也是最新的
            // setupChatEventListeners();  // 不再需要，因为元素是持久的
            console.log(`会话 ${sessionId} 已成功加载`);
        } else {
            showError(data.error || "无法加载会话。");
            console.error("加载会话失败:", data.error);
        }
    } catch (error) {
        chatArea.innerHTML = ''; // 确保出错时也移除加载动画
        showError("加载会话时发生网络错误。");
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
        showError("请先登录再上传文件！");
        return;
    }

    const file = event.target.files[0];
    if (!file) {
        return;
    }

    // --- 新增：检查会话ID ---
    if (!currentSessionId) {
        showError("没有活动的会话，无法上传文件。请新建一个对话或加载历史会话。");
        // 恢复按钮状态
        if (uploadCsvButton) {
            uploadCsvButton.textContent = '上传';
            uploadCsvButton.disabled = false;
        }
        event.target.value = null; // 清除文件选择
        return;
    }
    // -------------------------

    // 显示上传开始的用户消息和AI加载动画
    addMessage('user', `正在上传文件: ${file.name}`);
    const loadingMessageElement = addMessage('ai', '', true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', currentSessionId);

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

        // 移除加载动画，显示最终结果
        if (loadingMessageElement && loadingMessageElement.parentNode) {
            loadingMessageElement.parentNode.removeChild(loadingMessageElement);
        }

        if (data.success) {
            // 显示成功的AI回复消息
            addMessage('ai', `已接收您的文件：${file.name}\n\n${data.message}\n\n您现在可以询问我对此文件进行因果分析。`);
            loadFiles(); // --- 新增：刷新文件列表
        } else {
            // 显示错误的AI回复消息
            addMessage('ai', `文件上传失败：${data.error || '未知错误'}`);
            showError(data.error || '文件上传失败。');
        }
    } catch (error) {
        console.error("CSV Upload error:", error);
        
        // 移除加载动画
        if (loadingMessageElement && loadingMessageElement.parentNode) {
            loadingMessageElement.parentNode.removeChild(loadingMessageElement);
        }
        
        // 显示网络错误的AI回复
        addMessage('ai', '文件上传时发生网络错误，请检查网络连接后重试。');
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
        console.error(`无法找到ID为 "${containerId}" 的容器来渲染图表。`);
        return;
    }
     // 确保 graphData 是预期的格式
    if (!graphData || !Array.isArray(graphData.nodes) || !Array.isArray(graphData.edges)) {
        console.error('无效的图表数据格式:', graphData);
        container.textContent = '错误：无法加载图表，数据格式不正确。';
        return;
    }

    // 将 causal-learn 格式的节点和边转换为 vis.js 格式
    const nodes = new vis.DataSet(graphData.nodes);
    const edges = new vis.DataSet(graphData.edges);

    const data = {
        nodes: nodes,
        edges: edges,
    };
    const options = {
        layout: {
            hierarchical: {
                enabled: false, // 可以设为 true 尝试层次布局
            },
        },
        edges: {
            arrows: {
                to: { enabled: true, scaleFactor: 1, type: 'arrow' }
            },
            color: '#848484',
            font: {
                size: 12,
            },
            smooth: {
                enabled: true,
                type: 'dynamic', // 'dynamic' 对于非层次结构通常效果更好
            },
        },
        nodes: {
            shape: 'box', // 节点形状
            size: 30,
            font: {
                size: 14,
                color: '#333'
            },
            borderWidth: 2,
        },
        interaction: {
            dragNodes: true,
            dragView: true,
            zoomView: true,
        },
        physics: {
            enabled: true, // 启用物理引擎以自动布局
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.3,
                springLength: 95,
                springConstant: 0.04,
                damping: 0.09,
                avoidOverlap: 0.1
            },
            solver: 'barnesHut',
            stabilization: {
                iterations: 1000,
            },
        },
    };

    try {
        const network = new vis.Network(container, data, options);
        // 稳定后关闭物理引擎，以节省CPU
        network.on("stabilizationIterationsDone", function () {
            network.setOptions( { physics: false } );
        });
    } catch (err) {
        console.error("创建 vis.js 网络时出错:", err);
        container.textContent = "渲染图表时发生错误。";
    }
}

// 加载消息
function addMessage(sender, messageData, isLoading = false) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);

    const contentElement = document.createElement('div');
    contentElement.classList.add('content');

    if (isLoading) {
        const loadingDots = document.createElement('div');
        loadingDots.classList.add('loading-dots');
        for (let i = 0; i < 3; i++) {
            loadingDots.appendChild(document.createElement('div'));
        }
        contentElement.appendChild(loadingDots);
    } else {
        // --- 核心修改：处理不同类型的 messageData ---
        if (sender === 'ai' && typeof messageData === 'object' && messageData !== null) {
            // 处理AI的结构化响应
            if (messageData.type === 'causal_graph' && messageData.data) {
                // 1. 添加总结文本（如果存在）
                if (messageData.summary) {
                    const summaryDiv = document.createElement('div');
                    summaryDiv.innerHTML = marked.parse(messageData.summary);
                    contentElement.appendChild(summaryDiv);
                }

                // 2. 创建并渲染因果图
                const graphContainerId = `graph-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                const graphContainer = document.createElement('div');
                graphContainer.id = graphContainerId;
                graphContainer.classList.add('causal-graph-container'); // 用于样式
                contentElement.appendChild(graphContainer);

                // 使用 setTimeout 确保元素已添加到 DOM 中
                // vis.js 需要一个已挂载的容器来进行初始化
                setTimeout(() => {
                    renderCausalGraph(graphContainerId, messageData.data);
                }, 100);

            } else if (messageData.summary) {
                // 对于其他类型的结构化响应（例如只有 'type': 'text'），只显示总结
                contentElement.innerHTML = marked.parse(messageData.summary);
            } else {
                // 如果对象无法识别，则作为字符串显示以供调试
                contentElement.textContent = JSON.stringify(messageData, null, 2);
            }
        } else {
            // 对于用户消息（总是字符串）和旧的纯文本AI消息
            contentElement.innerHTML = marked.parse(messageData.toString());
        }
        // --- 修改结束 ---
    }
    
    // 直接添加内容元素，不需要头像
    messageElement.appendChild(contentElement);
    
    chatArea.appendChild(messageElement);
    chatArea.scrollTop = chatArea.scrollHeight; // 自动滚动到底部
    
    // 返回消息元素，以便后续可以移除（例如加载动画）
    return messageElement;
}

// --- 新增：加载文件列表的函数 ---
async function loadFiles() {
    if (!currentUsername) {
        if (fileList) fileList.innerHTML = '<p class="files-empty-message">请先登录以查看文件列表。</p>';
        return;
    }
    console.log(`为用户 ${currentUsername} 加载文件列表...`);
    if (!fileList) return;

    try {
        const response = await fetch(`/api/files`);
        if (!response.ok) {
            if (response.status === 401) {
                 fileList.innerHTML = '<p class="files-empty-message">会话已过期，请重新登录。</p>';
                 return;
            }
            throw new Error(`服务器错误: ${response.status}`);
        }
        const files = await response.json();
        
        fileList.innerHTML = ''; // 清空旧列表

        if (Object.keys(files).length === 0) {
            fileList.innerHTML = '<p class="files-empty-message">还没有任何文件记录。</p>';
        } else {
            let currentlyOpenFileItem = null;

            files.forEach(file => {
                const file_id = file[0];
                const info = file[1];

                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.setAttribute('data-file-id', file_id);

                const swipeActions = document.createElement('div');
                swipeActions.className = 'swipe-actions';

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'delete-btn';
                deleteBtn.textContent = '删除';
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    handleDeleteFile(file_id, fileItem);
                };
                swipeActions.appendChild(deleteBtn);

                const itemContent = document.createElement('div');
                itemContent.className = 'file-item-content';
                
                const sessionInfo = document.createElement('div');
                sessionInfo.className = 'session-info';
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'session-time';
                timeDiv.textContent = info.last_time;
                
                const previewDiv = document.createElement('div');
                previewDiv.className = 'preview-text';
                previewDiv.textContent = info.preview;
                previewDiv.title = info.preview;
                
                sessionInfo.appendChild(timeDiv);
                itemContent.appendChild(sessionInfo);
                itemContent.appendChild(previewDiv);

                fileItem.appendChild(swipeActions);
                fileItem.appendChild(itemContent);

                fileList.appendChild(fileItem);

                // --- 滑动逻辑 (与 history item 相同) ---
                let isDragging = false, startX = 0, currentX = 0, hasMoved = false;
                const threshold = -70;

                const closeCurrentlyOpen = () => {
                    if (currentlyOpenFileItem && currentlyOpenFileItem !== itemContent) {
                        currentlyOpenFileItem.style.transform = 'translateX(0px)';
                    }
                    currentlyOpenFileItem = null;
                };

                const onDragStart = (e) => {
                    closeCurrentlyOpen();
                    hasMoved = false;
                    isDragging = true;
                    startX = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
                    itemContent.style.transition = 'none';
                };

                const onDragMove = (e) => {
                    if (!isDragging) return;
                    if (!hasMoved && Math.abs((e.type.includes('mouse') ? e.pageX : e.touches[0].pageX) - startX) > 5) {
                        hasMoved = true;
                    }
                    e.preventDefault();
                    currentX = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
                    let diff = currentX - startX;
                    if (diff > 0) diff = 0;
                    if (diff < threshold * 1.5) diff = threshold * 1.5;
                    itemContent.style.transform = `translateX(${diff}px)`;
                };

                const onDragEnd = () => {
                    if (!isDragging) return;
                    isDragging = false;
                    itemContent.style.transition = 'transform 0.3s ease';
                    const computedStyle = window.getComputedStyle(itemContent);
                    const transformMatrix = new DOMMatrix(computedStyle.transform);
                    const finalX = transformMatrix.m41;

                    if (finalX < threshold / 2) {
                        itemContent.style.transform = `translateX(${threshold}px)`;
                        currentlyOpenFileItem = itemContent;
                    } else {
                        itemContent.style.transform = 'translateX(0px)';
                    }
                };
                
                // 如果点击该文件
                itemContent.addEventListener('click', (e) => {
                    if (hasMoved) {
                        e.stopPropagation();
                        return;
                    }
                    // 点击文件项的逻辑：将文件名插入输入框
                    const userInput = document.getElementById('userInput');
                    if (userInput) {
                        // 将预设的分析指令和文件名填入输入框
                        userInput.value = `请对文件"${info.preview}"进行因果分析`;
                        // 聚焦输入框，方便用户直接发送
                        userInput.focus();
                    }
                    console.log(`File "${info.preview}" reference inserted into input box.`);
                });

                itemContent.addEventListener('mousedown', onDragStart);
                itemContent.addEventListener('mousemove', onDragMove);
                itemContent.addEventListener('mouseup', onDragEnd);
                itemContent.addEventListener('mouseleave', onDragEnd);
                itemContent.addEventListener('touchstart', onDragStart);
                itemContent.addEventListener('touchmove', onDragMove);
                itemContent.addEventListener('touchend', onDragEnd);
            });
        }
    } catch (error) {
        console.error("加载文件列表失败:", error);
        fileList.innerHTML = `<p class="files-empty-message">加载文件列表失败: ${error.message}</p>`;
    }
}

// --- 新增：处理文件删除的函数 ---
async function handleDeleteFile(fileId, element) {
    if (!confirm("确定要永久删除此文件吗？此操作无法撤销。")) {
        return;
    }

    try {
        const response = await fetch('/api/delete_file', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ file_id: fileId })
        });
        const data = await response.json();

        if (data.success) {
            console.log("文件已在后端删除");
            // 平滑的删除动画
            element.style.height = `${element.offsetHeight}px`; // 固定高度
            element.style.opacity = '0';
            element.style.transform = 'translateX(-100%)';
            element.style.marginBottom = `-${element.offsetHeight}px`;
            
            setTimeout(() => {
                element.remove();
                if (fileList.children.length === 0) {
                     fileList.innerHTML = '<p class="files-empty-message">还没有任何文件记录。</p>';
                }
            }, 300); // 匹配CSS过渡时间
        } else {
            showError(data.error || "删除失败，请重试。");
            // 如果删除失败，则关闭滑动状态
            const content = element.querySelector('.file-item-content');
            if (content) {
                content.style.transform = 'translateX(0px)';
            }
        }
    } catch (error) {
        showError("删除文件时发生网络错误。");
        console.error("删除文件错误:", error);
    }
}
