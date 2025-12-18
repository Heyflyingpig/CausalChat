//  认证相关变量和函数 
const authOverlay = document.getElementById('authOverlay');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const loginError = document.getElementById('loginError');
const registerError = document.getElementById('registerError'); // 错误
const userAvatar = document.getElementById('userAvatar'); //头像
const userInfoPopup = document.getElementById('userInfoPopup');
const userInfoContent = document.getElementById('userInfoContent');
const historyList = document.getElementById('historyList'); // 获取 historyList 元素
const fileList = document.getElementById('fileList'); //  获取 fileList 元素
const settingPopup = document.getElementById('settingPopup'); // 获取设置
const settingOptions = document.getElementById('settingOptions'); // 获取设置选项容器
const settingContentDisplay = document.getElementById('settingContentDisplay'); // 获取内容显示区域
const backToSettingsButton = document.getElementById('backToSettingsButton'); // 获取返回按钮
const csvUploaderInput = document.getElementById('csvUploader'); // 获取CSV上传器
const uploadCsvButton = document.getElementById('uploadCsvButton'); // 获取上传按钮
const chatArea = document.getElementById('chatArea');
//全局变量存储当前会话的用户名
let currentUsername = null;
let currentSessionId = null; // < 全局变量跟踪当前会话ID
let isNewSessionPendingDisplay = false; //  用于跟踪新会话是否已在UI中临时显示
let chatEventListenersAttached = false; // 跟踪事件监听器是否已附加
let currentLanguage = localStorage.getItem('language') || 'zh'; // 当前语言，默认中文，从localStorage读取

// 多语言文本配置
const i18n = {
    zh: {
        // 登录/注册页面
        login: '登录',
        register: '注册',
        username: '用户名:',
        password: '密码:',
        confirmPassword: '确认密码:',
        loginBtn: '登录',
        registerBtn: '注册',
        noAccountRegister: '还没有账号？点击注册',
        hasAccountLogin: '已有账号？点击登录',
        // 用户信息弹窗
        userInfo: '用户信息',
        logout: '退出登录',
        close: '关闭',
        // 设置弹窗
        settings: '设置',
        userAgreement: '用户协议',
        userManual: '操作文档',
        checkUpdate: '检查更新',
        toggleLanguage: '切换英文',
        back: '返回',
        // 侧边栏
        newChat: '新建对话',
        fileList: '文件列表',
        // 输入区域
        inputPlaceholder: '输入消息...',
        upload: '上传',
        send: '发送',
        // 动态消息
        accountPrefix: '账号: ',
        noHistoryMessage: '还没有任何对话记录。',
        noFilesMessage: '还没有任何文件记录。',
        pleaseLoginHistory: '请先登录以查看历史记录。',
        pleaseLoginFiles: '请先登录以查看文件列表。',
        sessionExpired: '会话已过期，请重新登录。',
        justNow: '刚刚',
        deleteBtn: '删除',
        thinking: '正在思考...',
        thinkingWait: '请稍等...',
        inProgress: '进行中...',
        newChatGreeting: '你好！这是一个新的对话。你想聊些什么？你可以上传因果的数据文件，我将对该文件进行分析',
        uploading: '上传中...',
        uploadingFile: '正在上传文件: ',
        fileReceived: '已接收您的文件：',
        askCausalAnalysis: '\n\n您现在可以询问我对此文件进行因果分析。',
        uploadFailed: '文件上传失败：',
        networkError: '文件上传时发生网络错误，请检查网络连接后重试。',
        versionUpdated: '版本已经更新到最新',
        loadingContent: '正在加载内容...',
        loadFailed: '加载内容失败: ',
        loadError: '加载内容时出错: '
    },
    en: {
        // Login/Register page
        login: 'Login',
        register: 'Register',
        username: 'Username:',
        password: 'Password:',
        confirmPassword: 'Confirm Password:',
        loginBtn: 'Login',
        registerBtn: 'Register',
        noAccountRegister: "Don't have an account? Register",
        hasAccountLogin: 'Already have an account? Login',
        // User info popup
        userInfo: 'User Info',
        logout: 'Logout',
        close: 'Close',
        // Settings popup
        settings: 'Settings',
        userAgreement: 'User Agreement',
        userManual: 'User Manual',
        checkUpdate: 'Check Update',
        toggleLanguage: 'Switch to Chinese',
        back: 'Back',
        // Sidebar
        newChat: 'New Chat',
        fileList: 'File List',
        // Input area
        inputPlaceholder: 'Type a message...',
        upload: 'Upload',
        send: 'Send',
        // Dynamic messages
        accountPrefix: 'Account: ',
        noHistoryMessage: 'No conversation history yet.',
        noFilesMessage: 'No files yet.',
        pleaseLoginHistory: 'Please login to view history.',
        pleaseLoginFiles: 'Please login to view files.',
        sessionExpired: 'Session expired, please login again.',
        justNow: 'Just now',
        deleteBtn: 'Delete',
        thinking: 'Thinking...',
        thinkingWait: 'please wait...',
        inProgress: 'In progress...',
        newChatGreeting: 'Hello! This is a new conversation. What would you like to talk about? You can upload a causal data file for analysis.',
        uploading: 'Uploading...',
        uploadingFile: 'Uploading file: ',
        fileReceived: 'File received: ',
        askCausalAnalysis: '\n\nYou can now ask me to perform causal analysis on this file.',
        uploadFailed: 'File upload failed: ',
        networkError: 'Network error during file upload. Please check your connection and try again.',
        versionUpdated: 'Version is up to date',
        loadingContent: 'Loading content...',
        loadFailed: 'Failed to load content: ',
        loadError: 'Error loading content: '
    }
};

// 获取当前语言的文本
function getText(key) {
    return i18n[currentLanguage][key] || i18n['zh'][key] || key;
}

// 应用语言到所有带有 data-i18n 属性的元素
function applyLanguage() {
    // 更新所有带有 data-i18n 属性的元素的文本内容
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (i18n[currentLanguage][key]) {
            element.textContent = i18n[currentLanguage][key];
        }
    });

    // 更新所有带有 data-i18n-placeholder 属性的元素的 placeholder
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        if (i18n[currentLanguage][key]) {
            element.placeholder = i18n[currentLanguage][key];
        }
    });

    // 更新用户信息弹窗内容（如果用户已登录）
    if (currentUsername) {
        userInfoContent.textContent = getText('accountPrefix') + currentUsername;
    }
}

// 切换语言
function toggleLanguage() {
    currentLanguage = currentLanguage === 'zh' ? 'en' : 'zh';
    localStorage.setItem('language', currentLanguage); // 保存到 localStorage
    applyLanguage();
    console.log(`语言已切换为: ${currentLanguage === 'zh' ? '中文' : 'English'}`);
}

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
        // 直接发送明文密码（通过 HTTPS 保护传输，后端使用 bcrypt 进行哈希）
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ username: username, password: password }) // 发送明文密码
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

// 处理登录 - 
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
        const response = await fetch('/api/login', {
             method: 'POST',
             headers: {'Content-Type': 'application/json'},
             body: JSON.stringify({ username: username, password: password})
        });
        const data = await response.json();

        if (data.success && data.username) { // 确保返回了 username
            // 登录成功
            currentUsername = data.username; //  设置全局变量
            document.body.classList.add('logged-in'); // 添加标记类
            authOverlay.classList.remove('active'); // 隐藏登录/注册层
            
            // 登录成功后，绑定聊天界面的事件
            setupChatEventListeners();

            updateUserInfo(); // 更新用户信息显示
            loadHistory(); //  先加载历史记录
            loadFiles(); //   加载文件列表 
            newChat(); //  然后准备一个新对话界面
             // 清空登录表单
            usernameInput.value = '';
            passwordInput.value = '';
        } else {
            loginError.textContent = data.error || '登录失败，请检查用户名和密码。';
            currentUsername = null; //  确保登录失败时全局变量为空
        }
    } catch (error) {
        console.error("Login error:", error);
        loginError.textContent = '登录过程中发生错误。';
        currentUsername = null; //  确保出错时全局变量为空
    }
}

// 处理退出登录 - 
async function handleLogout() {
    const username = currentUsername; //  使用全局变量获取当前用户 (主要用于日志)
    if (!username) return; // 如果没有当前用户，直接返回

    console.log(`用户 ${username} 正在请求退出登录`);

    try {
        const response = await fetch('/api/logout', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            console.log("后端登出成功");
            currentUsername = null; //  清除全局变量
            chatEventListenersAttached = false; //  重置监听器标志 
            document.body.classList.remove('logged-in'); // 移除标记类
            authOverlay.classList.add('active'); // 显示登录/注册层
            loginForm.style.display = 'block'; // 确保显示的是登录表单
            registerForm.style.display = 'none';
            closeUserInfoPopup(); // 关闭用户信息弹窗
            document.getElementById('chatArea').innerHTML = ''; // 清空聊天区域
            historyList.innerHTML = ''; // 清空历史列表
            fileList.innerHTML = ''; //  清空文件列表 
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
            loadFiles(); // 加载文件列表 
        } else {
            console.log("后端验证：用户未登录，显示登录界面");
            currentUsername = null; // **修改**: 确保全局变量为空
            document.body.classList.remove('logged-in');
            authOverlay.classList.add('active');
            loginForm.style.display = 'block';
            registerForm.style.display = 'none';
            historyList.innerHTML = ''; // 清空可能存在的旧历史记录
            fileList.innerHTML = ''; // 清空文件列表 
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
        userInfoContent.textContent = getText('accountPrefix') + currentUsername; // 设置弹窗内容，使用多语言
    } else {
        userAvatar.textContent = ''; // 未登录则清空
        userInfoContent.textContent = ''; // 清空弹窗内容
    }
}

// 显示用户信息弹窗 - 
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
    backToSettingsButton.style.display = 'none'; // 确保返回按钮隐藏
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

    // 处理语言切换 - 不需要显示内容区域，直接切换语言
    if (optionId === 'toggleLanguage') {
        toggleLanguage();
        return; // 直接结束函数
    }

    settingOptions.style.display = 'none'; // 修正：隐藏选项列表容器
    settingContentDisplay.style.display = 'block'; // 显示内容区域
    backToSettingsButton.style.display = 'inline-block'; // 显示返回按钮

    settingContentDisplay.innerHTML = `<p>${getText('loadingContent')}</p>`; // 显示加载提示

    if (optionId === 'checkUpdate') {
        settingContentDisplay.innerHTML = `<p>${getText('versionUpdated')}</p>`; // 显示提示信息
        return; // 直接结束函数，不执行后续的 fetch
    }
    try {
        const response = await fetch(`/api/setting?topic=${encodeURIComponent(optionId)}`); // 不需要区分选项，直接调用
        const data = await response.json();

        if (data.success && data.messages) {
            console.log("成功获取设置内容");

            settingContentDisplay.innerHTML = marked.parse(data.messages);

        } else {
            console.error("获取设置内容失败:", data.error);
            // 显示错误信息
            settingContentDisplay.innerHTML = `<p style="color: red;">${getText('loadFailed')}${data.error || '未知错误'}</p>`;
        }
    } catch (error) {
        console.error("处理设置选项时出错:", error);
        // 显示网络或请求错误信息
        settingContentDisplay.innerHTML = `<p style="color: red;">${getText('loadError')}${error.message}</p>`;
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
    //  防止重复绑定 
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

    chatEventListenersAttached = true; //  设置标志 
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

    // 应用保存的语言设置
    applyLanguage();

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
    const sendButton = document.getElementById('sendButton'); // 获取发送按钮
    const message = userInput.value.trim();

    if (!message) {
        return;
    }

    if (!currentUsername) {
        showError("请先登录再发送消息！");
        return;
    }

    //  核心修改：如果会话ID不存在，则先在后端获取一个 
    if (!currentSessionId) {
        console.log("检测到新对话（无会话ID），正在后端获取ID...");
        try {
            const response = await fetch('/api/new_chat', { method: 'POST' });
            const data = await response.json();
            if (data.success) {
                currentSessionId = data.new_session_id; // 更新全局ID
                isNewSessionPendingDisplay = true; // 标记为待显示
                console.log(`新会话ID已获取: ${currentSessionId}`);
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

    // 核心修改：如果是一个待显示的新会话，立即在UI上创建临时条目
    if (isNewSessionPendingDisplay) {
        addTemporarySessionToUI(currentSessionId, message);
        isNewSessionPendingDisplay = false; // 重置标志，防止重复创建
    }

    addMessage('user', message);
    userInput.value = ''; // 清空输入框

    userInput.disabled = true;
    sendButton.disabled = true;

    // 创建思考过程元素（独立的气泡和详情面板）
    const thinkingElements = addThinkingMessage();

    try {
        // 使用流式端点
        const response = await fetch('/api/send_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                message: message, 
                username: currentUsername,
                session_id: currentSessionId 
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }

        // 使用ReadableStream读取SSE数据
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // 解析SSE消息（以 \n\n 分隔）
            const messages = buffer.split('\n\n');
            buffer = messages.pop(); // 保留不完整的消息
            
            for (const msg of messages) {
                if (msg.startsWith('data: ')) {
                    try {
                        const eventData = JSON.parse(msg.substring(6)); // 去掉 "data: " 前缀
                        handleStreamEvent(eventData, thinkingElements);
                    } catch (e) {
                        console.warn('解析SSE事件失败:', msg, e);
                    }
                }
            }
        }
        
        // 加载历史记录
        loadHistory();
        
    } catch (error) {
        console.error("发送消息时出错:", error);
        
        // 移除思考过程的两个独立元素
        if (thinkingElements.bubble && thinkingElements.bubble.parentNode) {
            thinkingElements.bubble.parentNode.removeChild(thinkingElements.bubble);
        }
        if (thinkingElements.detailContainer && thinkingElements.detailContainer.parentNode) {
            thinkingElements.detailContainer.parentNode.removeChild(thinkingElements.detailContainer);
        }
        
        showError('发送消息时发生网络错误。');
    } finally {
        //  无论成功或失败，都重新启用输入和发送按钮 
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus(); // 重新聚焦到输入框，方便用户继续输入
    }
}


/**
 * 创建思考过程气泡
 */
function addThinkingMessage() {
    // 创建简洁的思考提示气泡
    const bubble = document.createElement('div');
    bubble.className = 'message ai-message thinking-bubble';
    
    const header = document.createElement('div');
    header.className = 'thinking-header';
    
    const text = document.createElement('span');
    text.className = 'thinking-text';
    text.textContent = getText('thinking');
    
    const dots = document.createElement('span');
    dots.className = 'thinking-dots';
    dots.textContent = '...';
    
    // 添加展开/收起图标
    const expandIcon = document.createElement('span');
    expandIcon.className = 'expand-icon';
    expandIcon.textContent = '▶';  // 默认收起状态
    
    header.appendChild(text);
    header.appendChild(dots);
    header.appendChild(expandIcon);
    
    bubble.appendChild(header);
    
    // 创建独立的详情面板容器
    const detailContainer = document.createElement('div');
    detailContainer.className = 'message ai-message thinking-detail-container';
    detailContainer.style.display = 'none';  // 默认隐藏
    
    const detail = document.createElement('div');
    detail.className = 'thinking-detail';
    
    detailContainer.appendChild(detail);
    
    // 添加点击事件来切换详情面板显示（使用闭包访问detailContainer）
    header.onclick = () => toggleThinkingDetail(detailContainer, expandIcon);
    
    // 添加到聊天区域（两个独立的元素）
    chatArea.appendChild(bubble);
    chatArea.appendChild(detailContainer);
    chatArea.scrollTop = chatArea.scrollHeight;
    
    return { bubble,detail, detailContainer };
}

/**
 * 切换思考过程详情显示（适配独立布局）
 */
function toggleThinkingDetail(detailContainer, expandIcon) {
    if (detailContainer.style.display === 'none') {
        detailContainer.style.display = 'block';
        expandIcon.textContent = '▼';
        // 滚动到底部以确保详情面板可见
        setTimeout(() => {
            chatArea.scrollTop = chatArea.scrollHeight;
        }, 100);
    } else {
        detailContainer.style.display = 'none';
        expandIcon.textContent = '▶';
    }
}



/**
 * 处理流式事件
 */
function handleStreamEvent(eventData, thinkingElements) {
    const eventType = eventData.type;
    
    console.log('[SSE Event]:', eventType, eventData);
    
    switch (eventType) {
        case 'node_start':
            handleNodeStart(eventData, thinkingElements);
            break;
            
        case 'node_end':
            handleNodeEnd(eventData, thinkingElements);
            break;
            
        case 'final_result':
            handleFinalResult(eventData, thinkingElements);
            break;
            
        case 'interrupt':
            handleInterrupt(eventData, thinkingElements);
            break;
            
        case 'error':
            handleStreamError(eventData, thinkingElements);
            break;
            
        default:
            console.warn('未知的事件类型:', eventType);
    }
}

/**
 * 处理节点开始事件
 */
function handleNodeStart(eventData, thinkingElements) {
    const { node_name, node_desc } = eventData;
    
    // 更新简洁视图的文字（在 bubble 中）
    const thinkingText = thinkingElements.bubble.querySelector('.thinking-text');
    if (thinkingText) {
        thinkingText.textContent = node_desc + getText('thinkingWait');
    }

    // 在独立的详情面板中添加步骤项
    const detail = thinkingElements.detail;
    const stepItem = document.createElement('div');
    stepItem.className = 'step-item in-progress';
    stepItem.id = `step-${node_name}`;

    const statusIcon = document.createElement('span');
    statusIcon.className = 'step-status';
    statusIcon.textContent = '▶';

    const stepName = document.createElement('span');
    stepName.className = 'step-name';
    stepName.textContent = node_desc;

    const stepTime = document.createElement('span');
    stepTime.className = 'step-time';
    stepTime.textContent = getText('inProgress');
    
    stepItem.appendChild(statusIcon);
    stepItem.appendChild(stepName);
    stepItem.appendChild(stepTime);
    
    detail.appendChild(stepItem);
    
    // 滚动到底部
    chatArea.scrollTop = chatArea.scrollHeight;
}

/**
 * 处理节点结束事件
 */
function handleNodeEnd(eventData, thinkingElements) {
    const { node_name, duration } = eventData;
    
    // 在独立的详情面板中查找并更新步骤项
    const stepItem = thinkingElements.detail.querySelector(`#step-${node_name}`);
    if (stepItem) {
        stepItem.className = 'step-item completed';
        
        const statusIcon = stepItem.querySelector('.step-status');
        if (statusIcon) {
            statusIcon.textContent = '✓';
        }
        
        const stepTime = stepItem.querySelector('.step-time');
        if (stepTime) {
            stepTime.textContent = `${duration}s`;
        }
    }
}

/**
 * 处理最终结果事件
 */
function handleFinalResult(eventData, thinkingElements) {
    const { data } = eventData;
    
    // 移除思考过程的两个独立元素
    if (thinkingElements.bubble && thinkingElements.bubble.parentNode) {
        thinkingElements.bubble.parentNode.removeChild(thinkingElements.bubble);
    }
    if (thinkingElements.detailContainer && thinkingElements.detailContainer.parentNode) {
        thinkingElements.detailContainer.parentNode.removeChild(thinkingElements.detailContainer);
    }
    
    // 添加最终回复
    addMessage('ai', data);
}

/**
 * 处理interrupt事件
 */
function handleInterrupt(eventData, thinkingElements) {
    const { message } = eventData;
    
    // 移除思考过程的两个独立元素
    if (thinkingElements.bubble && thinkingElements.bubble.parentNode) {
        thinkingElements.bubble.parentNode.removeChild(thinkingElements.bubble);
    }
    if (thinkingElements.detailContainer && thinkingElements.detailContainer.parentNode) {
        thinkingElements.detailContainer.parentNode.removeChild(thinkingElements.detailContainer);
    }
    
    // 添加需要用户输入的消息
    addMessage('ai', {
        type: 'human_input_required',
        summary: message
    });
}

/**
 * 处理流式传输错误
 */
function handleStreamError(eventData, thinkingElements) {
    const { message } = eventData;
    
    // 移除思考过程的两个独立元素
    if (thinkingElements.bubble && thinkingElements.bubble.parentNode) {
        thinkingElements.bubble.parentNode.removeChild(thinkingElements.bubble);
    }
    if (thinkingElements.detailContainer && thinkingElements.detailContainer.parentNode) {
        thinkingElements.detailContainer.parentNode.removeChild(thinkingElements.detailContainer);
    }
    
    // 显示错误消息
    addMessage('ai', {
        type: 'text',
        summary: `错误：${message}`
    });
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
            isNewSessionPendingDisplay = true; //  标记这个新会话等待用户输入后在UI显示
            
            addMessage('ai', getText('newChatGreeting'));
            document.getElementById('userInput').focus();

            // 注意：此时不调用loadHistory，因为新会话还不在数据库里
            // await loadHistory(); // 重新加载历史列表以显示新会话
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
        //  修改：使用POST方法，并将数据放在body中 
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
            return true; //  修改：仅返回成功状态 
        } else {
            console.error("更新会话标题失败:", data.error);
            showError(`更新标题失败: ${data.error || '未知错误'}`);
            return false; //  修改：仅返回失败状态 
        }
    } catch (error) {
        console.error("更新会话标题时发生网络错误:", error);
        showError("更新标题时发生网络错误。");
        return false; //  修改：仅返回失败状态 
    }
}

// 为标题添加可编辑监听器的辅助函数 
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

//  会话标题可编辑 
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

//  在前端临时添加一个会话条目，以实现即时反馈 
function addTemporarySessionToUI(sessionId, title) {
    console.log("正在前端临时添加新会话条目以提高UI响应性...");

    // 移除"无记录"的提示消息
    const emptyMessage = historyList.querySelector('.history-empty-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }

    // 创建一个新的、简化的会话条目DOM元素
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.setAttribute('data-session-id', sessionId);
    historyItem.style.opacity = '0.7'; // 让临时条目半透明以作区分

    const itemContent = document.createElement('div');
    itemContent.className = 'history-item-content';

    const sessionInfo = document.createElement('div');
    sessionInfo.className = 'session-info';

    const timeDiv = document.createElement('div');
    timeDiv.className = 'session-time';
    timeDiv.textContent = getText('justNow');

    const previewDiv = document.createElement('div');
    previewDiv.className = 'preview-text';
    const previewText = title.length > 25 ? title.substring(0, 25) + '...' : title;
    previewDiv.textContent = previewText;

    sessionInfo.appendChild(timeDiv);
    itemContent.appendChild(sessionInfo);
    itemContent.appendChild(previewDiv);
    historyItem.appendChild(itemContent);

    // 将新条目添加到历史记录列表的顶部
    historyList.prepend(historyItem);
}

// 加载历史记录
async function loadHistory() {
    if (!currentUsername) {
        historyList.innerHTML = `<p class="history-empty-message">${getText('pleaseLoginHistory')}</p>`;
        return;
    }
    console.log(`为用户 ${currentUsername} 加载历史会话...`);

    try {
        const response = await fetch(`/api/sessions`);
        if (!response.ok) {
            if (response.status === 401) {
                // 如果是401未授权，可能是会话过期，可以提示用户重新登录
                 historyList.innerHTML = `<p class="history-empty-message">${getText('sessionExpired')}</p>`;
                 handleLogout(); // 可以选择直接触发登出流程
                 return;
            }
            throw new Error(`服务器错误: ${response.status}`);
        }
        const sessions = await response.json();

        historyList.innerHTML = ''; // 清空旧列表

        if (Object.keys(sessions).length === 0) {
            historyList.innerHTML = `<p class="history-empty-message">${getText('noHistoryMessage')}</p>`;
        } else {
            //  用于跟踪当前打开的滑动项 
            let currentlyOpenItem = null;

            // 重建历史记录部分
            sessions.forEach(session => {
                const session_id = session[0];
                const info = session[1];

                //  核心修改：创建新的DOM结构以支持滑动 
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.setAttribute('data-session-id', session_id);

                // 删除按钮容器
                const swipeActions = document.createElement('div');
                swipeActions.className = 'swipe-actions';

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'delete-btn';
                deleteBtn.textContent = getText('deleteBtn');
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

                //  滑动逻辑 
                let isDragging = false;
                let startX = 0;
                let currentX = 0;
                let hasMoved = false; //  关键：用于区分拖拽和点击的标志 
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
                    hasMoved = false; //  关键：每次开始拖拽时，重置标志 
                    isDragging = true;
                    startX = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
                    itemContent.style.transition = 'none'; // 拖动时禁用平滑过渡
                };

                const onDragMove = (e) => {
                    if (!isDragging) return;

                    const moveX = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
                    //  关键：如果移动超过一个微小距离，就确认为拖拽 
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
                    
                    //  核心修改：使用 getComputedStyle 来可靠地获取当前位置 
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
                    //  关键：如果是拖拽，则不执行点击逻辑 
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

//  处理会话删除的函数 
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
                     historyList.innerHTML = `<p class="history-empty-message">${getText('noHistoryMessage')}</p>`;
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
            currentSessionId = sessionId; // < 核心修改：更新全局会话ID
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

    //  检查会话ID
    if (!currentSessionId) {
        showError("没有活动的会话，无法上传文件。请新建一个对话或加载历史会话。");
        // 恢复按钮状态
        if (uploadCsvButton) {
            uploadCsvButton.textContent = getText('upload');
            uploadCsvButton.disabled = false;
        }
        event.target.value = null; // 清除文件选择
        return;
    }
    // -

    // 显示上传开始的用户消息和AI加载动画
    addMessage('user', getText('uploadingFile') + file.name);
    const loadingMessageElement = addMessage('ai', '', true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', currentSessionId);

    if (uploadCsvButton) {
        uploadCsvButton.textContent = getText('uploading');
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
            addMessage('ai', getText('fileReceived') + file.name + '\n\n' + data.message + getText('askCausalAnalysis'));
            loadFiles(); //  刷新文件列表
        } else {
            // 显示错误的AI回复消息
            addMessage('ai', getText('uploadFailed') + (data.error || '未知错误'));
            showError(data.error || '文件上传失败。');
        }
    } catch (error) {
        console.error("CSV Upload error:", error);

        // 移除加载动画
        if (loadingMessageElement && loadingMessageElement.parentNode) {
            loadingMessageElement.parentNode.removeChild(loadingMessageElement);
        }

        // 显示网络错误的AI回复
        addMessage('ai', getText('networkError'));
        showError('上传文件时发生网络错误。');
    } finally {
        if (uploadCsvButton) {
            uploadCsvButton.textContent = getText('upload');
            uploadCsvButton.disabled = false;
        }
        event.target.value = null;
    }
}

//  渲染因果图表的函数 
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

        if (sender === 'ai' && typeof messageData === 'object' && messageData !== null) {
            // 处理AI的结构化响应
            if (messageData.type === 'causal_graph' && messageData.data) {
                // 1. 添加总结文本（如果存在）
                // isReportLayout: 只要是因果图类型，就按报告样式展示；若显式标记 layout === 'report' 也同样视为报告
                const isReportLayout = messageData.layout === 'report' || messageData.type === 'causal_graph';
                let reportContainer = contentElement;
                if (isReportLayout) {
                    const reportWrapper = document.createElement('div');
                    reportWrapper.classList.add('causal-report');
                    contentElement.appendChild(reportWrapper);
                    reportContainer = reportWrapper;
                }

                if (messageData.summary) {
                    const summaryDiv = document.createElement('div');
                    summaryDiv.innerHTML = marked.parse(messageData.summary);
                    reportContainer.appendChild(summaryDiv);
                }

                // 2. 创建并渲染因果图
                // 生成唯一id，防止id重复
                const graphContainerId = `graph-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                const graphContainer = document.createElement('div');
                graphContainer.id = graphContainerId;
                graphContainer.classList.add('causal-graph-container'); // 用于样式
                reportContainer.appendChild(graphContainer);

                // 使用 setTimeout 确保元素已添加到 DOM 中
                // vis.js 需要一个已挂载的容器来进行初始化
                setTimeout(() => {
                    renderCausalGraph(graphContainerId, messageData.data);
                }, 100);

            } else if (messageData.summary) {
                // 对于其他类型的结构化响应（例如只有 'type': 'text'），只显示总结
                if (messageData.layout === 'report') {
                    const reportWrapper = document.createElement('div');
                    reportWrapper.classList.add('causal-report');
                    reportWrapper.innerHTML = marked.parse(messageData.summary);
                    contentElement.appendChild(reportWrapper);
                } else {
                    contentElement.innerHTML = marked.parse(messageData.summary);
                }
            } else {
                // 如果对象无法识别，则作为字符串显示以供调试
                contentElement.textContent = JSON.stringify(messageData, null, 2);
            }
        } else {
            // 对于用户消息（总是字符串）和旧的纯文本AI消息
            contentElement.innerHTML = marked.parse(messageData.toString());
        }

    }

    // 直接添加内容元素，不需要头像
    messageElement.appendChild(contentElement);

    chatArea.appendChild(messageElement);
    chatArea.scrollTop = chatArea.scrollHeight; // 自动滚动到底部

    // 返回消息元素，以便后续可以移除（例如加载动画）
    return messageElement;
}

//  加载文件列表的函数
async function loadFiles() {
    if (!currentUsername) {
        if (fileList) fileList.innerHTML = `<p class="files-empty-message">${getText('pleaseLoginFiles')}</p>`;
        return;
    }
    console.log(`为用户 ${currentUsername} 加载文件列表...`);
    if (!fileList) return;

    try {
        const response = await fetch(`/api/files`);
        if (!response.ok) {
            if (response.status === 401) {
                 fileList.innerHTML = `<p class="files-empty-message">${getText('sessionExpired')}</p>`;
                 return;
            }
            throw new Error(`服务器错误: ${response.status}`);
        }
        const files = await response.json();

        fileList.innerHTML = ''; // 清空旧列表

        if (Object.keys(files).length === 0) {
            fileList.innerHTML = `<p class="files-empty-message">${getText('noFilesMessage')}</p>`;
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
                deleteBtn.textContent = getText('deleteBtn');
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

                //  滑动逻辑 (与 history item 相同) 
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
                        userInput.value += `请对文件"${info.preview}"进行因果分析`;
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

//  处理文件删除的函数 
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
                     fileList.innerHTML = `<p class="files-empty-message">${getText('noFilesMessage')}</p>`;
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
