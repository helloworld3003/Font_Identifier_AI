document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const imageUpload = document.getElementById('imageUpload');
    const backendUrlInput = document.getElementById('backendUrl');
    const customTextInput = document.getElementById('customText');
    const startBtn = document.getElementById('startBtn');
    const themeToggle = document.getElementById('themeToggle');
    
    const terminalToggle = document.getElementById('terminalToggle');
    const terminalContainer = document.getElementById('terminalContainer');
    const terminalLogs = document.getElementById('terminalLogs');
    
    const resultsSection = document.getElementById('resultsSection');
    const queryImagePreview = document.getElementById('queryImagePreview');
    const matchesContainer = document.getElementById('matchesContainer');
    const apiStatusElement = document.getElementById('apiStatus');

    let uploadedFile = null;

    // --- API Status Polling ---
    async function checkApiStatus() {
        const url = backendUrlInput.value.trim().replace(/\/$/, "");
        try {
            const pingPromise = fetch(`${url}/`, { method: 'GET' }).catch(() => null);
            const timeoutPromise = new Promise(resolve => setTimeout(() => resolve('timeout'), 5000));
            const response = await Promise.race([pingPromise, timeoutPromise]);
            
            if (response !== 'timeout' && response && response.ok) {
                apiStatusElement.innerHTML = '<span style="color: #10b981;">🟢</span> Online & Ready';
                apiStatusElement.style.borderColor = '#10b981';
                apiStatusElement.style.color = '#10b981';
            } else {
                apiStatusElement.innerHTML = '<span style="color: #ef4444;">🔴</span> Offline or Building...';
                apiStatusElement.style.borderColor = '#ef4444';
                apiStatusElement.style.color = '#ef4444';
            }
        } catch (e) {
            apiStatusElement.innerHTML = '<span style="color: #ef4444;">🔴</span> Offline or Building...';
            apiStatusElement.style.borderColor = '#ef4444';
            apiStatusElement.style.color = '#ef4444';
        }
    }

    // Initial check and interval
    checkApiStatus();
    setInterval(checkApiStatus, 10000); // Check every 10 seconds
    backendUrlInput.addEventListener('change', checkApiStatus);

    // --- Sidebar Toggle ---
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        sidebarToggle.textContent = sidebar.classList.contains('collapsed') ? '▶' : '◀';
    });

    // --- Theme Toggle ---
    themeToggle.addEventListener('click', () => {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        document.body.setAttribute('data-theme', isDark ? 'light' : 'dark');
    });

    // --- Expander Toggle ---
    terminalToggle.addEventListener('click', () => {
        terminalContainer.classList.toggle('hidden');
        const icon = terminalToggle.querySelector('.expander-icon');
        icon.textContent = terminalContainer.classList.contains('hidden') ? '▶' : '▼';
    });

    // --- Terminal Logger ---
    function logToTerminal(message, type = 'info') {
        const span = document.createElement('span');
        span.className = `log-${type}`;
        span.textContent = `[${new Date().toLocaleTimeString()}] ${message}\n`;
        terminalLogs.appendChild(span);
        terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }

    // --- Image Upload ---
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                queryImagePreview.src = e.target.result;
                document.getElementById('sidebarImagePreview').style.display = 'block';
            };
            reader.readAsDataURL(file);
            logToTerminal(`Loaded image: ${file.name}`, 'info');
        }
    });

    customTextInput.addEventListener('input', (e) => {
        const newText = e.target.value || "Sphinx of black quartz, judge my vow.";
        document.querySelectorAll('.match-preview').forEach(el => {
            el.textContent = newText;
        });
    });

    // --- Dynamic Font Loading ---
    function loadDynamicFont(fontName, fontUrl) {
        // Create a unique CSS class and @font-face rule
        const safeFontName = fontName.replace(/[^a-zA-Z0-9]/g, '');
        const fontFaceRule = `
            @font-face {
                font-family: '${safeFontName}';
                src: url('${fontUrl}') format('truetype');
            }
        `;
        const style = document.createElement('style');
        style.appendChild(document.createTextNode(fontFaceRule));
        document.head.appendChild(style);
        
        return safeFontName;
    }

    // --- Start Pipeline ---
    startBtn.addEventListener('click', async () => {
        if (!uploadedFile) {
            alert('Please upload an image first!');
            return;
        }

        const backendUrl = backendUrlInput.value.replace(/\/$/, ""); // Remove trailing slash
        if (!backendUrl) {
            alert('Please enter a backend API URL!');
            return;
        }

        startBtn.disabled = true;
        startBtn.textContent = 'Processing...';
        resultsSection.classList.add('hidden');
        matchesContainer.innerHTML = '';
        
        terminalContainer.classList.remove('hidden');
        terminalToggle.querySelector('.expander-icon').textContent = '▼';

        logToTerminal('Initializing AI Inference Pipeline...', 'info');
        logToTerminal(`Target Backend: ${backendUrl}`, 'info');
        
        const formData = new FormData();
        formData.append('file', uploadedFile);

        try {
            logToTerminal('Pinging backend API...', 'info');
            
            // Ping the root endpoint to check if the server is awake
            const pingPromise = fetch(`${backendUrl}/`, { method: 'GET' }).catch(() => null);
            
            // Wait 2 seconds to see if it responds quickly
            const timeoutPromise = new Promise(resolve => setTimeout(() => resolve('timeout'), 2000));
            const firstResult = await Promise.race([pingPromise, timeoutPromise]);
            
            if (firstResult === 'timeout') {
                logToTerminal('The free API instance is currently spinning up from sleep...', 'error');
                logToTerminal('This initial wake-up can take up to 50 seconds. Please do not refresh...', 'info');
                await pingPromise; // wait for the ping to actually finish
            }
            
            logToTerminal('✅ Backend API is awake and connected!', 'success');
            logToTerminal('Uploading image tensor to PyTorch backend...', 'info');
            
            const response = await fetch(`${backendUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server responded with ${response.status}`);
            }

            const data = await response.json();
            logToTerminal('✅ FAISS Index query complete! Matches found.', 'success');
            
            renderResults(data.results, backendUrl);
            resultsSection.classList.remove('hidden');
            
        } catch (error) {
            logToTerminal(`CRITICAL ERROR: ${error.message}`, 'error');
            console.error(error);
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Identification Pipeline';
        }
    });

    function renderResults(results, backendUrl) {
        const textToPreview = customTextInput.value || "Sphinx of black quartz, judge my vow.";
        
        results.forEach((match, index) => {
            logToTerminal(`Match ${match.rank}: ${match.font_name} (${match.confidence.toFixed(2)}%)`, 'info');
            
            // Build the font URL to temporarily download it from the backend
            const fontUrl = `${backendUrl}/font/${encodeURIComponent(match.filename)}`;
            
            // Inject @font-face
            const safeFontFamily = loadDynamicFont(match.font_name, fontUrl);

            // Create Card
            const card = document.createElement('div');
            card.className = 'match-card';
            
            card.innerHTML = `
                <div class="match-header">
                    <div class="match-rank">#${match.rank} - ${match.font_name}</div>
                    <div class="match-confidence">${match.confidence.toFixed(2)}% Match</div>
                </div>
                <!-- The style attribute applies the dynamically loaded font -->
                <div class="match-preview" style="font-family: '${safeFontFamily}', sans-serif;">
                    ${textToPreview}
                </div>
            `;
            
            matchesContainer.appendChild(card);
        });
    }
});
