// ============================================
// NEURAFORM - Frontend JavaScript
// ============================================

// Configuration
const CONFIG = {
    // Change this to your backend URL
    // Option 1: Colab ngrok URL (free)
    // Option 2: Your own server
    // Option 3: Hugging Face Spaces
    API_URL: 'YOUR_BACKEND_URL_HERE', // e.g., 'https://xxxx.ngrok.io'
    
    // Default generation settings
    temperature: 0.8,
    maxTokens: 200,
    topK: 50,
    topP: 0.9
};

// State
let isGenerating = false;
let conversationHistory = [];

// DOM Elements
const elements = {
    messages: document.getElementById('messages'),
    userInput: document.getElementById('userInput'),
    sendButton: document.getElementById('sendButton'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    charCount: document.getElementById('charCount'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    settingsPanel: document.getElementById('settingsPanel'),
    welcomeTime: document.getElementById('welcomeTime')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    init();
});

async function init() {
    // Set welcome time
    elements.welcomeTime.textContent = formatTime(new Date());
    
    // Setup input handlers
    setupInputHandlers();
    
    // Check API connection
    await checkConnection();
    
    // Hide loading overlay
    setTimeout(() => {
        elements.loadingOverlay.classList.add('hidden');
    }, 1000);
}

function setupInputHandlers() {
    // Auto-resize textarea
    elements.userInput.addEventListener('input', () => {
        elements.userInput.style.height = 'auto';
        elements.userInput.style.height = Math.min(elements.userInput.scrollHeight, 150) + 'px';
        
        // Update character count
        const count = elements.userInput.value.length;
        elements.charCount.textContent = count;
        
        if (count > 900) {
            elements.charCount.style.color = 'var(--error)';
        } else if (count > 700) {
            elements.charCount.style.color = 'var(--warning)';
        } else {
            elements.charCount.style.color = '';
        }
    });
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function checkConnection() {
    try {
        updateStatus('connecting');
        
        const response = await fetch(`${CONFIG.API_URL}/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const data = await response.json();
            updateStatus('online');
            
            if (data.model_info) {
                document.getElementById('modelInfo').textContent = 
                    `Model: ${data.model_info}`;
            }
        } else {
            updateStatus('offline');
        }
    } catch (error) {
        console.error('Connection error:', error);
        updateStatus('offline');
        
        // Show demo mode message
        showDemoModeMessage();
    }
}

function updateStatus(status) {
    elements.statusDot.className = 'status-dot';
    
    switch (status) {
        case 'online':
            elements.statusDot.classList.add('online');
            elements.statusText.textContent = 'Online';
            break;
        case 'offline':
            elements.statusDot.classList.add('offline');
            elements.statusText.textContent = 'Offline';
            break;
        case 'connecting':
            elements.statusText.textContent = 'Connecting...';
            break;
        case 'generating':
            elements.statusText.textContent = 'Generating...';
            break;
    }
}

function showDemoModeMessage() {
    addMessage('bot', `
        ⚠️ **Demo Mode Active**
        
        I couldn't connect to the Neuraform backend. To use the full AI:
        
        1. Run the backend on Google Colab (free!)
        2. Copy the ngrok URL
        3. Update the API_URL in script.js
        
        For now, I'll respond with pre-written examples.
    `);
}

async function sendMessage() {
    const text = elements.userInput.value.trim();
    
    if (!text || isGenerating) return;
    if (text.length > 1000) {
        alert('Message too long! Maximum 1000 characters.');
        return;
    }
    
    // Add user message
    addMessage('user', text);
    
    // Clear input
    elements.userInput.value = '';
    elements.userInput.style.height = 'auto';
    elements.charCount.textContent = '0';
    
    // Add to history
    conversationHistory.push({ role: 'user', content: text });
    
    // Generate response
    await generateResponse(text);
}

function sendQuickPrompt(text) {
    elements.userInput.value = text;
    sendMessage();
}

async function generateResponse(prompt) {
    isGenerating = true;
    elements.sendButton.disabled = true;
    updateStatus('generating');
    
    // Add typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch(`${CONFIG.API_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: CONFIG.maxTokens,
                temperature: CONFIG.temperature,
                top_k: CONFIG.topK,
                top_p: CONFIG.topP,
                history: conversationHistory.slice(-10) // Last 10 messages
            })
        });
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.text) {
            addMessage('bot', data.text);
            conversationHistory.push({ role: 'assistant', content: data.text });
            
            // Update stats if available
            if (data.tokens_generated) {
                console.log(`Generated ${data.tokens_generated} tokens in ${data.time_taken}s`);
            }
        } else if (data.error) {
            addMessage('bot', `❌ Error: ${data.error}`);
        }
        
        updateStatus('online');
        
    } catch (error) {
        console.error('Generation error:', error);
        removeTypingIndicator(typingId);
        
        // Fallback to demo responses
        const demoResponse = getDemoResponse(prompt);
        addMessage('bot', demoResponse);
        
        updateStatus('offline');
    }
    
    isGenerating = false;
    elements.sendButton.disabled = false;
}

function addMessage(role, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role === 'user' ? 'user-message' : 'bot-message'}`;
    
    const avatar = role === 'user' ? '👤' : '🧠';
    const name = role === 'user' ? 'You' : 'Neuraform';
    
    // Parse markdown-like formatting
    const formattedText = formatText(text);
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-name">${name}</span>
                <span class="message-time">${formatTime(new Date())}</span>
            </div>
            <div class="message-text">${formattedText}</div>
        </div>
    `;
    
    elements.messages.appendChild(messageDiv);
    scrollToBottom();
}

function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message bot-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">🧠</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    elements.messages.appendChild(typingDiv);
    scrollToBottom();
    
    return id;
}

function removeTypingIndicator(id) {
    const typingDiv = document.getElementById(id);
    if (typingDiv) {
        typingDiv.remove();
    }
}

function formatText(text) {
    // Bold: **text**
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    // Italic: *text*
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    
    // Code: `code`
    text = text.replace(/`(.+?)`/g, '<code>$1</code>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    // Paragraphs
    text = '<p>' + text.replace(/<br><br>/g, '</p><p>') + '</p>';
    
    return text;
}

function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

function scrollToBottom() {
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

// Settings
function toggleSettings() {
    elements.settingsPanel.classList.toggle('open');
}

function updateSetting(setting, value) {
    switch (setting) {
        case 'temp':
            CONFIG.temperature = parseFloat(value);
            document.getElementById('tempValue').textContent = value;
            break;
        case 'tokens':
            CONFIG.maxTokens = parseInt(value);
            document.getElementById('tokensValue').textContent = value;
            break;
        case 'topk':
            CONFIG.topK = parseInt(value);
            document.getElementById('topkValue').textContent = value;
            break;
        case 'topp':
            CONFIG.topP = parseFloat(value);
            document.getElementById('toppValue').textContent = value;
            break;
    }
}

// Demo Responses (when API is offline)
function getDemoResponse(prompt) {
    const lowerPrompt = prompt.toLowerCase();
    
    const responses = {
        'quantum': `**Quantum Physics** is the branch of physics that studies the behavior of matter and energy at the smallest scales.

Key concepts include:
• **Wave-particle duality**: Particles can behave as both waves and particles
• **Uncertainty principle**: We cannot know both position and momentum precisely
• **Superposition**: Particles can exist in multiple states simultaneously
• **Entanglement**: Particles can be connected across distances

This field revolutionized our understanding of nature at the atomic level.`,

        'universe': `**The Universe** is vast and awe-inspiring!

Here are some amazing facts:
• The universe is approximately **13.8 billion years old**
• It contains over **200 billion galaxies**
• The observable universe is about **93 billion light-years** in diameter
• **Dark matter** makes up about 27% of the universe
• **Dark energy** makes up about 68%, driving cosmic expansion

The Big Bang theory explains how it all began from an incredibly hot, dense state.`,

        'ai': `**Artificial Intelligence** is the simulation of human intelligence by machines.

Key components:
• **Machine Learning**: Systems that learn from data
• **Neural Networks**: Inspired by biological brains
• **Deep Learning**: Neural networks with many layers
• **Natural Language Processing**: Understanding human language

AI is transforming industries from healthcare to transportation, and models like me are built using transformer architectures trained on vast amounts of text data.`,

        'evolution': `**Evolution** is the process of change in living organisms over generations.

Key principles:
• **Natural Selection**: Organisms better adapted to their environment survive and reproduce
• **Genetic Variation**: Mutations create diversity in populations
• **Inheritance**: Traits are passed from parents to offspring
• **Speciation**: New species form when populations become isolated

Charles Darwin's theory of evolution by natural selection revolutionized biology and our understanding of life on Earth.`,

        'default': `I'm **Neuraform**, an AI language model built from scratch!

I've been trained on knowledge spanning:
• 🔬 Science & Physics
• 🧬 Biology & Medicine
• 📜 History & Geography
• 💻 Technology & Computing
• 📚 Literature & Philosophy
• And much more!

Ask me about any topic, and I'll share what I know. Note: I'm currently in demo mode. Connect the backend for full AI generation!`
    };
    
    // Find matching response
    for (const [key, response] of Object.entries(responses)) {
        if (key !== 'default' && lowerPrompt.includes(key)) {
            return response;
        }
    }
    
    return responses.default;
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, sendMessage, formatText };
      }
