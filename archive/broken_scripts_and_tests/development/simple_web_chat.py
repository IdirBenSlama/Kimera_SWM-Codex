#!/usr/bin/env python3
"""Simple KIMERA Web Chat Interface"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import logging
import requests # Using requests to call the main KIMERA API
from backend.engines.universal_translator_hub import UniversalTranslatorHub # Import the real hub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🌟 KIMERA Chat</title>
    <style>
        body { 
            font-family: Arial; 
            background: #1a1a1a; 
            color: white; 
            margin: 0; 
            padding: 20px; 
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            padding: 20px; 
            background: rgba(0,255,136,0.1); 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .header h1 {
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .chat-messages { 
            height: 400px; 
            overflow-y: auto; 
            background: rgba(255,255,255,0.05); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .message { 
            margin-bottom: 15px; 
            padding: 10px; 
            border-radius: 8px; 
            max-width: 80%;
        }
        .user { 
            background: #0066cc; 
            text-align: right; 
            margin-left: auto;
        }
        .kimera { 
            background: #cc0066; 
            margin-right: auto;
        }
        .system { 
            background: rgba(255,255,255,0.1); 
            text-align: center; 
            margin: 0 auto;
            max-width: 60%;
        }
        .input-area { 
            display: flex; 
            gap: 10px; 
            align-items: flex-end;
        }
        select { 
            background: rgba(255,255,255,0.1); 
            color: white; 
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 5px;
            padding: 10px;
        }
        textarea { 
            flex: 1; 
            background: rgba(255,255,255,0.1); 
            color: white; 
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 12px;
            resize: vertical; 
            min-height: 50px;
        }
        button { 
            background: linear-gradient(45deg, #00ff88, #00ccff); 
            color: black; 
            font-weight: bold; 
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            cursor: pointer; 
        }
        button:hover {
            transform: translateY(-1px);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 KIMERA Universal Translator</h1>
            <p>Fast • Natural • Multi-Modal</p>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="message system">Welcome! I'm KIMERA, ready to chat naturally.</div>
        </div>
        <div class="input-area">
            <select id="mode">
                <option value="natural_language">Natural</option>
                <option value="mathematical">Math</option>
                <option value="echoform">EchoForm</option>
                <option value="emotional_resonance">Emotional</option>
                <option value="consciousness_field">Consciousness</option>
                <option value="quantum_entangled">Quantum</option>
            </select>
            <textarea id="input" placeholder="Type here... (Ctrl+Enter to send)"></textarea>
            <button onclick="send()">Send 🚀</button>
        </div>
    </div>
    <script>
        function addMessage(content, type) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = content.replace(/\\n/g, '<br>');
            document.getElementById('chatMessages').appendChild(div);
            document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
        }
        
        async function send() {
            const input = document.getElementById('input');
            const mode = document.getElementById('mode').value;
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message, mode})
                });
                const data = await response.json();
                addMessage(data.response, 'kimera');
            } catch (error) {
                addMessage('Connection error. Try again.', 'system');
            }
        }
        
        document.getElementById('input').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); send(); }
        });
        
        document.getElementById('input').focus();
    </script>
</body>
</html>
"""

# Initialize the real cognitive engine
try:
    translator_hub = UniversalTranslatorHub()
    logger.info("✅ Real KIMERA UniversalTranslatorHub initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize UniversalTranslatorHub: {e}", exc_info=True)
    translator_hub = None # Set to None if initialization fails

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    if not translator_hub:
        return jsonify({'response': 'Error: KIMERA cognitive engine is not available.'}), 503

    data = request.get_json()
    message = data.get('message', '')
    mode = data.get('mode', 'natural_language')
    
    try:
        response = translator_hub.translate(
            text=message,
            source_modality="natural_language",
            target_modality=mode,
        )
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error during translation: {e}", exc_info=True)
        return jsonify({'response': 'An error occurred within the KIMERA cognitive engine.'}), 500

if __name__ == "__main__":
    logger.info("🌟 KIMERA Web Chat starting at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False) 