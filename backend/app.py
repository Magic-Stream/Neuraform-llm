"""
🧠 NEURAFORM API Server
Run this on Google Colab or your own server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import time
import os

# Import model
from model import Neuraform, CharTokenizer, KNOWLEDGE_BASE

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variables
model = None
tokenizer = None
device = None
config = {
    'model_name': 'Neuraform-8L-256D',
    'version': '1.0.0'
}

def load_model():
    """Load the Neuraform model"""
    global model, tokenizer, device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Using device: {device}")
    
    # Create tokenizer
    tokenizer = CharTokenizer(KNOWLEDGE_BASE)
    print(f"📚 Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model = Neuraform(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=8,
        block_size=256
    ).to(device)
    
    # Load weights if available
    if os.path.exists('neuraform_model.pt'):
        checkpoint = torch.load('neuraform_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded model weights")
    else:
        print("⚠️ No saved weights found, using random initialization")
    
    model.eval()
    print("🧠 Neuraform ready!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Neuraform API',
        'version': config['version'],
        'status': 'running'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_info': config['model_name'],
        'device': str(device)
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from prompt"""
    global model, tokenizer, device
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Get parameters
        prompt = data.get('prompt', '')
        max_tokens = min(data.get('max_tokens', 200), 500)
        temperature = max(0.1, min(data.get('temperature', 0.8), 2.0))
        top_k = max(1, min(data.get('top_k', 50), 100))
        top_p = max(0.1, min(data.get('top_p', 0.9), 1.0))
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        print(f"📝 Generating for prompt: {prompt[:50]}...")
        
        # Tokenize
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        elapsed = time.time() - start_time
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        # Get only new tokens
        new_text = generated_text[len(prompt):]
        
        print(f"✅ Generated {len(new_text)} chars in {elapsed:.2f}s")
        
        return jsonify({
            'text': new_text,
            'full_text': generated_text,
            'prompt': prompt,
            'tokens_generated': max_tokens,
            'time_taken': round(elapsed, 2),
            'tokens_per_second': round(max_tokens / elapsed, 1)
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    n_params = sum(p.numel() for p in model.parameters())
    
    return jsonify({
        'name': config['model_name'],
        'version': config['version'],
        'parameters': n_params,
        'parameters_millions': round(n_params / 1e6, 2),
        'vocab_size': tokenizer.vocab_size,
        'device': str(device)
    })

if __name__ == '__main__':
    print("🧠 Starting Neuraform API Server...")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
