from flask import Flask, request, jsonify, render_template
import sys
import traceback
import os

# Import the model from the specific notebook implementation
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from instagram_audit import run_audit

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audit', methods=['POST'])
def audit():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        target_username = data.get('target_username')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
            
        result = run_audit(username, password, target_username)
        
        return jsonify({
            'success': result.get('status') == 'success',
            'data': result if result.get('status') == 'success' else None,
            'error': result.get('error') if result.get('status') == 'error' else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

# Enable CORS for frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
