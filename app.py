from flask import Flask, request, render_template, jsonify
from chatbot import mental_health_chatbot, setup_rag

app = Flask(__name__)

# Setup RAG components at startup
vector_store, generator = setup_rag()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = mental_health_chatbot(user_input, vector_store, generator)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print(f"Received input: {user_input}")  # Debug line
    response = mental_health_chatbot(user_input, vector_store, generator)
    return jsonify({'response': response})
