from flask import Flask
from flask_socketio import SocketIO, emit
import redis
import uuid
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

@socketio.on('send_message')
def handle_send_message_event(data):
    session_id = data['session_id']
    text = data['text']
    message_id = str(uuid.uuid4())
    message = {'id': message_id, 'text': text}
    redis_client.rpush(session_id, json.dumps(message))
    emit('receive_message', message, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
