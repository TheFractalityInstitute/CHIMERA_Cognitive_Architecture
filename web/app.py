"""
Main web application for CHIMERA
Integrates persistence, real-time updates, and learning visualization
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from chimera import CHIMERA
from chimera.memory.manager import MemoryManager
from chimera.learning.organic import OrganicLearningSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chimera-secret-key-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize CHIMERA components
config = {
    'db_path': 'data/chimera.db',
    'embedding_dim': 512,
    'decay_rate': 0.001,
    'consolidation_threshold': 0.7
}

# Global instances
chimera = None
memory_manager = None
learning_system = None
active_sessions = {}

def initialize_chimera():
    """Initialize CHIMERA with all components"""
    global chimera, memory_manager, learning_system
    
    # Initialize memory manager
    memory_manager = MemoryManager(config)
    
    # Initialize main CHIMERA system
    chimera = CHIMERA()
    chimera.memory_manager = memory_manager
    
    # Initialize learning system
    learning_system = OrganicLearningSystem("chimera_web")
    learning_system.memory_manager = memory_manager
    
    # Start CHIMERA in background
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(chimera.initialize())
    
    print("✓ CHIMERA initialized successfully")

# Initialize on startup
initialize_chimera()

# ============= REST API Endpoints =============

@app.route('/')
def index():
    """Serve main interface"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current CHIMERA status"""
    status = {
        'online': chimera is not None,
        'thoughts': len(chimera.agents['crystallization'].insights) if chimera else 0,
        'concepts': learning_system.total_words_learned if learning_system else 0,
        'conversations': learning_system.conversation_count if learning_system else 0,
        'curiosity_level': learning_system.curiosity_level if learning_system else 0,
        'development_stage': learning_system._get_development_stage() if learning_system else "Not initialized",
        'active_sessions': len(active_sessions),
        'uptime': time.time() - chimera.start_time if chimera and hasattr(chimera, 'start_time') else 0
    }
    return jsonify(status)

@app.route('/api/thoughts')
async def get_thoughts():
    """Get recent thoughts"""
    thoughts = []
    
    # Get from crystallization engine
    if chimera and 'crystallization' in chimera.agents:
        for insight_id, insight in chimera.agents['crystallization'].insights.items():
            thoughts.append({
                'id': insight.id,
                'type': 'crystallized',
                'content': insight.linguistic_expression,
                'confidence': insight.confidence,
                'timestamp': insight.timestamp,
                'verified': insight.verification_count > 0
            })
    
    # Get from reasoning engine
    if learning_system and learning_system.reasoning:
        for thought_id, thought in learning_system.reasoning.thought_network.items():
            thoughts.append({
                'id': thought_id,
                'type': 'thought',
                'content': thought.symbolic_form,
                'confidence': thought.confidence,
                'connections': len(thought.connections),
                'timestamp': thought.timestamp
            })
    
    # Sort by timestamp, most recent first
    thoughts.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(thoughts[:100])  # Return latest 100

@app.route('/api/concepts')
async def get_concepts():
    """Get learned concepts"""
    concepts = []
    
    if learning_system and learning_system.language:
        for word, data in learning_system.language.vocabulary.items():
            concepts.append({
                'term': word,
                'confidence': data.get('confidence', 0),
                'count': data.get('count', 0),
                'meanings': data.get('meanings', []),
                'taught': data.get('taught', False)
            })
    
    # Sort by confidence
    concepts.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify(concepts)

@app.route('/api/memory/save', methods=['POST'])
async def save_memory():
    """Save current memory state"""
    try:
        snapshot_name = request.json.get('name', None)
        snapshot_path = await memory_manager.save_snapshot(snapshot_name)
        
        return jsonify({
            'success': True,
            'path': str(snapshot_path),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/memory/load', methods=['POST'])
async def load_memory():
    """Load memory state from snapshot"""
    try:
        snapshot_name = request.json.get('name')
        await memory_manager.load_snapshot(snapshot_name)
        
        return jsonify({
            'success': True,
            'message': f'Loaded snapshot: {snapshot_name}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/<format>')
async def export_data(format):
    """Export conversation/learning data"""
    if format == 'json':
        data = {
            'conversations': await memory_manager.db.export_conversations(),
            'concepts': await memory_manager.db.export_concepts(),
            'insights': await memory_manager.db.export_insights(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    elif format == 'csv':
        # Generate CSV file
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write conversations
        writer.writerow(['Timestamp', 'Speaker', 'Message', 'Confidence'])
        conversations = await memory_manager.db.export_conversations()
        for conv in conversations:
            writer.writerow([
                conv['timestamp'],
                conv['speaker'],
                conv['message'],
                conv.get('understanding_confidence', '')
            ])
        
        # Create response
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'chimera_export_{int(time.time())}.csv'
        )

# ============= WebSocket Handlers =============

@socketio.on('connect')
def handle_connect():
    """Handle new connection"""
    session_id = request.sid
    active_sessions[session_id] = {
        'connected_at': time.time(),
        'messages': 0
    }
    
    emit('connected', {
        'session_id': session_id,
        'message': 'Connected to CHIMERA'
    })
    
    # Join personal room for targeted messages
    join_room(session_id)
    
    # Check for pending curiosities
    check_and_send_curiosities(session_id)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle disconnection"""
    session_id = request.sid
    if session_id in active_sessions:
        del active_sessions[session_id]
    leave_room(session_id)

@socketio.on('message')
async def handle_message(data):
    """Handle incoming message from user"""
    session_id = request.sid
    text = data.get('text', '').strip()
    
    if not text:
        return
    
    # Update session stats
    if session_id in active_sessions:
        active_sessions[session_id]['messages'] += 1
    
    # Store user message
    await memory_manager.store_conversation({
        'session_id': session_id,
        'timestamp': time.time(),
        'speaker': 'user',
        'message': text
    })
    
    # Process through learning system
    result = await learning_system.interact(text)
    
    # Store CHIMERA's response
    await memory_manager.store_conversation({
        'session_id': session_id,
        'timestamp': time.time(),
        'speaker': 'chimera',
        'message': result['response'],
        'understanding_confidence': result['understanding'],
        'thoughts_formed': result.get('thoughts_formed', 0)
    })
    
    # Send response
    emit('response', {
        'text': result['response'],
        'confidence': result['understanding'],
        'thoughts_formed': result['thoughts_formed'],
        'words_known': result['words_known'],
        'development_stage': result['development_stage'],
        'curiosity': result['curiosity']
    })
    
    # Broadcast learning milestones to all connected clients
    if learning_system.development_log:
        latest_milestone = learning_system.development_log[-1]
        if latest_milestone != 'shown':
            socketio.emit('milestone', {
                'text': latest_milestone,
                'timestamp': time.time()
            }, broadcast=True)
            learning_system.development_log.append('shown')
    
    # Update all clients with new stats
    await broadcast_stats_update()

@socketio.on('teach')
async def handle_teach(data):
    """Handle direct teaching"""
    concept = data.get('concept', '')
    explanation = data.get('explanation', '')
    examples = data.get('examples', [])
    
    if not concept:
        return
    
    # Teach the concept
    result = learning_system.teach(concept, explanation, examples)
    
    # Store in database
    await memory_manager.db.store_concept({
        'term': concept,
        'definition': explanation,
        'examples': examples,
        'confidence': result['current_understanding'].get('confidence', 0.7)
    })
    
    emit('teach_result', {
        'success': True,
        'concept': concept,
        'understanding': result['current_understanding']
    })
    
    await broadcast_stats_update()

@socketio.on('request_curiosity')
def handle_curiosity_request():
    """User requests to see CHIMERA's curiosity"""
    session_id = request.sid
    
    # Get top curiosity from the engine
    if chimera and chimera.agents.get('curiosity'):
        curiosity = chimera.agents['curiosity'].get_next_curiosity()
        if curiosity:
            emit('curiosity', {
                'question': curiosity['question'],
                'target': curiosity['target'],
                'priority': curiosity['priority']
            }, room=session_id)

def check_and_send_curiosities(session_id):
    """Check if CHIMERA has curiosities to share"""
    if not chimera:
        return
        
    curiosity_engine = chimera.agents.get('curiosity')
    if not curiosity_engine:
        return
        
    # Check attention budget
    if curiosity_engine.should_notify_user():
        curiosity = curiosity_engine.get_next_curiosity()
        if curiosity:
            socketio.emit('curiosity', {
                'question': curiosity['question'],
                'target': curiosity['target'],
                'priority': curiosity['priority'],
                'spontaneous': True
            }, room=session_id)

async def broadcast_stats_update():
    """Broadcast updated stats to all connected clients"""
    stats = {
        'total_thoughts': len(learning_system.reasoning.thought_network),
        'total_concepts': len(learning_system.language.vocabulary),
        'total_conversations': learning_system.conversation_count,
        'abstraction_level': learning_system.abstraction_level,
        'active_users': len(active_sessions)
    }
    
    socketio.emit('stats_update', stats, broadcast=True)

# ============= Background Tasks =============

def background_curiosity_checker():
    """Periodically check for curiosities to share"""
    while True:
        time.sleep(60)  # Check every minute
        
        for session_id in active_sessions:
            # Don't spam - check attention budget
            if active_sessions[session_id]['messages'] > 0:
                check_and_send_curiosities(session_id)

# Start background thread
import threading
curiosity_thread = threading.Thread(target=background_curiosity_checker, daemon=True)
curiosity_thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)