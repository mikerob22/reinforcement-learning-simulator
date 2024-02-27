from flask_socketio import SocketIO

socketio = SocketIO()


def simulation_progress(total_steps):
    for step in range(total_steps):
        # Perform simulation step
        # Calculate progress percentage
        progress = int((step + 1) / total_steps * 100)
        # Emit progress update
        socketio.emit('progress_update', {'progress': progress})
    # Emit task complete event
    socketio.emit('task_complete', {'result': 'Simulation completed.'})


def test_progress(total_steps):
    for step in range(total_steps):
        # Perform simulation step
        # Calculate progress percentage
        progress = int((step + 1) / total_steps * 100)
        # Emit progress update
        socketio.emit('progress_update', {'progress': progress}, broadcast=True)
    # Emit task complete event
    socketio.emit('task_complete', {'result': 'Testing completed.'}, broadcast=True)
