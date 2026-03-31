"""
Flask Web Application for Impulsive Noise Removal
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import uuid
import threading
from noise_remover import remove_impulsive_noise

app = Flask(__name__)
app.config['SECRET_KEY'] = 'noise-remover-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'opus'}

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store processing results in memory (use Redis/DB for production)
processing_results = {}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Get parameters from form
    try:
        ar_order = int(request.form.get('ar_order', 4))
        eta = float(request.form.get('eta', 3.5))
        forgetting_factor = float(request.form.get('forgetting_factor', 0.99))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid parameter values'}), 400

    # Validate parameter ranges
    if not (1 <= ar_order <= 20):
        return jsonify({'error': 'AR Order must be between 1 and 20'}), 400
    if not (0.5 <= eta <= 10.0):
        return jsonify({'error': 'Eta (η) must be between 0.5 and 10.0'}), 400
    if not (0.9 <= forgetting_factor <= 0.999):
        return jsonify({'error': 'Forgetting factor (λ) must be between 0.9 and 0.999'}), 400

    # Save uploaded file
    job_id = str(uuid.uuid4())
    original_ext = os.path.splitext(secure_filename(file.filename))[1]
    input_filename = f"{job_id}_input{original_ext}"
    output_filename = f"{job_id}_cleaned.wav"

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    file.save(input_path)

    # Store job info
    processing_results[job_id] = {
        'status': 'queued',
        'input_path': input_path,
        'output_path': output_path,
        'original_filename': file.filename,
    }

    # Start processing in background thread
    thread = threading.Thread(
        target=process_audio,
        args=(job_id, input_path, output_path, ar_order, eta, forgetting_factor)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'message': 'Processing started'}), 202


def process_audio(job_id, input_path, output_path, ar_order, eta, forgetting_factor):
    """Background audio processing task."""

    def progress_callback(percent, message):
        socketio.emit('progress', {
            'job_id': job_id,
            'percent': percent,
            'message': message
        })

    try:
        processing_results[job_id]['status'] = 'processing'
        progress_callback(0, "Starting...")

        results = remove_impulsive_noise(
            input_filepath=input_path,
            output_filepath=output_path,
            ar_rank=ar_order,
            eta=eta,
            forgetting_factor=forgetting_factor,
            progress_callback=progress_callback
        )

        processing_results[job_id]['status'] = 'completed'
        processing_results[job_id]['results'] = results

        socketio.emit('processing_complete', {
            'job_id': job_id,
            'results': {
                'sample_rate': results['sample_rate'],
                'duration': results['duration'],
                'total_samples': results['total_samples'],
                'clicks_detected': results['clicks_detected'],
                'clicks_per_second': results['clicks_per_second'],
                'ar_order': results['ar_order'],
                'eta': results['eta'],
                'lambda': results['lambda'],
                'plots': results['plots'],
            }
        })

    except Exception as e:
        processing_results[job_id]['status'] = 'error'
        processing_results[job_id]['error'] = str(e)
        socketio.emit('processing_error', {
            'job_id': job_id,
            'error': str(e)
        })


@app.route('/download/<job_id>')
def download_file(job_id):
    """Download the cleaned audio file."""
    if job_id not in processing_results:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_results[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Processing not yet complete'}), 400

    output_path = job['output_path']
    if not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404

    original_name = os.path.splitext(job['original_filename'])[0]
    download_name = f"{original_name}_cleaned.wav"

    return send_file(
        output_path,
        as_attachment=True,
        download_name=download_name,
        mimetype='audio/wav'
    )


@app.route('/status/<job_id>')
def job_status(job_id):
    """Check the status of a processing job."""
    if job_id not in processing_results:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_results[job_id]
    return jsonify({'status': job['status']})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)