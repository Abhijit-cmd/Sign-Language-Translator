# web_app.py - complete with auto-refresh, options page, and prediction page

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2
import shared_state  # Ensure this exists
import os

app = Flask(__name__, template_folder='Pages')
app_instance = None  # Pointer to your Tkinter app

@app.route('/')
def options():
    """Render the options selection page"""
    return render_template('options.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Render the prediction page based on selected language (for extensibility)"""
    language = request.form.get('language', 'english')
    return render_template('prediction.html', language=language)

@app.route("/prediction")
def get_prediction():
    """Send the current prediction and full sentence"""
    return jsonify({
        "current": shared_state.current_prediction,
        "sentence": app_instance.display_sentence.strip() if app_instance else ""
    })

@app.route("/clear")
def clear():
    """Clear current sentence and prediction"""
    if app_instance:
        app_instance.clear_text()
        return jsonify({'status': 'cleared'})
    return jsonify({'error': 'App instance not found'}), 500

@app.route('/video_feed')
def video_feed():
    """Stream the live webcam feed"""
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video():
    """Yield frames for live video"""
    global app_instance
    while True:
        if app_instance and hasattr(app_instance, "vs"):
            ret, frame = app_instance.vs.read()
            if ret:
                frame = cv2.flip(frame, 1)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def run_flask(app_ref=None):
    """Run Flask app alongside the GUI"""
    global app_instance
    app_instance = app_ref
    app.run(debug=False, port=5000, use_reloader=False)

if __name__ == "__main__":
    app.run(debug=True)