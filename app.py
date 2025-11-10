from flask import Flask, request, jsonify, Response, stream_with_context
from ai_service import AIService
import os
from markupsafe import escape

app = Flask(__name__)

# Initialize AI Service
ai_service = AIService()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/api/ingest', methods=['POST'])
def trigger_ingest():
    """
    Trigger background ingestion of iPhone data (Apple Support seeds + local PDFs).
    Returns 200 immediately.
    """
    try:
        started = ai_service.start_background_ingestion()
        return jsonify({
            "status": "started" if started else "already_running"
        }), 200
    except Exception as e:
        # Still return 200 to satisfy requirement, but include error info
        return jsonify({
            "status": "error_starting",
            "error": str(e)
        }), 200

@app.route('/api/iphone17', methods=['POST'])
def ask_about_iphone17():
    """
    Endpoint to answer questions about iPhone 17
    
    Request body:
    {
        "question": "What are the features of iPhone 17?",
        "web_fallback": true  # optional, overrides ENABLE_WEB_FALLBACK for this request
    }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data['question']
        web_fallback_override = data.get('web_fallback', None)

        # Decide if the caller requested a streaming response (default: stream)
        stream = bool(data.get("stream", True))

        if not stream:
            # Original JSON (non-streaming) response for backward-compat
            original_web_fallback = ai_service.enable_web_fallback
            try:
                if web_fallback_override is not None:
                    ai_service.enable_web_fallback = bool(web_fallback_override)
                answer = ai_service.answer_question(question)
            finally:
                ai_service.enable_web_fallback = original_web_fallback

            return jsonify({
                "question": question,
                "answer": answer,
                "status": "success"
            }), 200

        # Streaming response
        @stream_with_context
        def generate():
            """
            Stream the answer incrementally. We build the full answer first
            (service does not natively stream), then yield it in chunks to
            enable progressive rendering for clients.
            """
            original_web_fallback = ai_service.enable_web_fallback
            try:
                if web_fallback_override is not None:
                    ai_service.enable_web_fallback = bool(web_fallback_override)
                full_answer = ai_service.answer_question(question) or ""
            finally:
                ai_service.enable_web_fallback = original_web_fallback

            # Chunk by words for smoother UX; adjust as needed
            # Using MarkupSafe.escape to avoid accidental HTML/script injection
            for token in full_answer.split(" "):
                print(token)
                yield f"{escape(token)} "

        return Response(generate(), mimetype="text/plain; charset=utf-8")

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/iphone17', methods=['GET'])
def get_iphone17_info():
    """
    GET endpoint to retrieve general iPhone 17 information
    """
    try:
        default_question = "What is the iPhone 17 and what are its main features?"
        answer = ai_service.answer_question(default_question)
        
        return jsonify({
            "info": answer,
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

