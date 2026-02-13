import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS

from .main import MainService
from .timed_module import timed_module_registry


def create_flask_app(main_service: MainService) -> Flask:
    """Create a Flask app wired to an existing MainService instance.

    The app exposes an async inference API (POST to submit, GET to poll)
    and utility endpoints for logs, models, devices, and rebalancing.

    Args:
        main_service: A fully initialised ``MainService`` (models already
            registered via ``add_model``).

    Returns:
        A configured Flask application.
    """
    app = Flask(__name__)
    CORS(app)

    @app.route('/api/ping', methods=['GET'])
    def ping():
        return jsonify({"message": "pong"})

    @app.route('/api/models', methods=['GET'])
    def get_models():
        return jsonify(main_service.get_model_names())

    def _serialise_logs(logs: dict) -> dict:
        """Convert UUID keys in timing logs to module paths for JSON serialisation."""
        result = {}
        for module_uuid, times in logs.items():
            timed_module = timed_module_registry.get(module_uuid)
            key = timed_module.get_path() if timed_module is not None else str(module_uuid)
            result[key] = times
        return result

    @app.route('/api/logs', methods=['GET'])
    def get_logs():
        raw = main_service.get_logs()
        return jsonify({name: _serialise_logs(v) for name, v in raw.items()})

    @app.route('/api/logs/<model_name>', methods=['GET'])
    def get_logs_for_model(model_name: str):
        raw = main_service.get_logs()
        if model_name not in raw:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        return jsonify(_serialise_logs(raw[model_name]))

    @app.route('/api/devices', methods=['GET'])
    def get_devices():
        return jsonify(main_service.get_device_info())

    @app.route('/api/run-model/<model_name>', methods=['POST'])
    def run_model(model_name: str):
        body = request.get_json(silent=True) or {}
        input_data = body.get("input")

        import torch
        if input_data is not None:
            x = torch.tensor(input_data)
        else:
            return jsonify({"error": "Missing 'input' in request body"}), 400

        try:
            request_id = main_service.queue_work(model_name, x)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

        return jsonify({"request_id": str(request_id)})

    @app.route('/api/result/<request_id>', methods=['GET'])
    def get_result(request_id: str):
        try:
            rid = uuid.UUID(request_id)
        except ValueError:
            return jsonify({"error": "Invalid request_id"}), 400

        result = main_service.get_result(rid)
        if result is None:
            return jsonify({"status": "pending"})

        model_name, output = result
        output_json = output.detach().cpu().tolist() if hasattr(output, 'tolist') else output
        return jsonify({"status": "done", "model_name": model_name, "output": output_json})

    @app.route('/api/rebalance/<model_name>', methods=['POST'])
    def force_rebalance(model_name: str):
        try:
            main_service.force_rebalance(model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        return jsonify({"status": "rebalance requested", "model_name": model_name})

    return app
