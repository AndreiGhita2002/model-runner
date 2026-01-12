from flask import Flask, jsonify
from flask_cors import CORS

from main import MainService

flask_app = Flask(__name__)
CORS(flask_app)

main_service = MainService()

@flask_app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Hello from Python!"})

@flask_app.route('/api/run-model/<model_name>', methods=['POST'])
def run_model(model_name: str):

    model_output = main_service.run_model(model_name, None, randomise_input=True)

    if model_output is None:
        return jsonify({'output': 'null'})

    json_ready = model_output.detach().cpu().tolist()

    return jsonify({'output': json_ready})

@flask_app.route('/api/times', methods=['GET'])
def get_time_logs():
    return jsonify(main_service.get_logs())

@flask_app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(main_service.get_model_names())


if __name__ == '__main__':
    flask_app.run(debug=True)
