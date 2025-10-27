from flask import Flask, jsonify
from flask_cors import CORS


flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Hello from Python!"})

simple_net_output = None

@flask_app.route('/api/simple-net-test/', methods=['GET'])
def simple_net_test():
    global simple_net_output

    if simple_net_output is None:
        from tests.simple_net import simple_net_run
        simple_net_output = simple_net_run()

    return jsonify(simple_net_output)

conv_net_output = None

@flask_app.route('/api/conv-net-test/', methods=['GET'])
def mnist_test():
    global conv_net_output

    if conv_net_output is None:
        from tests.conv_next import conv_next_run
        conv_net_output = conv_next_run()

    return jsonify(conv_net_output)


def run_flask_app(debug=False):
    flask_app.run(debug=debug)
