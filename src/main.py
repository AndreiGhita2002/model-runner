from src import flask_app
from src.flask_app import run_flask_app
from tests.conv_next import conv_next_run
from tests.simple_net import simple_net_run


class MainService:
    model_endpoint = {}

    def __init__(self):
        # initialise models
        self.model_endpoint['simple-net-test'] = simple_net_run
        self.model_endpoint['conv-net-test'] = conv_next_run

        # open up the API endpoints
        run_flask_app(debug=True)


    def model_run(self, slug: str):
        return self.model_endpoint[slug]()


if __name__ == '__main__':
    _main = MainService()
