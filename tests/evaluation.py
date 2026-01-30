import types
import typing

from model_runner import MainService
from tests.testing_models import load_simple_net, load_conv_next, simple_net_rand_inputs, conv_next_rand_inputs

evaluation_models: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    ("simple_net", load_simple_net, simple_net_rand_inputs),
    ("conv_next", load_conv_next, conv_next_rand_inputs)
]


def evaluation_main():
    # Constants:
    requests: dict[str, list[int]] = dict()
    num_requests = 5

    # Init main
    print("Initialising main service...")
    main = MainService(verbose=True)

    # Adding models
    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"> Adding model {model_name} with load function {load_model.__name__}")
        main.add_model(model_name, load_model)

        # Adding some work
        requests[model_name] = list()
        for _ in range(num_requests):
            x = rand_inputs()
            req_id = main.queue_work(model_name, x)
            requests[model_name].append(req_id)
            print(f" > Work added with request id: {req_id}")

    # Running the main service:
    main.run(exit_when_done=True)

    # Checking the work
    #TODO run the models normally and compare the output

    #TODO: finish evaluation


if __name__ == '__main__':
    evaluation_main()