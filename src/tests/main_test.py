import pprint

from main import MainService
from tests.conv_next import ConvNext
from tests.simple_net import SimpleNet


def initialize_test_models(main: MainService):
    """Initialize test models."""
    print("Initializing MainService...")
    print(f"Creating models on primary device: {main.primary_device}")

    simple_net = SimpleNet(str(main.primary_device))
    conv_next = ConvNext(str(main.primary_device))

    main.add_model('simple_net', simple_net)
    main.add_model('conv-next', conv_next)

    print(f"Initialized {len(main.models)} models")

def test_main_service():
    N_RUNS = 5
    RESULT_FILE = "results.txt" # TODO print to file

    main = MainService(
        depth=2,
        use_multi_device=True,
        split_strategy="computation_based"
    )
    initialize_test_models(main)

    main.print_status()

    # Queue work
    for i in range(N_RUNS):
        for j, model_name in enumerate(main.models.keys()):
            req = j + i * len(main.models)
            main.queue_work(model_name, None, req)

    # Run the Main Service
    print("Running...")
    main.run(exit_when_done=True)

    # Work queue should be empty
    assert main.work_queue.empty()
    print("All work done!")

    # Extract the responses
    for i in range(N_RUNS):
        for j, model_name in enumerate(main.models.keys()):
            req = j + i * len(main.models)
            res = main.get_work_results(req)
            if res is None:
                print(f"[req:{req} ERR] Work results for {model_name}, run {j} failed!")
            else:
                print(f"[req:{req} OK] Work results for {model_name}, run {j} succeeded!")

    pprint.pprint(main.model_outputs)


if __name__ == '__main__':
    test_main_service()