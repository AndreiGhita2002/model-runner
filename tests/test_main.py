import pprint
import time

from main import MainService

# TODO: improve this test to call all MainService functions
def test_main_service():
    """Test function for MainService."""
    N_RUNS = 5
    RESULT_FILE = "results.txt"

    main = MainService(
        depth=2,
        use_multi_device=True,
        split_strategy="computation_based"
    )

    main.print_status()

    print("\n" + "=" * 80)
    print("PyTorch Model Load Balancer - Testing Suite")
    print("=" * 80)
    print(f"Running each model {N_RUNS} times...")

    all_results = {}

    # Profile and create multi-device models
    print("\n" + "=" * 80)
    print("Phase 1: Model Profiling")
    print("=" * 80)
    for model_name in main.get_model_names():
        main.profile_model(model_name, num_warmup=2, num_profile=3)
        if main.num_stages >= 2:
            main.create_multi_device_model(model_name)

    # Run each model N times
    print("\n" + "=" * 80)
    print("Phase 2: Model Execution")
    print("=" * 80)

    for model_name in main.get_model_names():
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 80}")

        model_results = []

        for run_idx in range(N_RUNS):
            print(f"\nRun {run_idx + 1}/{N_RUNS}...")

            start_time = time.time()
            result = main.run_model(model_name, None, randomise_input=True)
            elapsed = time.time() - start_time

            logs = main.get_logs()

            model_results.append({
                'run': run_idx + 1,
                'result': result,
                'elapsed_time': elapsed,
                'logs': logs[model_name] if model_name in logs else None
            })

            print(f"  Result type: {type(result)}")
            print(f"  Elapsed time: {elapsed*1000:.2f}ms")
            if logs.get(model_name):
                print(f"  Timing info available: Yes")

        all_results[model_name] = model_results
        avg_time = sum(r['elapsed_time'] for r in model_results) / len(model_results)
        print(f"\nCompleted {N_RUNS} runs for {model_name}")
        print(f"Average execution time: {avg_time*1000:.2f}ms")

    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Total runs: {len(results)}")
        successful_runs = sum(1 for r in results if r['result'] is not None and not isinstance(r['result'], dict))
        print(f"  Successful runs: {successful_runs}/{len(results)}")
        avg_time = sum(r['elapsed_time'] for r in results) / len(results)
        print(f"  Average time: {avg_time*1000:.2f}ms")

    main.print_status()

    # Write results
    print(f"\n{'=' * 80}")
    print(f"Writing detailed results to {RESULT_FILE}...")
    print(f"{'=' * 80}\n")

    with open(RESULT_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch Model Load Balancer - Test Results\n")
        f.write("=" * 80 + "\n\n")

        device_info = main.get_device_info()
        f.write(f"Number of devices: {device_info['num_devices']}\n")
        for dev in device_info['devices']:
            f.write(f"  Device {dev['index']}: {dev['name']}\n")
        f.write(f"\nDepth: {main.depth}\n")
        f.write(f"Number of runs per model: {N_RUNS}\n")
        f.write(f"Multi-device mode: {main.use_multi_device}\n")
        f.write(f"Split strategy: {main.split_strategy}\n\n")

        for model_name, results in all_results.items():
            f.write("=" * 80 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 80 + "\n\n")

            if model_name in main.timing_profiles:
                f.write("Timing Profile:\n")
                f.write(pprint.pformat(main.timing_profiles[model_name], width=100))
                f.write("\n\n")

            times = [r['elapsed_time'] for r in results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            f.write(f"Execution Statistics:\n")
            f.write(f"  Average: {avg_time*1000:.2f}ms\n")
            f.write(f"  Min: {min_time*1000:.2f}ms\n")
            f.write(f"  Max: {max_time*1000:.2f}ms\n\n")

            for result_data in results:
                f.write(f"--- Run {result_data['run']} ---\n")
                f.write(f"Result type: {type(result_data['result'])}\n")
                f.write(f"Elapsed time: {result_data['elapsed_time']*1000:.2f}ms\n\n")

                if result_data['logs']:
                    f.write("Logs:\n")
                    f.write(pprint.pformat(result_data['logs'], width=100))
                    f.write("\n\n")
                else:
                    f.write("No logs available\n\n")

            f.write("\n")

    print(f"Results written to {RESULT_FILE}")
    print("Testing complete!")

if __name__ == '__main__':
    test_main_service()