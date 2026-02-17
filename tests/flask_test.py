import json
import os
import threading
import uuid

import torch
import torch.distributed as dist

from model_runner.main import MainService
from model_runner.flask_app import create_flask_app
from model_runner.timed_module import timed_module_registry, timed_module_hierarchy
from tests.testing_models import evaluation_models


def test_flask_endpoints():
    print("\n" + "=" * 80)
    print("TEST: Flask endpoints (single-process distributed)")
    print("=" * 80)

    # --- distributed setup (single process, gloo) ---
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    try:
        # --- set up MainService and register models ---
        main_service = MainService(verbose=True)

        for model_name, load_model, rand_input in evaluation_models:
            print(f"  Registering model: {model_name}")
            main_service.add_model(
                model_name,
                load_model(),
                rand_input(),
                model_output_is_static=True,
            )

        # --- create Flask test client ---
        app = create_flask_app(main_service)
        client = app.test_client()

        # ============================================================
        # 1. GET /api/ping
        # ============================================================
        print("\n  [1] GET /api/ping")
        resp = client.get("/api/ping")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.get_json()
        assert "message" in data
        print(f"      -> {data}")

        # ============================================================
        # 2. GET /api/models
        # ============================================================
        print("\n  [2] GET /api/models")
        resp = client.get("/api/models")
        assert resp.status_code == 200
        model_names = resp.get_json()
        expected_names = [name for name, _, _ in evaluation_models]
        assert set(model_names) == set(expected_names), f"Expected {expected_names}, got {model_names}"
        print(f"      -> {model_names}")

        # ============================================================
        # 3. GET /api/devices
        # ============================================================
        print("\n  [3] GET /api/devices")
        resp = client.get("/api/devices")
        assert resp.status_code == 200
        devices = resp.get_json()
        assert "num_devices" in devices
        assert "devices" in devices
        assert devices["num_devices"] >= 1
        print(f"      -> num_devices={devices['num_devices']}")

        # ============================================================
        # 4. GET /api/logs  (empty at first, but should not error)
        # ============================================================
        print("\n  [4] GET /api/logs")
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        logs = resp.get_json()
        assert isinstance(logs, dict)
        print(f"      -> keys: {list(logs.keys())}")

        # ============================================================
        # 5. GET /api/logs/<model_name>
        # ============================================================
        first_model = expected_names[0]
        print(f"\n  [5] GET /api/logs/{first_model}")
        resp = client.get(f"/api/logs/{first_model}")
        assert resp.status_code == 200
        print(f"      -> 200 OK")

        # non-existent model should 404
        resp = client.get("/api/logs/nonexistent_model")
        assert resp.status_code == 404
        print(f"      -> /api/logs/nonexistent_model => 404 OK")

        # ============================================================
        # 6. POST /api/run-model/<model_name> — one request per model
        #    then run the pipeline so results are produced
        # ============================================================
        print("\n  [6] POST /api/run-model + GET /api/result")
        request_ids: dict[str, str] = {}
        seed = 37

        for model_name, _, rand_input_fn in evaluation_models:
            torch.manual_seed(seed)
            x = rand_input_fn()
            input_json = x.tolist()

            resp = client.post(
                f"/api/run-model/{model_name}",
                data=json.dumps({"input": input_json}),
                content_type="application/json",
            )
            assert resp.status_code == 200, f"run-model {model_name}: {resp.status_code} {resp.get_data(as_text=True)}"
            body = resp.get_json()
            assert "request_id" in body
            request_ids[model_name] = body["request_id"]
            print(f"      {model_name} -> request_id={body['request_id']}")

        # non-existent model should 404
        resp = client.post(
            "/api/run-model/nonexistent_model",
            data=json.dumps({"input": [1.0]}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        print(f"      /api/run-model/nonexistent_model => 404 OK")

        # missing input should 400
        resp = client.post(
            f"/api/run-model/{first_model}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        print(f"      missing input => 400 OK")

        # ============================================================
        # 7. Results should be pending before pipeline runs
        # ============================================================
        print("\n  [7] GET /api/result (before pipeline runs)")
        for model_name, rid in request_ids.items():
            resp = client.get(f"/api/result/{rid}")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["status"] == "pending", f"Expected pending, got {body}"
            print(f"      {model_name} -> pending")

        # invalid UUID should 400
        resp = client.get("/api/result/not-a-uuid")
        assert resp.status_code == 400
        print(f"      invalid UUID => 400 OK")

        # ============================================================
        # 8. Run the pipeline in a background thread
        # ============================================================
        print("\n  [8] Running pipeline (exit_when_done=True)...")
        pipeline_thread = threading.Thread(
            target=main_service.run,
            kwargs={"exit_when_done": True},
            daemon=True,
        )
        pipeline_thread.start()
        pipeline_thread.join(timeout=60)
        assert not pipeline_thread.is_alive(), "Pipeline did not finish in time"
        print("      Pipeline finished.")

        # ============================================================
        # 9. Results should now be available
        # ============================================================
        print("\n  [9] GET /api/result (after pipeline runs)")

        # Generate expected baseline outputs for comparison
        baselines: dict[str, torch.Tensor] = {}
        for model_name, load_model, rand_input_fn in evaluation_models:
            model = load_model()
            torch.manual_seed(seed)
            x = rand_input_fn()
            with torch.no_grad():
                baselines[model_name] = model(x)

        for model_name, rid in request_ids.items():
            resp = client.get(f"/api/result/{rid}")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["status"] == "done", f"Expected done, got {body['status']}"
            assert body["model_name"] == model_name

            # Check timing field is present with correct structure
            assert "timing" in body, f"Expected 'timing' in response, got {list(body.keys())}"
            if body["timing"] is not None:
                assert "forward" in body["timing"] and "rebalance" in body["timing"], \
                    f"Expected 'forward' and 'rebalance' in timing, got {body['timing']}"
                fwd = body["timing"]["forward"]
                assert "start" in fwd and "end" in fwd, f"Expected 'start'/'end' in forward, got {fwd}"
                reb = body["timing"]["rebalance"]
                assert "start" in reb and "end" in reb and "did_rebalance" in reb, \
                    f"Expected 'start'/'end'/'did_rebalance' in rebalance, got {reb}"

            output_tensor = torch.tensor(body["output"])
            expected = baselines[model_name]

            # The pipeline output may have been indexed differently (first microbatch)
            # so compare shapes first
            if output_tensor.shape != expected.shape:
                # Pipeline may return per-microbatch (unbatched), baseline is batched
                if output_tensor.shape == expected.squeeze(0).shape:
                    expected = expected.squeeze(0)

            assert torch.allclose(output_tensor, expected, atol=1e-5), (
                f"{model_name}: output mismatch\n"
                f"  got shape {output_tensor.shape}, expected {expected.shape}"
            )
            print(f"      {model_name} -> PASS (output matches baseline, timing present)")

        # ============================================================
        # 10. Polling again should return pending (result was popped)
        # ============================================================
        print("\n  [10] GET /api/result (second poll — should be pending/consumed)")
        for model_name, rid in request_ids.items():
            resp = client.get(f"/api/result/{rid}")
            body = resp.get_json()
            assert body["status"] == "pending", f"Expected pending after pop, got {body}"
            print(f"      {model_name} -> pending (consumed)")

        # ============================================================
        # 11. GET /api/logs should now have timing data
        # ============================================================
        print("\n  [11] GET /api/logs (after pipeline run, keys should be module paths)")
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        logs = resp.get_json()
        print(f"      -> models: {list(logs.keys())}")
        for model_name, model_logs in logs.items():
            if model_logs:
                sample_keys = list(model_logs.keys())[:5]
                print(f"      {model_name} log keys (sample): {sample_keys}")
                # Keys should be module paths (dot-separated), not UUIDs
                for key in model_logs:
                    try:
                        uuid.UUID(key)
                        assert False, f"Log key '{key}' looks like a UUID, expected a module path"
                    except ValueError:
                        pass  # Good — not a UUID

        # Also check per-model endpoint
        resp = client.get(f"/api/logs/{first_model}")
        assert resp.status_code == 200
        per_model_logs = resp.get_json()
        for key in per_model_logs:
            try:
                uuid.UUID(key)
                assert False, f"Per-model log key '{key}' looks like a UUID, expected a module path"
            except ValueError:
                pass
        print(f"      /api/logs/{first_model} keys also use module paths")

        print("\n" + "=" * 80)
        print("ALL FLASK TESTS PASSED")
        print("=" * 80)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    test_flask_endpoints()
