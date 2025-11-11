import uuid
import torch
import torch.nn as nn


# TODO what if you have one model running in parallel

# TODO use CUDA events

def pre_hook(_module: nn.Module, _args, logs):
    start_event = logs['times']['start']
    start_event.record()
    # torch.mps.Event.record(start_event)
    # torch._C._mps_recordEvent(start_event.__eventId)


def post_hook(_module: nn.Module, _args, _output, logs):
    end_event = logs['times']['end']
    end_event.record()

def post_hook_top_level(module: nn.Module, args, output, logs, logger):
    post_hook(module, args, output, logs)
    logger.save_logs(logs)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_device_event() -> torch.cuda.Event | torch.mps.Event | None:
    """
    Creates a device-specific event object. Supports CUDA and MPS currently.
    """
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    elif torch.backends.mps.is_available():
        return torch.mps.Event(enable_timing=True)
    else:
        return None # TODO implement logging for CPU

def syncronise_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    # Nothing required for CPU


class ModelLogger:  # maybe rename to ModelProfiler?
    time_logs = {}
    """
    Model Profiler.
    
    Time logs format:
    { 
        <unique execution id>: {
            'module_name': ...
            'time': {  # all in nanoseconds
                'start': ...
                'end': ...
                'elapsed' ... 
            }
            'children': [...]  # recursive
        } 
    }
    """
    models = []

    # TODO: add function for removing hooks


    def _process_times(self, log_node):
        """
        Recursively walks the log tree and calculates elapsed times
        from the recorded CUDA events.
        """
        if 'start' in log_node['times'] and 'end' in log_node['times']:
            # This is the actual time calculation
            start_event = log_node['times']['start']
            end_event = log_node['times']['end']

            # Calculate time (in ms) and convert to nanoseconds
            elapsed_ms = start_event.elapsed_time(end_event)
            log_node['times']['elapsed'] = elapsed_ms * 1_000_000

        # Recurse for children
        for child in log_node.get('children', []):
            self._process_times(child)


    def save_logs(self, logs):
        # Synchronise only once per forward pass:
        syncronise_device()

        # Calculate the elapsed times
        self._process_times(logs)

        # Save the logs
        session_id = uuid.uuid4()
        self.time_logs[session_id] = logs

    def patch_module(self, module: nn.Module, parent_logs=None, module_name=None):
        # print(f"Patching: { id(module) } or type: { type(module) }")
        if module_name is None:
            module_name = module._get_name()

        if parent_logs is None and not module_name in self.models:
            self.models.append(module_name)

        logs = {
            'module_name': module_name,
            'times': {
                'start': create_device_event(),
                'end': create_device_event(),
                'elapsed': 0,
            },
            'children': [],
        }

        if parent_logs is not None:
            parent_logs['children'].append(logs)

        # Register hooks with proper scoping
        hook_pre = lambda m, args: pre_hook(m, args, logs)
        module.register_forward_pre_hook(hook_pre)

        if parent_logs is None:
            hook_post = lambda m, args, out: post_hook_top_level(m, args, out, logs, logger=self)
            module.register_forward_hook(hook_post)
        else:
            hook_post = lambda m, args, out: post_hook(m, args, out, logs)
            module.register_forward_hook(hook_post)

        # Recursively patch children modules
        # modules_patched = {}
        for name, child in module.named_children():
            if isinstance(child, nn.Module): # and not modules_patched.get(id(child), False):
                # modules_patched[id(child)] = True
                self.patch_module(child, parent_logs=logs, module_name=name)


    def to_dict(self):
        # TODO output model information
        # print(self.time_logs)
        # return self.time_logs
        return {str(key): value for key, value in self.time_logs.items()}
