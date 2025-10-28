import uuid
import torch.nn as nn
import time


def pre_hook(_module: nn.Module, _args, logs):
    logs['times']['start'] = time.time_ns()

def post_hook(_module: nn.Module, _args, _output, logs):
    logs['times']['end'] = time.time_ns()
    elapsed = logs['times']['end'] - logs['times']['start']
    logs['times']['elapsed'] = elapsed

def post_hook_top_level(module: nn.Module, args, output, logs, logger):
    post_hook(module, args, output, logs)
    logger.save_logs(logs)


class ModelLogger:  # maybe rename to ModelProfiler?
    time_logs = {}
    """
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


    def save_logs(self, logs):
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
                'start': 0,
                'end': 0,
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
