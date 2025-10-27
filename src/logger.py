import torch.nn as nn
import time


def pre_hook(_module: nn.Module, _args, logs):
    logs['time'][-1]['start'] = time.time_ns()

def post_hook(_module: nn.Module, _args, _output, logs):
    logs['time'][-1]['end'] = time.time_ns()
    elapsed = logs['time'][-1]['end'] - logs['time'][-1]['start']
    logs['time'][-1]['elapsed'] = elapsed


class ModelLogger:
    time_logs = {}

    def patch_module(self, module: nn.Module, logs=None, name=None):
        # print(f"Patching: { id(module) } or type: { type(module) }")

        if logs is None:
            logs = self.time_logs

        if logs == {}:
            # logs['module'] = module
            logs['module_name'] = name
            logs['time'] = [{
                'start': 0,
                'end': 0,
                'elapsed': 0,
            }]
            logs['children'] = []
        else:
            # module has already been patched
            logs['time'].append({
                'start': 0,
                'end': 0,
                'elapsed': 0,
            })

        # Register hooks with proper scoping
        hook_pre = lambda m, args: pre_hook(m, args, logs)
        hook_post = lambda m, args, out: post_hook(m, args, out, logs)

        module.register_forward_pre_hook(hook_pre)
        module.register_forward_hook(hook_post)

        modules_patched = {}

        # Recursively patch children modules
        for name, child in module.named_children():
            if isinstance(child, nn.Module) and not modules_patched.get(id(child), False):
                logs['children'].append({})
                modules_patched[id(child)] = True
                self.patch_module(child, logs=logs['children'][-1], name=name)

    def to_dict(self):
        # TODO output model information
        return self.time_logs


LOGGER = ModelLogger()


# not used any more
# alternative way of measuring time: decorating the forward function
def patch_forward_func(forward_func, times):
    def wrapped(*args, **kwargs):
        start_time = time.time()
        # forward function call:
        ret = forward_func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        return ret
    return wrapped
