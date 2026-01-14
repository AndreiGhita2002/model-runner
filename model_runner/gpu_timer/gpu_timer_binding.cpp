#include <torch/extension.h>

void start_timer_launcher(torch::Tensor buffer);
void end_timer_launcher(torch::Tensor buffer);

void start_timer(torch::Tensor buffer) {
    // Safety checks
    TORCH_CHECK(buffer.device().is_cuda(), "Buffer must be a CUDA tensor");
    TORCH_CHECK(buffer.is_contiguous(), "Buffer must be contiguous");
    TORCH_CHECK(buffer.scalar_type() == torch::kInt64, "Buffer must be of type int64");
    TORCH_CHECK(buffer.numel() == 1, "Buffer must have one element");

    // Launch the kernel
    start_timer_launcher(buffer);
}

void end_timer(torch::Tensor buffer) {
    // Safety checks
    TORCH_CHECK(buffer.device().is_cuda(), "Buffer must be a CUDA tensor");
    TORCH_CHECK(buffer.is_contiguous(), "Buffer must be contiguous");
    TORCH_CHECK(buffer.scalar_type() == torch::kInt64, "Buffer must be of type int64");
    TORCH_CHECK(buffer.numel() == 1, "Buffer must have one element");

    // Launch the kernel
    end_timer_launcher(buffer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("start", &start_timer, "Start clock64 timer kernel");
    m.def("end", &end_timer, "End clock64 timer kernel and compute elapsed");
}
