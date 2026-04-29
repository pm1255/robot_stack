def disable_torch_jit_fusion():
    import torch

    for name in [
        "_jit_set_profiling_executor",
        "_jit_set_profiling_mode",
        "_jit_override_can_fuse_on_gpu",
        "_jit_override_can_fuse_on_cpu",
        "_jit_set_texpr_fuser_enabled",
        "_jit_set_nvfuser_enabled",
    ]:
        fn = getattr(torch._C, name, None)
        if fn is not None:
            try:
                fn(False)
            except Exception:
                pass
