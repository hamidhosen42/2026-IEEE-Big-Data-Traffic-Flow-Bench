"""Run one stage in this process:  python -m tfb.stage_exec <ctx.json>

Sets tfb.runtime.ctx, changes into the stage's output folder (as /kaggle/working was the CWD of the original kernel, so
CatBoost's catboost_info/ lands there too) and executes the vendored stage module as __main__.
The runner starts every stage in a fresh subprocess, so memory is released between stages and a crash costs one stage.
"""
import os
import runpy
import sys
import time


def _darwin_rss_in_kb():
    """macOS reports ru_maxrss in bytes, Linux (Kaggle, the reference platform) in KiB; the Task 1 kernels print / gate peak
    RSS assuming KiB. Local development on macOS only: present the value in KiB (no effect on Linux)."""
    import resource
    orig = resource.getrusage

    class _RU:
        def __init__(self, ru):
            self._ru = ru

        def __getattr__(self, k):
            v = getattr(self._ru, k)
            return v // 1024 if k == 'ru_maxrss' else v

    resource.getrusage = lambda who: _RU(orig(who))


def main(ctx_path):
    if sys.platform == 'darwin':
        _darwin_rss_in_kb()
    import tfb.runtime as rt
    rt.ctx = rt.Ctx.load(ctx_path)
    os.makedirs(rt.ctx.out, exist_ok=True)
    os.chdir(rt.ctx.out)
    t0 = time.time()
    print(f'[tfb] stage {rt.ctx.stage} -> {rt.ctx.out} | pick {rt.ctx.pick} | recipe {rt.ctx.recipe} | smoke {rt.ctx.smoke}', flush=True)
    runpy.run_module(f'tfb.stages.{rt.ctx.module}', run_name='__main__', alter_sys=True)
    print(f'[tfb] stage {rt.ctx.stage} finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
