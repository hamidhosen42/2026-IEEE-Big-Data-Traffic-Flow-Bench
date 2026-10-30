"""Kaggle helper on the official kaggle package (kaggle.json in the project root via KAGGLE_CONFIG_DIR), replacing kjob.py whose
Bearer token (~/.kaggle/access_token) returns 401 since 2026-10-29. Never prints or reads the credential itself.
  push(slug, script_path, kernel_sources) -> version number      status(slug) -> 'complete' | 'running' | 'error' | ...
  log_and_files(slug) -> (log_text, [file names])  (no file download)"""
import json, os, shutil, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KAGGLE_CONFIG_DIR', ROOT)
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

COMP = '2026-ieee-big-data-traffic-flow-bench'
USER = 'hosen42'
_api = None


def api():
    global _api
    if _api is None:
        _api = KaggleApi(); _api.authenticate()
    return _api


def push(slug, script_path, kernel_sources=()):
    d = tempfile.mkdtemp(prefix=f'push_{slug}_')
    try:
        shutil.copy(script_path, os.path.join(d, os.path.basename(script_path)))
        meta = {'id': f'{USER}/{slug}', 'title': slug, 'code_file': os.path.basename(script_path), 'language': 'python',
                'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
                'dataset_sources': [], 'competition_sources': [COMP], 'kernel_sources': list(kernel_sources), 'model_sources': []}
        json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
        r = api().kernels_push(d)
        err = getattr(r, 'error', None) or getattr(r, 'errorNullable', None)
        if err:
            raise RuntimeError(f'push {slug} failed: {err}')
        return getattr(r, 'version_number', None) or getattr(r, 'versionNumber', None)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def status(slug):
    r = api().kernels_status(f'{USER}/{slug}')
    s = getattr(r, 'status', r)
    return str(getattr(s, 'name', s)).lower()


def log_and_files(slug):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with api().build_kaggle_client() as k:
        req = ApiListKernelSessionOutputRequest(); req.user_name = USER; req.kernel_slug = slug
        resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
    raw = resp.log or ''
    try:
        text = ''.join(x.get('data', '') for x in json.loads(raw))
    except Exception:
        text = raw
    return text, [f.file_name for f in resp.files]
