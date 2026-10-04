"""helper: load a v21 kernel file as a module without running main() or touching /kaggle paths."""
import types


def load_kernel(path):
    src = open(path).read()
    line = "REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]"
    assert src.count(line) == 1 and src.rstrip().endswith('\nmain()')
    src = src.replace(line, "REL = '/nonexistent/kaggle_public'").rstrip()[:-len('main()')]
    mod = types.ModuleType('t2v21'); exec(compile(src, path, 'exec'), mod.__dict__)
    return mod
