'''
template_config['model']['first_step']= False
'''
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# print('evaluate using gpu is:',os.environ["CUDA_VISIBLE_DEVICES"])
# >>>
if __name__ == '__main__':
    import os
    import sys
    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional
import lib


def main(
    path: Path,
    n_seeds: int = 15,
    function: Optional[str] = None,
    *,
    force: bool = False,
):
    path = lib.get_path(path)
    if path.name.endswith('-tuning'):
        from_tuning = True
        assert function is None
        assert (path / 'DONE').exists()

        tuning_report = lib.load_report(path)  ##report.json
        function_qualname = tuning_report['config']['function']
        template_config = tuning_report['best']['config']
        # template_config['model']['first_step']= False  ##lzd在第二步时，first_step为False

        path = path.with_name(path.name.replace('tuning', 'evaluation'))
        path.mkdir(exist_ok=True)
    else:
        print('*********notation path.name.endswith  -evaluation*********')
        from_tuning = False
        assert path.name.endswith('-evaluation')
        assert function is not None
        function_qualname = function
        template_config = lib.load_config(path / '0.toml')

    function_: lib.Function = lib.import_(function_qualname) ##function_ 是一个变量,赋值为lib.Function = lib.import_(function_qualname)
    for seed in range(n_seeds):
        config = deepcopy(template_config)
        config['seed'] = seed
        if 'catboost' in function_qualname:
            if config['model']['task_type'] == 'GPU':
                # config['model']['task_type'] = 'CPU'  # this is crucial for good results
                thread_count = config['model'].get('thread_count', 1)
                config['model']['thread_count'] = max(thread_count, 4)
        config_path = path / f'{seed}.toml'
        try:
            if seed > 0 or from_tuning:
                lib.dump_config(config, config_path)
            function_(config, config_path.with_suffix(''), force=force)
        except Exception:
            if seed > 0 or from_tuning:
                config_path.unlink(True)
            shutil.rmtree(config_path.with_suffix(''), True)
            raise


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_cli(main)
