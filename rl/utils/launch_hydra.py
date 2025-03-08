from typing import Callable, Any
import logging
import sys
import hydra


# launch hydra using hydra.main to get same functionality as from cli (e.g. create the output directory)
def launch_hydra(config_path: str, config_name: str,
                 task_function: Callable[[Any], Any]):

    logging.getLogger().debug('Hello')

    cmd = ['cmd', '-cn', 'config', '-cd', '.']
    argv_save = sys.argv
    sys.argv = cmd
    try:
        job = hydra.main(config_path=config_path,
                         config_name=config_name,
                         version_base=None)(task_function)

        job()
    finally:
        sys.argv = argv_save
