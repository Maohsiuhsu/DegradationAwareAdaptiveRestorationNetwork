import argparse

"""
    init_handler.py
    Every time the model programe is running, the init_handler() function will be called. 
    You sould called it in every programe that may let user to run in the bash, e.g. train.py, inference.py
"""

def init_handler(module_name=None):
    """ Init some args, object or function for runing the model program

    Args:
        module_name (_type_, optional): _description_. Defaults to None.

    Returns:
        loggy (Loggy): loggy logger
    """
    assert module_name != None, "init_handler args' module_name=None is not allow"

    ###
    # Args Processor
    ###
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help="load the config.yaml file path")
    cmd_args = parser.parse_args()
    conf_path = cmd_args.config

    return conf_path