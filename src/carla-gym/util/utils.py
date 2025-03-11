import yaml


def parse_config(filename, controller_name):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)[controller_name]['ros__parameters']
    return config
