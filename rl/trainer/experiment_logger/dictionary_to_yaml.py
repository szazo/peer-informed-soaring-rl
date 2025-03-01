import ruamel.yaml
from ruamel.yaml.compat import StringIO


def dictionary_to_yaml(dictionary: dict):
    yaml = ruamel.yaml.YAML(typ='safe')
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(dictionary, stream)
    return stream.getvalue()
