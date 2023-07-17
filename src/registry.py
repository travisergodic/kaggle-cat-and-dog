import logging


logger = logging.getLogger(__name__) 


class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise ValueError(f'Value of a Registry must be a callable! Value: {value}')
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning(f'Key {key} already in registry {self._name}.')
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()
    
    def build(self, **kwargs):
        type = kwargs.pop('type')
        if type is None:
            return None
        obj = self.__getitem__(type)(**kwargs)
        logger.info(f'Build {obj.__class__.__name__} object successfully.')
        return obj 
    

MODEL = Register('model')
OPTIMIZER = Register('optimizer')
SCHEDULER = Register('scheduler')
