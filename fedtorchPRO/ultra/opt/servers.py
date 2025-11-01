from .server import FedOPT
from fedtorchPRO.superconfig import Parser

class FedAVGPlus(FedOPT):

    def _init(self):
        self._convert_FedAVG()
        return super()._init()

class FedSGD(FedOPT):

    def _init(self):
        self.arguments.config['fedopt']['method'] = 'sgd'
        self.arguments.config['fedopt']['momentum'] = 0
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config
        return super()._init()

class FedSGDM(FedOPT):

    def _init(self):
        self.arguments.config['fedopt']['method'] = 'sgd'
        self.arguments.config['fedopt']['momentum'] = 0.9
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config
        return super()._init()

class FedADAM(FedOPT):

    def _init(self):
        self.arguments.config['fedopt']['method'] = 'adam'
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config
        return super()._init()

class FedAMS(FedOPT):

    def _init(self):
        self.arguments.config['fedopt']['method'] = 'adam'
        self.arguments.config['fedopt']['ams'] = True
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config
        return super()._init()

class FedAdagrad(FedOPT):

    def _init(self):
        self.arguments.config['fedopt']['method'] = 'adagrad'
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config
        return super()._init()