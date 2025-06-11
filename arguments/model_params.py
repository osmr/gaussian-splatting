import os
from arguments.param_group import ParamGroup0


class ModelParams:
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.depths = ""
        self.resolution = -1
        self.white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False


class ModelParams0(ParamGroup0):
    def __init__(self,
                 parser,
                 sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
