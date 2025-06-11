
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
