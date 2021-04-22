import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()

        self.add('--name', help='Name of experiment', default="plot/")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--path', help='Name of experiment', default="../results/")
        self.add('--metric', help='Name of experiment', default="ss")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--log", action="store_true")