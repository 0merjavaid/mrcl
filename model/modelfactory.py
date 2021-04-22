import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300, cols=20):

        if "Sin" == dataset:
            if model_type == "representation":

                hidden_size = width
                return [

                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]
            elif model_type == "columnar":
                hidden_size = width
                return [
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"cols": cols, "out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"cols": cols, "out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'hidden', "adaptation": False, "meta": True,
                     "config": {"cols": cols, "out": 1, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": False,
                     "config": {"cols": cols, "out": output_dimension, "in": 1}}
                ]

        elif dataset == "omniglot":
            channels = 64

            return [
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rotate'},
                {"name": 'rep'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}}

            ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
