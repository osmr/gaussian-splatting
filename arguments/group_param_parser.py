from argparse import ArgumentParser


class GroupParamParser:
    @staticmethod
    def export_to_args(param_struct: object,
                       parser: ArgumentParser,
                       name: str,
                       fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(param_struct).items():
            t = type(value)
            value = value if not fill_none else None
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    @staticmethod
    def import_from_args(param_struct: object,
                         args):
        for arg in vars(args).items():
            if arg[0] in vars(param_struct) or ("_" + arg[0]) in vars(param_struct):
                setattr(param_struct, arg[0], arg[1])
