import argparse


def parse_params():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tag",
        type=str,
        default=None,
        help="The tag of execution, for record only.",
    )

    parser.add_argument(
        "-framework",
        type=str,
        default="",
        help="Framework to choose.",
    )

    # weight path
    parser.add_argument(
        "-model_weight",
        type=str,
        default=None,
        help="Specify the model weight path to evaluate.",
    )

    # hardware config
    parser.add_argument(
        "-batch_size",
        type=int,
        default=None,
        help="Specify the batch size for training.",
    )
    parser.add_argument(
        "-gpu",
        type=str,
        default="0",
        help="Specify which GPU to use.",
    )

    # specify whether to show detailed logs
    parser.add_argument(
        "-verbose",
        type=str,
        default="false",
        help="Whether to show detailed logs.",
    )

    args = parser.parse_args()

    return args