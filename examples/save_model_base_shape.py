import argparse
from copy import deepcopy
from pathlib import Path

from litgpt.config import Config

from scales.utils import get_mup_shape_diff


def get_args():
    parser = argparse.ArgumentParser(description="Parser for MuP training")

    parser.add_argument("--use_yamls", action="store_true", help="use yaml files for model config")
    parser.add_argument("--base_config", type=str, default=None, help="path to model yaml")
    parser.add_argument("--target_config", type=str, default=None, help="path to model yaml")

    parser.add_argument("--depth", type=int, default=4, help="n_layers of the GPT model")
    parser.add_argument("--block_size", type=int, default=1024, help="width of the GPT model")
    parser.add_argument("--vocab_size", type=int, default=50257, help="vocab size for training")
    parser.add_argument("--base_n_head", type=int, default=2, help="base heads of the GPT model")
    parser.add_argument("--target_n_head", type=int, default=4, help="target heads of the GPT model")
    parser.add_argument("--base_n_embd", type=int, default=32, help="base width of the GPT model")
    parser.add_argument("--target_n_embd", type=int, default=64, help="target width of the GPT model")

    parser.add_argument("--output_dir", type=str, default="./", help="output directory for training")
    parser.add_argument("--filename", type=str, default="width32.bsh", help="output filename for base shape")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not args.use_yamls:
        _config = Config.from_file(Path(__file__).absolute().parent / ".." / "examples" / "model.yaml")
        # Create your base config
        _config.n_layer = args.depth
        _config.block_size = args.block_size
        _config.vocab_size = args.vocab_size
        _config.n_head = args.base_n_head
        _config.n_embd = args.base_n_embd
        base_config = deepcopy(_config)
        # Create the delta config (ie change the scaling dimension)
        _config.n_head = args.target_n_head
        _config.n_embd = args.target_n_embd
        target_config = _config
    else:
        base_config = Config.from_yaml(args.base_config)
        target_config = Config.from_yaml(args.target_config)

    # Set output
    assert Path(args.output_dir).exists(), f"Output directory {args.output_dir} does not exist"
    output_file = Path(args.output_dir) / args.filename
    # Get model shapes
    get_mup_shape_diff(base_config, target_config, output_file, verbose=True)
