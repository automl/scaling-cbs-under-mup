from litgpt.config import Config
from mup import make_base_shapes

from scales.model import GPT_Scales

if __name__ == "__main__":
    width = 32

    # Create your base config
    base_config = Config(block_size=512, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=width)

    # Create the delta config (ie config to scale towards)
    delta_config = Config(block_size=512, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=width * 2)

    # Instantiate the models
    base_model = GPT_Scales(base_config)
    delta_model = GPT_Scales(delta_config)

    make_base_shapes(base_model, delta_model, "width32.bsh")
