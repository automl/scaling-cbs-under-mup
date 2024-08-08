from litgpt.config import Config
from mup import get_shapes, make_base_shapes

from scales.model import GPT_Scales

if __name__ == "__main__":
    width = 32

    # Create your base config
    base_config = Config(block_size=512, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=width)

    # Create the delta config (ie change the scaling dimension)
    delta_config = Config(block_size=512, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=width * 4)

    # Get model shapes
    base_model = get_shapes(GPT_Scales(base_config, mup=True))
    delta_model = get_shapes(GPT_Scales(delta_config, mup=True))

    make_base_shapes(base_model, delta_model, "width32.bsh")
    print("Scaling shape saved!")
