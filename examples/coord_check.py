import os

import lightning as L
import numpy as np
from litgpt.config import Config
from litgpt.utils import CycleIterator
from mup import set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data

from scales.config.data_config import DataHandler
from scales.config.utils import preprocess_wikitext
from scales.model import GPT_Scales


def coord_check(mup, lr, train_loader, nsteps, nseeds, plotdir="", legend=False):  # type: ignore
    def gen(w, standparam=False):  # type: ignore
        def f():  # type: ignore
            model_config = Config(block_size=1024, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=w)
            model = GPT_Scales(model_config, True)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, "width32.bsh")
            return model

        return f

    widths = 2 ** np.arange(6, 12)
    models = {w: gen(w, standparam=not mup) for w in widths}

    train_iterator = CycleIterator(train_loader)
    test_batch = []
    for i, batch in enumerate(train_iterator):
        test_batch.append(
            (batch[:, 0:1024].contiguous().long(), (batch[:, 1 : (1024 + 1)].contiguous().long().reshape(-1)))
        )
        if i == nsteps:
            break

    df = get_coord_data(
        models,
        test_batch,
        mup=mup,
        lr=lr,
        optimizer="adamw",
        flatten_output=True,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn="xent",
        cuda=False,
    )

    prm = "μP" if mup else "SP"
    return plot_coord_data(
        df,
        legend=legend,
        save_to=os.path.join(plotdir, f"{prm.lower()}_gpt_adamw_coord.png"),
        suptitle=f"{prm} GPT AdamW lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )


if __name__ == "__main__":
    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_overwrite=False,
        force_splits=True,
        subsample_index=0,
    )

    fabric = L.Fabric(devices="auto", strategy="auto")

    data.load_data_loaders(batch_size=1, block_size=1024, access_internet=True)
    train_dataloader = data.data_loaders["train"]
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    print(len(train_dataloader))

    print("testing parametrization")
    import os

    os.makedirs("coord_checks", exist_ok=True)
    plotdir = "coord_checks"
    coord_check(
        mup=True,
        lr=0.01,
        train_loader=train_dataloader,
        nsteps=3,
        nseeds=5,
        plotdir=plotdir,
        legend=False,
    )
    coord_check(
        mup=False,
        lr=0.01,
        train_loader=train_dataloader,
        nsteps=3,
        nseeds=5,
        plotdir=plotdir,
        legend=False,
    )
    import sys

    sys.exit()
