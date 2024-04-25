# TODO: Update this
# scales-n-arpeggios

My tentative folder structure:

```
├── neps (built from source with .toml changes)
├── litgpt (built from source, with changes)
├── lm-evaluation-harness (built from source)
├── scales-n-arpeggios
│   ├── scales
│   │   ├── ...
│   ├── tests
│   │   ├── ...
│   ├── conda-env.yml
```


## Installation

This will install necessary pre-commit and other maintenance tools (some subset of [these tools](https://github.com/automl/automl_template?tab=readme-ov-file#features))
```commandline
git clone https://github.com/automl/scales-n-arpeggios.git
cd scales-n-arpeggios
conda create -n scales python=3.11
conda activate scales

# Install for development
make install-dev
```

Then install `neps` and `litgpt` for development

## Note for PyCharm
When building the dependencies (`litgpt`, `lm-evaluation-harness`), use:

```commandline
pip install -e ... --config-settings editable_mode=compat
```
instead of:
```commandline
pip install -e ...
```
Refer [here](https://stackoverflow.com/questions/76301782/why-are-pycharm-and-pylance-not-detecting-packages-installed-in-editable-mode/76301809#76301809)

## Tasks _(in no particular order)_
- [ ] Parametrize dataloader module (Start with TinyLlama)
- [ ] Add support for different data modules
- [ ] Add `dataset_size` parameter to the dataloaders
- [ ] Auto-Assign "optimal" Dataset size based on model size
- [ ] Fix Tokenizer
- [ ] Add Evaluation interface with `lm-evaluation-harness` (through `litgpt evaluate`)
- [ ] Test with a simple run on a cluster
- [ ] Add Yaml config interface for `neps.api.run` calls
- [ ] CLI for easily running scripts with yaml configs
- [x] ~~Add pre-train <---> neps interface~~
- [x] ~~Return Validation perplexity from `litgpt` for neps~~


## Minimal Example

```
# Your code here
```
