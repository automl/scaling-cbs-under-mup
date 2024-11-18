from __future__ import annotations
import gpytorch
import torch
import math

from pathlib import Path
from scales.tblog_utils import read_csv_exp_group
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import yaml
from pathlib import Path
import numpy as np

def get_concat_df(exp_folder: str, hparams: list[str], force_reload: bool = False) -> pd.DataFrame:
    csv_name = "concat_results.csv"
    csv_path = Path(exp_folder + "/" + csv_name)
    if csv_path.exists():
        csv_columns = pd.read_csv(csv_path, index_col=[0, 1], nrows=0, float_precision="round_trip").columns.tolist()
        # and if all hparams in csv columns
        if all([hp in csv_columns for hp in hparams]):
            df = pd.read_csv(csv_path, index_col=[0, 1], float_precision="round_trip")
            df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
            return df
    df = read_csv_exp_group(Path(exp_folder), hparams=hparams, force_reload=force_reload)
    df.to_csv(str(csv_path))
    return df

def aggregate_results(df: pd.DataFrame, target_cols: list[str] | None, last_n: int = 5) -> pd.DataFrame:
    target_cols = target_cols or ["Train Loss"]
    # Mean last_n rows of all columns
    aggregate_cols = df.groupby(level=0).apply(lambda group: group.tail(last_n).mean(numeric_only=True))
    # print(aggregate_cols[target_cols[0]])
    new_df = df.loc[df.groupby(level=0).tail(1).index, :].copy()
    # print(new_df[target_cols[0]])
    for col in target_cols:
        new_df.loc[(aggregate_cols.index, ), f"mean {col}"] = aggregate_cols[col].values
    return new_df


def combine_scales(df_list: list, scale_column: str = "scale"):
    for idx, df in enumerate(df_list):
        df[scale_column] = idx
    combined = pd.concat(df_list)
    combined.set_index(scale_column, append=True, inplace=True)
    return combined


# Simplest GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel:
    def __init__(self,
                 gp: gpytorch.models.GP, 
                 likelihood: gpytorch.likelihoods.Likelihood,
                 ) -> None:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None

    
    def preprocess_dataframe(self, 
                             df: pd.DataFrame, 
                             target_col: str, 
                             scale_col: str,
                             log_cols: list[str],
                             input_scaler = MinMaxScaler()) -> tuple[torch.Tensor, torch.Tensor]:
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        cont_cols = [col for col in df.columns if col not in log_cols and col not in categorical_cols and col != target_col]

        # Pipeline for preprocessing log-scaled, continuous, and categorical columns
        self.input_preprocessor = ColumnTransformer(transformers=[
            ('log_scaled', Pipeline([('log', FunctionTransformer(np.log1p)), ('scale', input_scaler)]), log_cols),
            ('continuous', input_scaler, cont_cols),
            ('categorical', Pipeline([('ordinal', OrdinalEncoder()), ('scale', input_scaler)]), categorical_cols)
        ])

        self.y_mean = df.loc[:, target_col].mean()
        self.y_std = df.loc[:, target_col].std()

        # Preprocess the input data
        X = self.input_preprocessor.fit_transform(df.loc[:, categorical_cols + log_cols + cont_cols].copy(deep=True))
        y = (df.loc[:, target_col].values- self.y_mean)

        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y
    
    def train(self, train_x: torch.Tensor, train_y: torch.Tensor, 
              lr: float = 0.1, train_iters: int = 6000) -> None:
        self.model = ExactGPModel(train_x, train_y, self.likelihood)
        self.model.train()
        self.likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Loss function is the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict(self, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
        return observed_pred.mean, observed_pred.variance
    

def load_config_grid(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        config_grid = yaml.safe_load(stream)
    return config_grid
    
def get_hp_name(hp: str) -> str:
    return hp.split(".")[-1]
    
def collect_single_scale_data(path: Path, 
                              config_grid_path: Path,
                              scale: int, 
                              hp_keys: list[str] | None, 
                              target_cols: list[str] | None = None) -> pd.DataFrame:
    target_cols = target_cols or ["Train Loss", "Validation Loss"]
    if hp_keys is None:
        config_grid = load_config_grid(config_grid_path)
        hp_names = [get_hp_name(hp) for hp in config_grid.keys()]
    else:
        hp_names = hp_keys
    concat_df = get_concat_df(str(path), hp_names, False)
    # print("IN")
    # print(concat_df.head())
    aggr_df = aggregate_results(concat_df, None)
    aggr_df["scale"] = scale
    drop_cols = [col for col in aggr_df.columns if col not in ["scale"] + target_cols + hp_names]
    print(f"Dropping cols: n={len(drop_cols)} {drop_cols}")
    aggr_df = aggr_df.drop(drop_cols, axis=1)
    return aggr_df

class ExpDataManager:
    def __init__(self, 
                 config_folders: list[Path]) -> None:
        self.config_grid_paths = [folder / "config_grid.yaml" for folder in config_folders]
        self.results_folders = [self._get_results_folder(folder) for folder in config_folders]
        self.config_grids = [load_config_grid(path) for path in self.config_grid_paths]

    def get_hp_names(self, scale: int):
        return [get_hp_name(hp) for hp in self.config_grids[scale].keys()]
    

    def get_single_df(self, scale:int, drop_cols: list[str]) -> pd.DataFrame:
        hp_names = self.get_hp_names(scale)
        df = get_concat_df(str(self.results_folders[scale]), hp_names, False)
        # df_list = [get_concat_df(str(folder), hparams, force_reload) for folder in self.results_folders]
        return df
    
    def aggregate_single_df(self, scale:int, target_col: str = "Train Loss", aggregate_steps: int = 5) -> pd.DataFrame:
        df = self.get_single_df(scale)
        aggr_df = aggregate_results(df, target_cols=[target_col], last_n=aggregate_steps)
        aggr_df["scale"] = scale
        return aggr_df
    
    def get_combined_df():

    
    @staticmethod
    def _get_results_folder(exp_folder: Path) -> Path:
        return Path(str(exp_folder).replace("/configs/", "/results/"))
