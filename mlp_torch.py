import pandas as pd
import numpy as np
import optuna
import csv
import shap
from sklearn.model_selection import train_test_split
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import traceback
import sklearn
import torch
import torch.utils.data
pd.options.mode.chained_assignment = None  # default='warn'
from utils import impute_data, encode_data, get_indexes, drop_columns, run_optuna_optimization

target_column = "climatedeniers_2"
dataset = "Climate_Deniers_15K.csv"

class ClimateDeniersClassifier:

    def __init__(self):
        self.current_best_val_result = None
        self.early_stopping_point = None

    def preprocess_data(self, impute):
        data = pd.read_csv(dataset)

        if target_column == "climatedeniers_2":
            drop_columns(data,
                              ["climate_eb_problem", "climate_state_convinced", "climatedeniers_1", "climate_risk",
                               "d_q1_own_opinion"])
        elif target_column == "climatedeniers_1":
            drop_columns(data,
                              ["climate_eb_problem", "climate_state_convinced", "climatedeniers_2", "climate_risk",
                               "d_q1_own_opinion"])

        if impute:
            data = impute_data(data)
        else:
            data.dropna(inplace=True)

        data = encode_data(data)

        self.data = data.copy()

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader):
        self.model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.get_loss(outputs=outputs, targets=targets)
            # l1_loss = 0
            # for param in self.model.parameters():
            #     l1_loss += torch.sum(torch.abs(param))
            # loss += self.l1_factor * l1_loss
            loss.backward()
            self.optimizer.step()

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate one epoch

        :param val_loader: DataLoader with validation data

        :return: loss based on loss-criterion
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
                outputs = self.model(inputs)
                total_loss += self.get_loss(outputs=outputs, targets=targets).item()
        return total_loss / len(val_loader.dataset)

    def retrain(self, X_retrain: np.array, y_retrain: np.array):
        """
        Implementation of the retraining for PyTorch models.
        See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        self.X_scaler = sklearn.preprocessing.StandardScaler()
        X_retrain_standard = pd.DataFrame(self.X_scaler.fit_transform(X_retrain))
        X_retrain_standard.columns = X_retrain.columns
        retrain_loader = self.get_dataloader(X=X_retrain_standard, y=y_retrain)
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.to(device=self.device)
        for epoch in range(n_epochs_to_retrain):
            if epoch % 100 == 0:
                print('Retrain: Epoch ' + str(epoch + 1) + ' of ' + str(n_epochs_to_retrain))
            self.train_one_epoch(retrain_loader)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        X_in_standard = pd.DataFrame(self.X_scaler.transform(X_in))
        dataloader = self.get_dataloader(X=X_in_standard, shuffle=False)
        self.model.eval()
        predictions = None
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device=self.device)
                outputs = self.model(inputs)
                predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
        _, predictions = torch.max(predictions, 1)
        return predictions.cpu().detach().numpy()

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss based on the outputs and targets

        :param outputs: outputs of the model
        :param targets: targets of the dataset

        :return: loss
        """
        if type(self.loss_fn) in [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss]:
            targets = targets.long()
        return self.loss_fn(outputs, targets)

    def get_dataloader(self, X: np.array, y: np.array = None, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Get a Pytorch DataLoader using the specified data and batch size

        :param X: feature matrix to use
        :param y: optional target vector to use
        :param shuffle: shuffle parameter for DataLoader

        :return: Pytorch DataLoader
        """
        # drop last sample if last batch would only contain one sample
        if (len(X) % self.batch_size) == 1:
            X = X[:-1]
            y = y.iloc[:-1] if y is not None else None
        X = torch.tensor(X.values).float()
        # X = torch.swapaxes(X, 1, 2)
        y = torch.reshape(torch.tensor(y.values).float(), (-1, 1)) if y is not None else None
        y = torch.flatten(y) if y is not None else y
        dataset = torch.utils.data.TensorDataset(X, y) if y is not None else X
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> np.array:
        """
        Implementation of a train and validation loop for  PyTorch models.
        See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        self.X_scaler = sklearn.preprocessing.StandardScaler()
        X_train_standard = pd.DataFrame(self.X_scaler.fit_transform(X_train))
        X_train_standard.columns = X_train.columns
        X_val_standard = pd.DataFrame(self.X_scaler.fit_transform(X_val))
        X_val_standard.columns = X_val.columns
        train_loader = self.get_dataloader(X=X_train_standard, y=y_train)
        val_loader = self.get_dataloader(X=X_val_standard, y=y_val)
        best_model = copy.deepcopy(self.model)
        self.model.to(device=self.device)
        best_loss = None
        epochs_wo_improvement = 0
        for epoch in range(self.n_epochs):
            self.train_one_epoch(train_loader=train_loader)
            val_loss = self.validate_one_epoch(val_loader=val_loader)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                epochs_wo_improvement = 0
                best_model = copy.deepcopy(self.model)
            else:
                epochs_wo_improvement += 1
            if epoch % 100 == 0:
                print('Epoch ' + str(epoch + 1) + ' of ' + str(self.n_epochs))
                print('Current val_loss=' + str(val_loss) + ', best val_loss=' + str(best_loss))
            if epoch >= 20 and epochs_wo_improvement >= self.early_stopping_patience:
                print("Early Stopping at " + str(epoch + 1) + ' of ' + str(self.n_epochs))
                self.early_stopping_point = epoch - self.early_stopping_patience
                self.model = best_model
                return self.predict(X_in=X_val)
        return self.predict(X_in=X_val)

    def get_torch_object_for_string(self, string_to_get: str):
        string_to_object_dict = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh(),
        }
        return string_to_object_dict[string_to_get] if string_to_get is not None else None

    def objective(self, trial: optuna.trial.Trial):

        sampling = trial.suggest_categorical(
            "sampling",
            [None, "over", "under"])
        sampling_strategy = trial.suggest_categorical(
            "sampling_strategy",
            ["auto", "all", "minmajority", "not majority", "not minority"])
        if sampling == "over" and sampling_strategy == "minmajority":
            sampling_strategy = "minority"
        if sampling == "under" and sampling_strategy == "minmajority":
            sampling_strategy = "majority"
        impute = trial.suggest_categorical(
            "impute",
            [False, True])  # False, True

        self.preprocess_data(impute=impute)
        self.train_val, self.test = train_test_split(self.data, test_size=0.2, random_state=42, shuffle=False)

        self.n_features = self.data.shape[1] - 1
        in_features = self.n_features
        n_layers = trial.suggest_int("n_layers", 1, 5)
        act_function = self.get_torch_object_for_string(
            string_to_get=trial.suggest_categorical("act_function", ["tanh", "relu"])
        )
        out_features = int(in_features * trial.suggest_float("n_initial_units_factor", 0.15, 0.95, step=0.05))
        p = trial.suggest_float("dropout", 0, 0.5, step=0.1)
        perc_decrease = trial.suggest_float("perc_decrease_per_layer", 0.05, 0.5, step=0.05)
        batch_norm = trial.suggest_categorical("batch_norm", [False, True])

        model = []
        for layer in range(n_layers):
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            if act_function is not None:
                model.append(act_function)
            if batch_norm:
                model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p=p))
            in_features = out_features
            out_features = int(in_features * (1 - perc_decrease))
        model.append(torch.nn.Linear(in_features=in_features, out_features=4))
        self.model = torch.nn.Sequential(*model)

        self.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128, 256, 512])
        self.n_epochs = trial.suggest_int("n_epochs", 1000, 10000, step=1000)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=trial.suggest_categorical("learning_rate",
                                                                       [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.l1_factor = self.suggest_hyperparam_to_optuna('l1_factor')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.early_stopping_patience = trial.suggest_int("early_stopping_patience", 0, 100, step=10)
        self.early_stopping_point = None

        early_stopping_points = []
        objective_values = []

        train_indexes, val_indexes = get_indexes(df=self.train_val, target_column=target_column)

        for fold in range(5):
            train, val = (self.train_val.iloc[train_indexes[fold]], self.train_val.iloc[val_indexes[fold]])

            if sampling != None:
                if sampling == "over":
                    sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                else:
                    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                try:
                    train_X_sampled, train_y_sampled = sampler.fit_resample(
                        train.drop(target_column, axis=1), train[target_column]
                    )
                    train = pd.concat([train_X_sampled, train_y_sampled], axis=1)
                    train = train.sample(frac=1).reset_index(drop=True)
                except (sklearn.utils._param_validation.InvalidParameterError) as exc:
                    print(traceback.format_exc())
                    print(exc)
                    print('Trial failed. Error in optim loop.')
                    raise optuna.exceptions.TrialPruned()

            y_pred = self.train_val_loop(
                X_train=train.drop(target_column, axis=1), y_train=train[target_column],
                X_val=val.drop(target_column, axis=1), y_val=val[target_column]
            )

            early_stopping_points.append(
                self.early_stopping_point if self.early_stopping_point is not None else self.n_epochs)

            if len(y_pred) == (len(val[target_column]) - 1):
                # might happen if batch size leads to a last batch with only one sample which will be dropped then
                print('y_val has one element less than y_true (e.g. due to batch size config) -> drop last element')
                val.drop(val.tail(1).index, inplace=True)

            objective_value = matthews_corrcoef(val[target_column], y_pred)

            objective_values.append(objective_value)

        current_val_result = float(np.mean(objective_values))

        if self.current_best_val_result is None or current_val_result > self.current_best_val_result:
            self.current_best_val_result = current_val_result
            if hasattr(model, 'early_stopping_point'):
                # take mean of early stopping points of all innerfolds for refitting of final model
                self.early_stopping_point = int(np.mean(early_stopping_points))

        return current_val_result

    def shap(self):
        test = pd.DataFrame(
            self.X_scaler.transform(self.test.drop(target_column, axis=1)))
        test.columns = self.test.drop(target_column, axis=1).columns

        explainer = shap.Explainer(self.predict, test)
        shap_values = explainer(test)

        filename_expl = 'explainer.sav'
        pickle.dump(explainer, open("explainer/" + filename_expl, 'wb'))

        filename = 'shapvalues.sav'
        pickle.dump(shap_values, open("shapvalues/" + filename, 'wb'))

        for feature in self.data.drop(target_column, axis=1).columns:
            shap.partial_dependence_plot(
                feature,
                self.predict,
                test,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            f = plt.gcf()
            f.savefig("partial_dependence_plots/shap.partial_dependence_plot" + feature + ".pdf", format='pdf',
                      bbox_inches='tight')

    def run_pipeline(self):
        best_params = run_optuna_optimization(self.objective)

        in_features = self.n_features
        n_layers = best_params["n_layers"]
        act_function = self.get_torch_object_for_string(string_to_get=best_params["act_function"])
        out_features = int(in_features * best_params["n_initial_units_factor"])

        model = []
        for layer in range(n_layers):
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            if act_function is not None:
                model.append(act_function)
            if best_params["batch_norm"]:
                model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p=best_params["dropout"]))
            in_features = out_features
            out_features = int(in_features * (1 - best_params["perc_decrease_per_layer"]))
        model.append(torch.nn.Linear(in_features=in_features, out_features=4))
        self.model = torch.nn.Sequential(*model)

        self.batch_size = best_params["batch_size"]
        self.n_epochs = best_params["n_epochs"]
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=best_params["perc_decrease_per_layer"])

        self.retrain(
            self.train_val.drop(target_column, axis=1),
            self.train_val[target_column])

        pickle.dump(self.model, open("models/model_mlp_torch", 'wb'))

        predictions = self.predict(
            X_in=self.test.drop(target_column, axis=1))

        np.savetxt("predictions/predictions_mlp_torch.csv", predictions, delimiter=",")
        self.test.to_csv("testsets/test_mlp_torch.csv", index=False,
                                               sep=',', decimal='.', float_format='%.10f')
        with open('best_params/best_params_mlp_torch.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(best_params.items())
        print(classification_report(y_true=self.test[target_column], y_pred=predictions))
        print(matthews_corrcoef(y_true=self.test[target_column], y_pred=predictions))
        with open('matthews_corrcoef/matthews_corrcoef_mlp_torch.txt', 'w') as f:
            f.write("matthews_corrcoef: %.2f" % matthews_corrcoef(y_true=self.test[target_column],
                                                                  y_pred=predictions))

        self.shap()


climate_deniers_classifier = ClimateDeniersClassifier()
climate_deniers_classifier.run_pipeline()