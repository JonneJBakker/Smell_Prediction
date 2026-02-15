from torch import nn

from models.simple_mlp import SimpleMLP



def build_model(model_name: str, best_params=None, **kwargs) -> nn.Module:
    """
    Factory function to build a model based on a string name.
    Allows passing in a dict of best_params plus extra kwargs.
    In case of conflicts, kwargs override best_params.
    """

    # If best_params is None, just create an empty dict for merging.
    if best_params is None:
        best_params = {}

    # Merge best_params (lower priority) with kwargs (higher priority).
    # So if the same key is in both best_params and kwargs,
    # the value in kwargs wins.
    all_params = {**best_params, **kwargs}

    model_name_lower = model_name.lower()
    print(model_name_lower)
    print(all_params)

    # if model_name_lower == "hybrid-gnn":
    #     model = HybridGNN(
    #         in_channels=all_params["in_channels"],
    #         hidden_channels=all_params["hidden_channels"],
    #         num_numerical_features=all_params["num_numerical_features"],
    #         num_gcn_layers=all_params.get("num_gcn_layers", 2),
    #         num_encoder_layers=all_params.get("num_mlp_layers", 2),
    #         num_final_layers=all_params.get("num_mlp_layers", 2),
    #         fusion=all_params.get("fusion", "cat"),
    #         dropout=all_params.get("dropout", 0.0),
    #     )

    # elif model_name_lower == "hybrid-gat":
    #     model = HybridGAT(
    #         in_channels=all_params["in_channels"],
    #         hidden_channels=all_params["hidden_channels"],
    #         num_numerical_features=all_params["num_numerical_features"],
    #         num_gat_layers=all_params.get("num_gcn_layers", 2),
    #         num_encoder_layers=all_params.get("num_mlp_layers", 2),
    #         num_final_layers=all_params.get("num_mlp_layers", 2),
    #         fusion=all_params.get("fusion", "cat"),
    #         dropout=all_params.get("dropout", 0.0),
    #     )

    # elif model_name_lower == "gnn":
    #     model = GNN(
    #         in_channels=all_params["in_channels"],
    #         hidden_channels=all_params["hidden_channels"],
    #         num_gcn_layers=all_params.get("num_gcn_layers", 2),
    #         dropout=all_params.get("dropout", 0.0),
    #     )

    if model_name_lower == "mlp":
        model = SimpleMLP(
            input_dim=all_params["num_numerical_features"],
            hidden_channels=all_params["hidden_channels"],
            num_layers=all_params.get("num_mlp_layers", 3),
            dropout=all_params.get("dropout", 0.0),
            output_dim=138,
        )

    # elif model_name_lower == "linear":
    #     num_numerical_features = all_params.get("num_numerical_features", 50)
    #     model = LinearModel(num_numerical_features=num_numerical_features)

    # elif model_name_lower == "naive":
    #     model = NaiveBaselineModel()

    # elif model_name_lower == "chemberta":
    #     # ChemBERTa model is handled separately in the main script
    #     # This is just a placeholder to avoid the ValueError
    #     # The actual model is created in run_chemberta_finetuning
    #     model = SimpleMLP(
    #         input_dim=all_params.get("num_numerical_features", 1),
    #         hidden_channels=all_params.get("hidden_channels", 64),
    #         num_layers=all_params.get("num_mlp_layers", 2),
    #         dropout=all_params.get("dropout", 0.0),
    #         output_dim=1,
    #     )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model
