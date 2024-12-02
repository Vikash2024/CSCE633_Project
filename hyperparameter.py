from train import *
import itertools


def tune_parameters(train_dataloader, val_dataloader, lr, epochs, input_dim, output_dim, 
                    model_dim_list, num_heads_list, num_layers_list, additional_features_dim_list):
    """
    Tunes hyperparameters for the Transformer model.
    
    Parameters:
        train_dataloader (torch.utils.data.dataloader): Training data.
        val_dataloader (torch.utils.data.dataloader): Validation data.
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        model_dim_list (list[int]): List of model dimensions to try.
        num_heads_list (list[int]): List of head counts to try.
        num_layers_list (list[int]): List of layer counts to try.
        additional_features_dim_list (list[int]): List of additional feature dimensions to try.

    Returns:
        dict: Dictionary containing the best model, its parameters, and validation score.
    """
    best_model = None
    best_score = float('inf')
    best_params = None
    results = []

    # Generate all combinations of hyperparameters
    param_combinations = itertools.product(model_dim_list, num_heads_list, num_layers_list, additional_features_dim_list)

    for model_dim, num_heads, num_layers, additional_features_dim in param_combinations:
        print(f"Testing combination: model_dim={model_dim}, num_heads={num_heads}, "
              f"num_layers={num_layers}, additional_features_dim={additional_features_dim}")

        # Train the model with the current hyperparameters
        model = trainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr=lr,
            epochs=epochs,
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            additional_features_dim=additional_features_dim,
            output_dim=output_dim
        )

        
        val_score = validate(model, val_dataloader)  

        print(f"Validation Score: {val_score}")

        # Track results
        results.append({
            "model": model,
            "params": {
                "model_dim": model_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "additional_features_dim": additional_features_dim,
            },
            "val_score": val_score
        })

        # Update best model if current one is better
        if val_score < best_score:
            best_model = model
            best_score = val_score
            best_params = {
                "model_dim": model_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "additional_features_dim": additional_features_dim
            }
            

    print(f"Best Model Found: {best_params} with Validation Score: {best_score}")
    model_state_dir = "best_model_state"
    os.makedirs(model_state_dir, exist_ok=True)
    filepath_best_model = os.path.join(model_state_dir, f'Best_Model_{model_dim}_{num_heads}_{num_layers}')
    torch.save(best_model.state_dict(), filepath_best_model)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "results": results
    }


def validate(model, val_dataloader):
    model.eval()
    with torch.no_grad():
        for cgm_sequence, tabular_features , test_labels in val_dataloader:
            predictions = model(cgm_sequence, tabular_features )
            test_labels = test_labels
            predictions = predictions
            rmsre_val = rmsre_loss(test_labels.numpy(),predictions.squeeze().numpy())
        return rmsre_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSCE633 FINAL PROJECT")
    parser.add_argument('--runtime',  default = 'testrun')
    args = parser.parse_args()
    runtime = args.runtime
    train_dataloader, val_dataloader = read_dataset()
    best_result = tune_parameters(
                                    trainer=trainer,
                                    train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader,
                                    lr=0.01,
                                    epochs=500,
                                    input_dim=2,
                                    output_dim=1,
                                    model_dim_list=[64, 128],
                                    num_heads_list=[4],
                                    num_layers_list=[2],
                                    additional_features_dim_list=[28]
                                )

    print("Best Parameters:", best_result["best_params"])
