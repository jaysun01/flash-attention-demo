import torch
import argparse
from torch.utils.data import DataLoader
from experiment import SimpleTransformerModel, RandomDataset

def validate_model_consistency(model_baseline, model_flash, dataloader, device, tolerance=1e-4):
    """
    Validate that the outputs of the baseline model and FlashAttention model are consistent.
    Args:
        model_baseline: Transformer model using standard attention.
        model_flash: Transformer model using FlashAttention.
        dataloader: DataLoader for generating test data.
        device: Device (e.g., cuda or cpu) to run the models on.
        tolerance: Maximum allowable difference between outputs of the two models.
    Returns:
        Boolean indicating whether the outputs are consistent.
    """
    model_baseline.eval()
    model_flash.eval()
    
    consistent = True
    for i, (inp, _) in enumerate(dataloader):
        inp = inp.to(device)

        with torch.no_grad():
            out_baseline = model_baseline(inp)
            out_flash = model_flash(inp)

        max_diff = torch.max(torch.abs(out_baseline - out_flash)).item()
        if max_diff > tolerance:
            consistent = False
            print(f"Discrepancy found in batch {i}: max difference = {max_diff}")
        else:
            print(f"Batch {i}: Outputs consistent (max difference = {max_diff})")

    return consistent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = RandomDataset(args.num_samples, args.seq_length, args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model_baseline = SimpleTransformerModel(
        dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        vocab_size=args.vocab_size,
        flash=False
    ).to(device)

    model_flash = SimpleTransformerModel(
        dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        vocab_size=args.vocab_size,
        flash=True
    ).to(device)

    print("Validating model consistency...")
    consistent = validate_model_consistency(model_baseline, model_flash, dataloader, device, tolerance=args.tolerance)

    if consistent:
        print("Validation successful: Baseline and FlashAttention models produce consistent outputs.")
    else:
        print("Validation failed: Discrepancies found between Baseline and FlashAttention models.")
