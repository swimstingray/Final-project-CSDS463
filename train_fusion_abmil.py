import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle

import train_repro_clam as baseline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--merged_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight_decay", type=float, default=0.005)

    # Original ABMIL model configuration.
    parser.add_argument("--model", default="abmil")
    parser.add_argument("--dataset", default="camelyon16")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--feats_size", type=int, default=512)
    parser.add_argument("--q_n", type=int, default=128)
    parser.add_argument("--non_linearity", type=float, default=1)
    parser.add_argument("--dropout_patch", type=float, default=0)
    parser.add_argument("--dropout_node", type=float, default=0)
    parser.add_argument("--average", action="store_true", default=True)
    parser.add_argument("--init_type", default="orthogonal")
    parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])

    args = parser.parse_args()

    assert args.num_epochs >= 2
    assert torch.cuda.is_available(), "CUDA GPU required"

    checkpoint = Path(args.merged_checkpoint)
    output_dir = Path(args.output_dir)

    assert checkpoint.is_file(), checkpoint

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path("datasets") / args.dataset

    bags = pd.read_csv(dataset_dir / f"{args.dataset}.csv")
    train_csv = pd.read_csv(dataset_dir / "train.csv")
    val_csv = pd.read_csv(dataset_dir / "val.csv")

    train_names = set(Path(x).name for x in train_csv["path"])
    val_names = set(Path(x).name for x in val_csv["path"])

    assert train_names.isdisjoint(val_names), "Train/val overlap"

    train_df = bags[
        bags["path"].map(lambda x: Path(x).name in train_names)
    ].copy()

    val_df = bags[
        bags["path"].map(lambda x: Path(x).name in val_names)
    ].copy()

    assert len(train_df) == len(train_names)
    assert len(val_df) == len(val_names)

    model = baseline.get_model(args).cuda()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.num_epochs,
        0.000005
    )

    best_auc = -float("inf")
    history = []

    for epoch in range(1, args.num_epochs):
        # Identical shuffle seeds for both lambda conditions.
        rng = np.random.RandomState(args.seed + epoch)

        train_epoch = shuffle(
            train_df, random_state=rng
        ).reset_index(drop=True)

        val_epoch = shuffle(
            val_df, random_state=rng
        ).reset_index(drop=True)

        train_loss = baseline.train(
            train_epoch, model, criterion, optimizer, args, rng
        )

        val_loss, avg_score, aucs, thresholds = baseline.test(
            val_epoch, model, criterion, optimizer, args, rng
        )

        scheduler.step()

        val_auc = float(np.sum(aucs))
        assert np.isfinite(val_auc)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": val_auc
        })

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"val_auc={val_auc:.6f}",
            flush=True
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(),
                output_dir / "best_validation.pth"
            )

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "merged_checkpoint": str(checkpoint),
                "best_validation_auc": best_auc,
                "epochs": history
            },
            f,
            indent=2
        )

    print("BEST VALIDATION AUC:", best_auc)
    print("OUTPUT DIRECTORY:", output_dir)
    print("POST-FUSION TRAINING COMPLETE")


if __name__ == "__main__":
    main()
