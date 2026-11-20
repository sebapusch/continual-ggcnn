from __future__ import annotations

import argparse
import csv
import datetime
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from models import get_network
from models.common import post_process_output
from utils.data.grasp_data import GraspDatasetBase
from utils.data.jacquard_data import JacquardDataset
from utils.dataset_processing import evaluation


TASK_ORDER = ("food", "kitchen", "tools", "shapes", "tasks")


@dataclass(frozen=True)
class Sample:
    rgb_path: Path
    grasp_path: Path
    object_id: str


@dataclass(frozen=True)
class Task:
    name: str
    train_samples: list[Sample]
    val_samples: list[Sample]


@dataclass
class Metrics:
    loss: float
    losses: dict[str, float]
    correct: int = 0
    total: int = 0

    @property
    def iou(self) -> float:
        return self.correct / self.total if self.total else 0.0


class CsvJacquardDataset(JacquardDataset):
    def __init__(self, samples: list[Sample], use_depth: bool, use_rgb: bool, augment: bool) -> None:
        GraspDatasetBase.__init__(
            self,
            include_depth=use_depth,
            include_rgb=use_rgb,
            random_rotate=augment,
            random_zoom=augment,
        )
        self.grasp_files = [str(sample.grasp_path) for sample in samples]
        self.depth_files = [path.replace("_grasps.txt", "_perfect_depth.tiff") for path in self.grasp_files]
        self.rgb_files = [str(sample.rgb_path) for sample in samples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GG-CNN on a sequence of Jacquard tasks")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Extracted Jacquard V2 root")
    parser.add_argument("--splits-dir", type=Path, default=Path(__file__).with_name("splits"))
    parser.add_argument(
        "--task-order", nargs="+", default=list(TASK_ORDER), help="CSV stems in training order"
    )
    parser.add_argument("--network", default="ggcnn", help="Network name in models")
    parser.add_argument("--use-depth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use-rgb", type=int, choices=(0, 1), default=0)
    parser.add_argument("--split", type=float, default=0.9, help="Fraction of objects used for training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per task")
    parser.add_argument("--batches-per-epoch", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=250)
    parser.add_argument(
        "--validate-all-seen", action="store_true",
        help="Validate the current task and every earlier task after each epoch",
    )
    parser.add_argument("--description", default="")
    parser.add_argument("--outdir", type=Path, default=Path("output/models"))
    parser.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    args = parser.parse_args()
    if not 0 < args.split < 1:
        parser.error("--split must be between 0 and 1")
    if not args.use_depth and not args.use_rgb:
        parser.error("enable --use-depth or --use-rgb")
    if any(value <= 0 for value in (args.batch_size, args.epochs, args.batches_per_epoch, args.val_batches)):
        parser.error("batch size, epochs, and batch counts must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if len(args.task_order) != len(set(args.task_order)):
        parser.error("--task-order cannot contain duplicate tasks")
    return args


def load_task_sequence(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for name in args.task_order:
        csv_path = args.splits_dir / f"{name}.csv"
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required_fields = {"sample_path", "grasp_path", "object_id", "group"}
            if not reader.fieldnames or not required_fields <= set(reader.fieldnames):
                raise ValueError(f"Missing required columns in {csv_path}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"No samples in {csv_path}")
        if any(row["group"] != name for row in rows):
            raise ValueError(f"Group in {csv_path} does not match task name {name}")

        object_ids = sorted({row["object_id"] for row in rows})
        if len(object_ids) < 2:
            raise ValueError(f"Task {name} needs at least two objects for a train/validation split")
        random.Random(args.seed).shuffle(object_ids)
        train_count = max(1, min(len(object_ids) - 1, int(len(object_ids) * args.split)))
        train_ids = set(object_ids[:train_count])
        dataset_root = args.dataset_path / name if (args.dataset_path / name).is_dir() else args.dataset_path
        samples = [
            Sample(dataset_root / row["sample_path"],
                   dataset_root / row["grasp_path"], row["object_id"])
            for row in rows
        ]
        for sample in samples:
            required = [sample.grasp_path]
            if args.use_rgb:
                required.append(sample.rgb_path)
            if args.use_depth:
                required.append(Path(str(sample.grasp_path).replace("_grasps.txt", "_perfect_depth.tiff")))
            missing = next((path for path in required if not path.is_file()), None)
            if missing is not None:
                raise FileNotFoundError(f"Missing Jacquard file for task {name}: {missing}")
        train_samples = [sample for sample in samples if sample.object_id in train_ids]
        val_samples = [sample for sample in samples if sample.object_id not in train_ids]
        tasks.append(Task(name, train_samples, val_samples))
    return tasks


def make_loaders(task: Task, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = CsvJacquardDataset(
        task.train_samples, bool(args.use_depth), bool(args.use_rgb), augment=True
    )
    val_dataset = CsvJacquardDataset(
        task.val_samples, bool(args.use_depth), bool(args.use_rgb), augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


def train_epoch(
    net: nn.Module, device: torch.device, loader: DataLoader, optimizer: Optimizer, batches: int
) -> Metrics:
    net.train()
    totals: dict[str, float] = {}
    total_loss = 0.0
    iterator = iter(loader)
    for batch_index in range(batches):
        try:
            x, y, _, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _, _ = next(iterator)
        loss_data = net.compute_loss(x.to(device), [target.to(device) for target in y])
        optimizer.zero_grad()
        loss_data["loss"].backward()
        optimizer.step()
        total_loss += loss_data["loss"].item()
        for name, loss in loss_data["losses"].items():
            totals[name] = totals.get(name, 0.0) + loss.item()
        if (batch_index + 1) % 100 == 0:
            logging.info("Batch %d/%d: loss %.4f", batch_index + 1, batches, loss_data["loss"].item())
    return Metrics(total_loss / batches, {name: value / batches for name, value in totals.items()})


def validate(net: nn.Module, device: torch.device, loader: DataLoader, max_batches: int) -> Metrics:
    net.eval()
    totals: dict[str, float] = {}
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for x, y, indices, rotations, zooms in loader:
            loss_data = net.compute_loss(x.to(device), [target.to(device) for target in y])
            total_loss += loss_data["loss"].item()
            for name, loss in loss_data["losses"].items():
                totals[name] = totals.get(name, 0.0) + loss.item()
            prediction: dict[str, Tensor] = loss_data["pred"]
            quality, angle, width = post_process_output(
                prediction["pos"], prediction["cos"], prediction["sin"], prediction["width"]
            )
            ground_truth = loader.dataset.get_gtbb(indices.item(), rotations.item(), zooms.item())
            correct += int(evaluation.calculate_iou_match(quality, angle, ground_truth, grasp_width=width))
            count += 1
            if count >= max_batches:
                break
    average_losses = {name: value / count for name, value in totals.items()}
    return Metrics(total_loss / count, average_losses, correct, count)


def log_train_metrics(writer: SummaryWriter, task: str, epoch: int, train: Metrics) -> None:
    writer.add_scalar(f"{task}/loss/train", train.loss, epoch)
    for name, value in train.losses.items():
        writer.add_scalar(f"{task}/train/{name}", value, epoch)


def log_validation_metrics(
    writer: SummaryWriter, training_task: str, evaluated_task: str, epoch: int, val: Metrics
) -> None:
    prefix = f"{training_task}/validation/{evaluated_task}"
    writer.add_scalar(f"{prefix}/loss", val.loss, epoch)
    writer.add_scalar(f"{prefix}/iou", val.iou, epoch)
    for name, value in val.losses.items():
        writer.add_scalar(f"{prefix}/{name}", value, epoch)


def run_task(
    task: Task, task_index: int, task_count: int, net: nn.Module, optimizer: Optimizer,
    device: torch.device, args: argparse.Namespace, save_dir: Path, writer: SummaryWriter,
    seen_validation_loaders: dict[str, DataLoader],
) -> None:
    logging.info("Task %d/%d: %s (%d train, %d validation samples)",
                 task_index, task_count, task.name, len(task.train_samples), len(task.val_samples))
    train_loader, val_loader = make_loaders(task, args)
    seen_validation_loaders[task.name] = val_loader
    best_iou = -1.0
    for epoch in range(args.epochs):
        logging.info("Task %s, epoch %d/%d", task.name, epoch + 1, args.epochs)
        train = train_epoch(net, device, train_loader, optimizer, args.batches_per_epoch)
        log_train_metrics(writer, task.name, epoch, train)
        validation_loaders = seen_validation_loaders if args.validate_all_seen else {task.name: val_loader}
        for evaluated_task, loader in validation_loaders.items():
            val = validate(net, device, loader, args.val_batches)
            log_validation_metrics(writer, task.name, evaluated_task, epoch, val)
            logging.info(
                "Validation on %s: %d/%d correct, IoU %.4f, loss %.4f",
                evaluated_task, val.correct, val.total, val.iou, val.loss,
            )
            if evaluated_task == task.name and val.iou > best_iou:
                best_iou = val.iou
                torch.save(net.state_dict(), save_dir / f"task_{task_index:02d}_{task.name}_best.pt")
    torch.save(net.state_dict(), save_dir / f"task_{task_index:02d}_{task.name}_final.pt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    tasks = load_task_sequence(args)
    run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    if args.description.strip():
        run_name += "_" + "_".join(args.description.split())
    save_dir = args.outdir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_channels = args.use_depth + 3 * args.use_rgb
    net = get_network(args.network)(input_channels=input_channels).to(device)
    optimizer = Adam(net.parameters())
    logging.info("Training on %s; task order: %s", device, ", ".join(task.name for task in tasks))
    with SummaryWriter(str(args.logdir / run_name)) as writer:
        seen_validation_loaders: dict[str, DataLoader] = {}
        for task_index, task in enumerate(tasks, start=1):
            run_task(
                task, task_index, len(tasks), net, optimizer, device, args, save_dir, writer,
                seen_validation_loaders,
            )


if __name__ == "__main__":
    main()
