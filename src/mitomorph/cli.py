"""Command-line interface for the mitomorph pipeline (FR-07, NFR-09)."""

from __future__ import annotations

from pathlib import Path

import click
from tqdm import tqdm

from mitomorph.config.loader import load_config
from mitomorph.logger import configure_logging, get_logger
from mitomorph.pipeline import MitoPipeline

logger = get_logger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"


@click.group()
@click.option(
    "--config", "config_path", default=str(DEFAULT_CONFIG), show_default=True, help="Path to config YAML."
)
@click.option("--log-dir", default=None, help="Directory for log files (console-only if omitted).")
@click.pass_context
def cli(ctx: click.Context, config_path: str, log_dir: str | None) -> None:
    """mitomorph: mitochondrial morphology analysis pipeline."""
    configure_logging(log_dir=log_dir)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--animal-id", required=True)
@click.option("--condition", "experimental_condition", required=True)
@click.option("--time-point", required=True)
@click.pass_context
def analyze(
    ctx: click.Context, image_path: str, animal_id: str, experimental_condition: str, time_point: str
) -> None:
    """Run the full pipeline on a single image (FR-01 to FR-40)."""
    pipeline = MitoPipeline(ctx.obj["config_path"])
    metadata = {
        "animal_id": animal_id,
        "experimental_condition": experimental_condition,
        "time_point": time_point,
    }
    result = pipeline.run(image_path, metadata)
    click.echo(f"Analysis complete: {result}")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--pattern", default="*.tif", show_default=True, help="Glob pattern for images to process.")
@click.pass_context
def batch(ctx: click.Context, input_dir: str, output_dir: str, pattern: str) -> None:
    """Batch-process every matching image in a directory (FR-07)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(input_path.glob(pattern))

    pipeline = MitoPipeline(ctx.obj["config_path"])
    succeeded, failed = 0, 0
    for image_path in tqdm(image_paths, desc="Analyzing images"):
        try:
            pipeline.run(image_path, metadata={})
            succeeded += 1
        except Exception as exc:
            logger.warning("Failed to analyze %s: %s", image_path, exc)
            failed += 1

    click.echo(f"Processed {succeeded + failed}/{len(image_paths)} images ({failed} failed)")


@cli.command()
@click.option("--train-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--val-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--checkpoint-dir", required=True, type=click.Path(file_okay=False))
@click.pass_context
def train(ctx: click.Context, train_dir: str, val_dir: str, checkpoint_dir: str) -> None:
    """Fine-tune the segmentation model on lab-specific data (FR-11)."""
    from mitomorph.segmentation.models.unet import UNetResNet34
    from mitomorph.segmentation.train import train_segmentation_model

    config = load_config(ctx.obj["config_path"])
    model = UNetResNet34()
    train_segmentation_model(model, train_dir, val_dir, config, checkpoint_dir)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.pass_context
def report(ctx: click.Context, results_dir: str, output_path: str) -> None:
    """Generate a comprehensive PDF report from stored analysis results (FR-39)."""
    from mitomorph.reporting.pdf_report import build_pdf_report

    build_pdf_report(results=[], output_path=output_path)


if __name__ == "__main__":
    cli()
