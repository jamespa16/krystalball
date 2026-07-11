"""CLI configuration for krystalball. All tunables live here, not buried in main.py."""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live webcam next-frame prediction, trained online in real time."
    )
    p.add_argument("--camera-index", type=int, default=0,
                    help="OpenCV camera index (default: 0)")
    p.add_argument("--width", type=int, default=96,
                    help="Internal working width in pixels; must be divisible by 4 (default: 96)")
    p.add_argument("--height", type=int, default=72,
                    help="Internal working height in pixels; must be divisible by 4 (default: 72)")
    p.add_argument("--upscale", type=int, default=6,
                    help="Factor to upscale each pane by for display only (default: 6)")
    p.add_argument("--lr", type=float, default=1e-3,
                    help="Optimizer learning rate (default: 1e-3)")
    p.add_argument("--hidden-channels", type=int, default=32,
                    help="Channel count of the ConvLSTM hidden/cell state (default: 32)")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                    help="Optimizer to use (default: adam)")
    p.add_argument("--loss", choices=["mse"], default="mse",
                    help="Loss function between prediction and real frame (default: mse)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Compute device; 'auto' picks cuda if available (default: auto)")
    p.add_argument("--loss-window", type=int, default=100,
                    help="Number of recent losses averaged for the on-screen readout (default: 100)")
    return p
