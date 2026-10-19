"""Notebook display helper: shows the SAME PNG files that were saved to Drive."""

from .context import Artifact


def show(artifacts: list[Artifact], max_table_rows: int = 20) -> None:
    from IPython.display import Image, Markdown, display
    import pandas as pd
    for artifact in artifacts:
        if artifact.kind == "figure":
            display(Markdown(f"**{artifact.relative}** — {artifact.caption}"))
            display(Image(filename=str(artifact.path)))
            continue
        display(Markdown(f"Tabla guardada: `{artifact.relative}` — {artifact.caption}"))
        if artifact.kind == "table":
            frame = pd.read_csv(artifact.path)
            if len(frame) <= max_table_rows:
                display(frame)
