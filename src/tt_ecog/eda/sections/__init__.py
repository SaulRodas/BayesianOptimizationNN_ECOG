"""EDA sections, one per preprocessing decision, in notebook order."""

from . import (s01_overview, s02_channels, s03_reference, s04_band, s05_notch, s06_morlet,
               s07_decimation, s08_glove, s09_delay, s10_scaling, s11_validation)

SECTIONS = [s01_overview.run, s02_channels.run, s03_reference.run, s04_band.run, s05_notch.run,
            s06_morlet.run, s07_decimation.run, s08_glove.run, s09_delay.run, s10_scaling.run,
            s11_validation.run]

__all__ = ["SECTIONS", "s01_overview", "s02_channels", "s03_reference", "s04_band", "s05_notch",
           "s06_morlet", "s07_decimation", "s08_glove", "s09_delay", "s10_scaling", "s11_validation"]
