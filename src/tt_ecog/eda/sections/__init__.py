"""EDA sections, one per preprocessing decision, in notebook order."""

from . import (s01_overview, s02_channels, s03_reference, s04_band, s05_notch, s06_morlet,
               s07_decimation, s08_glove, s09_delay, s10_scaling, s11_validation)

SECTIONS = [s01_overview, s02_channels, s03_reference, s04_band, s05_notch,
            s06_morlet, s07_decimation, s08_glove, s09_delay, s10_scaling,
            s11_validation]

__all__ = ["SECTIONS", "s01_overview", "s02_channels", "s03_reference", "s04_band", "s05_notch",
           "s06_morlet", "s07_decimation", "s08_glove", "s09_delay", "s10_scaling", "s11_validation"]
