# chimera_core/__init__.py
"""
CHIMERA Cognitive Architecture
==============================

An experimental multi-agent cognitive architecture that develops language and
understanding organically through interaction.

This package root is intentionally lightweight. Importing ``chimera_core`` must
NOT eagerly pull in optional/heavy subsystems (sensors, collective networking,
mobile UI), many of which are aspirational or have unmet dependencies. Import
the concrete piece you need directly, e.g.::

    from chimera_core.language.chimera_language_learning import OrganicLearningSystem

See SALVAGE.md for the current status of each subsystem.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
