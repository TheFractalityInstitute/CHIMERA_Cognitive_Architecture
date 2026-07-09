# chimera_core/collective/__init__.py
"""
CHIMERA Collective — pool independent CHIMERA nodes into shared intelligence.

The working implementation lives in ``hub`` (the shared-knowledge server) and
``client`` (the node side). Import those directly, e.g.::

    from chimera_core.collective.hub import CollectiveHub
    from chimera_core.collective.client import CollectiveNode, SocketNodeClient

The older modules in this package (``server.py``, ``mobile_client.py``,
``mesh_network.py``, ``hybrid_architecture.py``) target an earlier, never-wired
"embodied/quantum" agent interface and do not run. They are kept as reference
only — see SALVAGE.md. This package root intentionally imports nothing heavy.
"""

__all__ = ["hub", "client"]
