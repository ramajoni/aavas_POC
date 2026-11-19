"""Model monitoring: data, model (prediction) and concept drift, shipped to OpenObserve.

``drift.report()`` compares a current window of the book with the reference window
the champions were fitted on; ``openobserve`` ships the findings and provisions the
alerts and dashboard. Settings: ``config/monitoring.yaml``.
"""
