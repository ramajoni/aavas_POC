"""Agentic OS: rebuild the models as a workflow of agents.

    data -> policy -> training -> serving

Each step is an agent with pre-conditions, work and post-conditions (``workflow``).
A run keeps its own workspace - dataset, models, log - under
``reports/agentic_runs/<run id>/``, so nothing it does touches the live dataset or
the champion models until it is promoted. Training engines: in-process
(``engines.local``), an Amazon SageMaker training job (``engines.sagemaker``), or
the AutoGluon Assistant agent (``engines.mlzero``). Serving: in-process, or a
SageMaker Serverless endpoint (``serving``). Settings: ``config/agentic_os.yaml``.
"""
