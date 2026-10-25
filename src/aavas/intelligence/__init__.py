"""Sales-note intelligence: what the notes say about competitors, read and counted.

The structured application carries one competitor name and one rate. The notes
carry what actually happened - a second lender, a WhatsApp screenshot instead of
a sanction letter, a lender the structured field never had a slot for. This
package reads that text deterministically and keeps every claim traceable to the
span it came from.

    registry          who we compete with, and how branches write their names
    note_extraction   one note -> competitor mentions, quoted rates, evidence,
                      customer signals; current-loan merge; historical batch
    quality           the extractor scored against the notes' latent truth
                      (measurable only because the data is synthetic)
    competitors       historical analysis joined to outcomes, and the per-loan
                      competitor context a reviewer sees

None of it moves a price. Competitor context is for the human reading the file;
the pricing path's handling of competitor claims is unchanged (see the docstring
of ``agents/sales_context_agent.py``).
"""
