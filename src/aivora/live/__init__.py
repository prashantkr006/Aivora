"""Live trading layer — paper and real execution.

The layer above :mod:`aivora.pipeline` and :mod:`aivora.ml`.  It
does not learn or engineer features itself; it consumes the
already-trained UP/DOWN binary pair and drives it against a
5-minute tick of live market data.

Strict source-of-truth rule: every capital / P&L number the UI
displays is read from the ``Portfolio`` JSON file.  Nothing lives
in memory that isn't reflected there.
"""
