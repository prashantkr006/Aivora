"""Multi-user web application layer.

Adds authentication, encrypted broker credentials, and per-user
data isolation on top of the existing single-user modules.  Runs
alongside — not in place of — the original single-user code so an
in-progress paper-trading session isn't disrupted.
"""
