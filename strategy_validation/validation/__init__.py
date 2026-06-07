"""
validation/ — rigorous statistical-validation layer for the strategy-edge-validator.

Single source of truth for the statistical court. Every report module routes its
stats through here; no inline duplicates. P&L always comes from pnl_engine.pnl().

Modules:
  bootstrap.py            block / stationary bootstrap index generators
  significance.py         stationary-bootstrap, sign, HAC-t mean tests (no degenerate shuffle)
  white_rc.py             White (2000) Reality Check + Hansen (2005) SPA, (count+1)/(B+1)
  monte_carlo.py          bootstrap / block / regime MC of CAGR, Sharpe, MaxDD, Recovery
  serial_dependence.py    Ljung-Box, runs test, Hurst exponent
  entropy.py              Shannon / permutation / sample / approximate entropy
  risk_metrics.py         Sortino, Calmar, recovery factor, tail ratio, Omega
  probability_of_ruin.py  drawdown-breach probabilities (MC, block, classic ROR)
  verdict.py              Block 9 final gate: REJECTED / WEAK / CANDIDATE EDGE
"""
