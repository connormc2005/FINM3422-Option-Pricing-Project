# FINM3422 Exotic Derivatives Trading Desk Analysis

## Project Overview

This project implements an object-oriented option pricing and portfolio management system for four complex Over-the-Counter (OTC) derivative trades for a Sydney-based investment bank's Exotic Derivatives Trading Desk. The system employs industry-standard valuation methodologies and provides comprehensive risk management capabilities.

## Trading Positions

The system prices and manages four distinct OTC option positions:

1. **European Call Option on BHP Group Ltd** - Strike at 98% of current price, expiry September 15, 2027
2. **American Put Option on Commonwealth Bank (CBA)** - $170 strike, expiry May 15, 2026  
3. **European Up-and-In Barrier Call on Wesfarmers (WES)** - $80 strike, $100 barrier, expiry September 15, 2027
4. **European Basket Call Option** - $175 strike on weighted basket (10% BHP, 35% CSL, 15% WDS, 40% MQG), expiry July 17, 2025

## Technical Approach

### Object-Oriented Architecture
- **Base Option Class**: Common functionality for all option types
- **Specialized Classes**: EuropeanOption, AmericanOption, BarrierOption, BasketOption
- **Inheritance Structure**: Minimizes code duplication while providing appropriate pricing methods

### Valuation Methodologies
- **Black-Scholes**: European options with analytical solutions
- **Binomial Trees**: American options with early exercise capability
- **Monte Carlo Simulation**: Barrier and basket options for path-dependent payoffs

### Market Data Integration
- Real-time ASX equity prices and correlations via yfinance
- Interest rate term structure from Bloomberg market data
- Implied volatilities from traded options
- Dividend yield adjustments for accurate pricing

### Risk Management
- Comprehensive Greek calculations (Delta, Gamma, Theta, Vega, Rho)
- Portfolio-level risk aggregation
- Actionable hedging strategies with specific trade recommendations


## File Structure

```
├── README.md                      # Project documentation
├── trading-desk-analysis.ipynb    # Main analysis and results
├── Option_Classes.py             # Core option pricing classes  
├── functions.py                  # Supporting pricing functions
└── pip-install-requirements.txt  # Python dependencies
```

This implementation demonstrates derivatives pricing and risk management capabilities suitable for professional trading desk operations.