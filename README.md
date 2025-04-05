# Portfolio Pulse

<!-- Table of Contents -->
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Environment Variables](#environment-variables)
- [Contributions](#contributions)
- [License](#license)

<!-- Introduction -->
**Portfolio Pulse** is an intuitive and comprehensive portfolio analysis tool designed for investors to analyze and optimize their stock portfolios. Leveraging modern portfolio theory, it offers insights into historical performance, risk metrics, correlations, and optimized asset allocation. It also features an interactive Equal-Weighted Index Dashboard for detailed market insights.

## Features

- **Stock Selection:** Easily add stocks from the S&P 500 to your portfolio for analysis.
- **Historical Data Analysis:** Visualize historical closing prices, daily returns, and cumulative returns.
- **Equal-Weighted Index Dashboard:** View interactive performance plots, summary metrics, rolling volatility and drawdown charts, and composition changes.
- **Correlation Matrix:** Examine the correlation between the stocks in your portfolio for effective diversification.
- **Risk and Return Metrics:** Analyze daily returns, annualized volatility, and return per unit of risk.
- **Portfolio Optimization:** Optimize your portfolio using the efficient frontier for the maximum Sharpe ratio.
- **Visualizations:** Generate interactive charts and plots for enhanced data presentation.
- **Performance Metrics:** Track KPIs like expected annual return, annual volatility, and Sharpe ratio.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/portfolio-pulse.git
   cd portfolio-pulse
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Install system dependencies (see packages.txt):
   ```sh
   sudo xargs apt install -y < packages.txt
   ```

4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage

1. Open the app in your web browser.
2. Navigate between different dashboards (e.g., Equal-Weighted Index Dashboard).
3. For the Equal-Weighted Index Dashboard:
   - Select a date and period.
   - Click "Analyze" to generate insights and visualizations.

## Development

- A devcontainer is provided in the `.devcontainer/` directory for a consistent development environment.
- Open the project in VS Code using the Remote - Containers extension or Codespaces.
- The devcontainer automatically installs system packages from `packages.txt` and Python dependencies from `requirements.txt`.

## Environment Variables

Create a `.env` file in the root directory with your API keys:
```
ALPHAVENTAGE=your_alpha_vantage_key
FMP_API_KEY=your_fmp_api_key
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
