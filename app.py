"""Interactive carbon-intensity and valuation regression dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data for project 2 (Emissions)"
    / "valuation_dataset.csv"
)
MAX_EV_TO_EBITDA = 50


@st.cache_data
def load_and_clean_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {
        "parent_company",
        "listed_company_name",
        "ticker",
        "sector",
        "total_scope1_emissions_mtco2e",
        "Carbon_Intensity",
        "enterprise_value_usd",
        "ebitda_usd",
        "ev_to_ebitda",
        "log_market_cap",
        "ebitda_margin",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Valuation dataset is missing columns: {sorted(missing_columns)}"
        )

    numeric_columns = [
        "Carbon_Intensity",
        "total_scope1_emissions_mtco2e",
        "enterprise_value_usd",
        "ebitda_usd",
        "ev_to_ebitda",
        "log_market_cap",
        "ebitda_margin",
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=numeric_columns + ["sector"]).copy()
    data = data[
        (data["enterprise_value_usd"] > 0)
        & (data["ebitda_usd"] > 0)
        & (data["ev_to_ebitda"] <= MAX_EV_TO_EBITDA)
    ].copy()

    data["Sector"] = data["sector"].astype(str).str.strip()
    data = data[data["Sector"] != ""].copy()
    return data.sort_values("Carbon_Intensity").reset_index(drop=True)


def run_ols(data: pd.DataFrame):
    return smf.ols(
        formula=(
            "ev_to_ebitda ~ Carbon_Intensity + log_market_cap "
            "+ ebitda_margin + C(Sector)"
        ),
        data=data,
    ).fit()


def format_p_value(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else f"{value:.2e}"


def run_carbon_tax_stress_test(
    data: pd.DataFrame,
    tax_rate: int,
) -> pd.DataFrame:
    simulation = data.copy()
    simulation["Carbon_Tax_Liability"] = (
        simulation["total_scope1_emissions_mtco2e"] * tax_rate
    )
    simulation["Adjusted_EBITDA"] = (
        simulation["ebitda_usd"] - simulation["Carbon_Tax_Liability"]
    )
    simulation["Pct_EBITDA_Lost"] = (
        simulation["Carbon_Tax_Liability"] / simulation["ebitda_usd"] * 100
    ).clip(upper=100)
    profitable = simulation["Adjusted_EBITDA"] > 0
    simulation["Adjusted_EV_to_EBITDA"] = np.where(
        profitable,
        simulation["enterprise_value_usd"] / simulation["Adjusted_EBITDA"],
        np.nan,
    )
    simulation["Stress_Status"] = np.where(
        profitable,
        "Profitable",
        "Negative Earnings",
    )
    return simulation


def add_fixed_effect_lines(
    figure: go.Figure,
    data: pd.DataFrame,
    model,
) -> None:
    color_by_sector = {
        trace.name: trace.marker.color
        for trace in figure.data
        if trace.mode == "markers"
    }

    for sector, sector_data in data.groupby("Sector", observed=True):
        median_log_market_cap = sector_data["log_market_cap"].median()
        median_ebitda_margin = sector_data["ebitda_margin"].median()
        x_values = np.linspace(
            sector_data["Carbon_Intensity"].min(),
            sector_data["Carbon_Intensity"].max(),
            50,
        )
        predictions = model.predict(
            pd.DataFrame(
                {
                    "Carbon_Intensity": x_values,
                    "log_market_cap": median_log_market_cap,
                    "ebitda_margin": median_ebitda_margin,
                    "Sector": sector,
                }
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=predictions,
                mode="lines",
                name=f"{sector} fixed-effect fit",
                line={
                    "color": color_by_sector.get(sector),
                    "width": 2,
                    "dash": "dash",
                },
                hovertemplate=(
                    f"Sector: {sector}<br>"
                    "Carbon Intensity: %{x:.6f}<br>"
                    "Predicted EV/EBITDA: %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )


def main() -> None:
    st.set_page_config(
        page_title="Climate Transition Risk & Valuation Model",
        layout="wide",
    )
    st.title("Climate Transition Risk & Valuation Model")

    try:
        data = load_and_clean_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if (
        len(data) < 3
        or data["Carbon_Intensity"].nunique() < 2
        or data["Sector"].nunique() < 2
    ):
        st.error(
            "The cleaned dataset does not contain enough variation for "
            "sector fixed-effects OLS."
        )
        st.stop()

    st.sidebar.subheader("📖 About this Engine")
    st.sidebar.markdown(
        "This tool aggregates US EPA facility-level Scope 1 emissions and "
        "merges them with point-in-time financial fundamentals to quantify "
        "climate transition risk."
    )
    st.sidebar.subheader("Data Sources")
    st.sidebar.markdown(
        "- EPA GHGRP (2023)\n"
        "- Yahoo Finance"
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Model Limitations")
    st.sidebar.caption(
        "**Zero Cost Pass-Through:** Assumes 100% of the tax liability is "
        "absorbed internally. In reality, utilities often pass costs to "
        "consumers."
    )
    st.sidebar.caption(
        "**Static Operations:** This is a point-in-time snapshot. It does not "
        "forecast future green CapEx or emissions reduction initiatives."
    )
    st.sidebar.caption(
        "*For informational research only. Not investment advice.*"
    )
    st.sidebar.divider()
    st.sidebar.markdown("### 👨‍💻 Connect")
    st.sidebar.caption(
        "[LinkedIn Profile](https://linkedin.com) | "
        "[GitHub Repository](https://github.com/spencermerolla12/"
        "climate-valuation-modeler)"
    )

    model = run_ols(data)
    carbon_coefficient = model.params["Carbon_Intensity"]
    carbon_p_value = model.pvalues["Carbon_Intensity"]

    market_tab, stress_test_tab = st.tabs(
        ["📈 Market Reality", "🚨 Transition Risk Stress Test"]
    )

    with market_tab:
        col_summary, col_stats = st.columns([3, 1])

        with col_summary:
            with st.container(border=True):
                st.markdown(
                    "**The Bottom Line:** This model evaluates how equity "
                    "markets currently price carbon transition risk across "
                    "heavy industries. The data reveals a stark reality: "
                    "**Wall Street does not currently penalize companies for "
                    "their carbon footprint.** Valuations are driven almost "
                    "entirely by size and profitability. This dashboard "
                    "quantifies that market disconnect and stress-tests what "
                    "happens when regulatory carbon pricing is introduced."
                )

        with col_stats:
            st.markdown(
                f"""
**(OLS Regression Output)**
* **R-squared:** {model.rsquared:.3f}
* **P-value:** {carbon_p_value:.3f}
* **Carbon Coef:** {carbon_coefficient:,.2f}
"""
            )

        figure = px.scatter(
            data,
            x="Carbon_Intensity",
            y="ev_to_ebitda",
            color="Sector",
            hover_name="listed_company_name",
            hover_data={
                "ticker": True,
                "parent_company": True,
                "Sector": True,
                "Carbon_Intensity": ":.6f",
                "ev_to_ebitda": ":.2f",
            },
            labels={
                "Carbon_Intensity": (
                    "Carbon Intensity (Scope 1 mtCO2e / Revenue USD)"
                ),
                "ev_to_ebitda": "EV / EBITDA",
                "ticker": "Ticker",
                "parent_company": "EPA Parent Company",
            },
            template="plotly_white",
        )
        figure.update_traces(
            marker={"size": 10, "opacity": 0.8},
            selector={"mode": "markers"},
        )
        add_fixed_effect_lines(figure, data, model)
        figure.update_layout(
            hovermode="closest",
            legend_title_text="Sector",
            dragmode="pan",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        figure.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(b=100),
        )

        st.subheader(
            "Carbon Intensity vs. EV/EBITDA with Sector Fixed Effects"
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
            config={"scrollZoom": True},
        )
        st.caption(
            "Data: 2023 Fiscal Year Financials (Yahoo Finance) & 2023 Scope 1 "
            "Facility Emissions (US EPA GHGRP)."
        )

        with st.expander("🧮 Methodology: The OLS Regression Model"):
            st.markdown("### The Conceptual Model")
            st.latex(
                r"\text{Valuation} = \text{Anchor } (\beta_0) + "
                r"\text{Carbon } (\beta_1) + \text{Size } (\beta_2) + "
                r"\text{Profit } (\beta_3) + \text{Sector } (\alpha) + "
                r"\text{Noise } (\epsilon)"
            )

            st.markdown("### The Fully Expanded OLS Equation")
            st.latex(
                r"\textcolor{#2E86C1}{\frac{EV}{EBITDA}_i} = "
                r"\textcolor{#8E44AD}{\beta_0} + "
                r"\textcolor{#27AE60}{\beta_1\left(\frac{Scope 1}{Revenue}"
                r"\right)_i} + "
                r"\textcolor{#C0392B}{\beta_2(\ln(Market Cap_i)) + "
                r"\beta_3\left(\frac{EBITDA}{Revenue}\right)_i + "
                r"\alpha_{Sector}} + \epsilon_i"
            )
            st.markdown(
                """
<div style="color: #2E86C1;">
<strong>Dependent Variable (What we are predicting):</strong><br>
<strong>EV/EBITDA:</strong> The Enterprise Multiple. A capital-structure-neutral valuation metric.
</div>
<br>
<div style="color: #8E44AD;">
<strong>The Anchor (Starting Line):</strong><br>
<strong>Baseline Intercept (&beta;<sub>0</sub>):</strong> Think of this like the initial base fare on a taxi meter. It is the theoretical starting valuation before any financial, environmental, or sector-specific adjustments are applied.
</div>
<br>
<div style="color: #27AE60;">
<strong>Independent Variable of Interest (What we are testing):</strong><br>
<strong>Carbon Intensity (&beta;<sub>1</sub>):</strong> Calculated as Scope 1 Emissions / Total Revenue. We use a ratio to fairly compare the emissions of massive corporations against smaller regional players.
</div>
<br>
<div style="color: #C0392B;">
<strong>Control Variables (Removing the noise):</strong><br>
<strong>ln(Market Cap) (&beta;<sub>2</sub>):</strong> The natural log of market capitalization. This compresses the exponential size of mega-caps into a linear scale, preventing outliers from warping the baseline.<br>
<strong>EBITDA Margin (&beta;<sub>3</sub>):</strong> Calculated as EBITDA / Total Revenue. Highly profitable companies inherently trade at premium multiples. We must control for this so we don't mistakenly blame a low valuation on carbon.<br>
<strong>Sector Fixed Effects (&alpha;):</strong> Dummy variables that calculate a unique starting offset for each industry, ensuring we compare utilities strictly to utilities.<br>
<strong>Error Term (&epsilon;):</strong> The statistical noise, or the remaining variance not explained by our model.
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown(
                "By isolating the control variables, this Ordinary Least "
                "Squares (OLS) regression forces the model to measure the pure "
                "impact of a company's carbon footprint on its market "
                "valuation."
            )

        st.caption(
            f"Multivariate OLS sample: {len(data)} companies across "
            f"{data['Sector'].nunique()} sector classifications after excluding "
            f"non-positive EV or EBITDA and multiples above "
            f"{MAX_EV_TO_EBITDA}x."
        )

    with stress_test_tab:
        with st.container(border=True):
            st.markdown("### ⚙️ Stress Test Configurations")
            col_slider, col_status = st.columns([2, 1])

            with col_slider:
                tax_rate = st.slider(
                    "Hypothetical Carbon Tax ($/Metric Ton)",
                    min_value=0,
                    max_value=150,
                    value=65,
                    step=5,
                    help=(
                        "Simulate how future regulatory pricing compresses "
                        "current corporate EBITDA margins based on their raw "
                        "Scope 1 emissions."
                    ),
                )

            with col_status:
                if tax_rate == 0:
                    scenario_name = "🟩 Status Quo"
                    scenario_desc = (
                        "Current US regulatory framework. Zero direct carbon "
                        "penalty applied. Markets continue to prioritize raw "
                        "profitability."
                    )
                elif tax_rate <= 40:
                    scenario_name = "🟨 Conservative Pricing"
                    scenario_desc = (
                        "Early-stage policy intervention. Represents "
                        "localized carbon pricing or initial phase-in of a "
                        "federal tax. Expect moderate margin compression."
                    )
                elif tax_rate <= 80:
                    scenario_name = "🟧 Baseline Shock"
                    scenario_desc = (
                        "Aligns with EPA Social Cost of Carbon & IMF 2030 "
                        "targets. Simulates a severe regulatory correction "
                        "forcing immediate operational repricing."
                    )
                else:
                    scenario_name = "🟥 Aggressive Transition"
                    scenario_desc = (
                        "Stringent climate policy matching top-tier European "
                        "ETS markets. Simulates worst-case scenario EBITDA "
                        "destruction for heavy emitters."
                    )

                st.write("")
                st.markdown(f"**Scenario:** {scenario_name}")
                st.caption(scenario_desc)

        st.subheader("Transition Risk Stress Test")
        selected_sector = st.radio(
            "Navigate by Sector:",
            options=["All Sectors"] + list(data["Sector"].unique()),
            horizontal=True,
        )

        simulation = run_carbon_tax_stress_test(data, tax_rate)
        unprofitable_count = int((simulation["Adjusted_EBITDA"] <= 0).sum())

        unprofitable_col, tax_rate_col = st.columns(2)
        unprofitable_col.metric(
            "Companies with Negative Earnings",
            f"{unprofitable_count} of {len(simulation)}",
        )
        tax_rate_col.metric(
            "Carbon Tax Scenario",
            f"${tax_rate}/Metric Ton",
        )

        df_sim = simulation.rename(
            columns={
                "listed_company_name": "Company",
                "ticker": "Ticker",
                "ebitda_usd": "EBITDA",
            }
        )
        df_sim["Treemap_Label"] = (
            df_sim["Company"]
            + "<br>"
            + df_sim["Ticker"]
            + "<br>("
            + df_sim["Pct_EBITDA_Lost"].apply(lambda value: f"{value:.2f}%")
            + " Lost)"
        )
        if selected_sector == "All Sectors":
            filtered_df_sim = df_sim
        else:
            filtered_df_sim = df_sim[
                df_sim["Sector"] == selected_sector
            ].copy()

        with st.expander("Why Stress Test for Carbon Taxes?"):
            st.markdown(
                """
### The Mechanics of a Carbon Tax

A carbon tax is a regulatory fee levied directly on greenhouse gas emissions. In this model, we apply the hypothetical tax rate directly to a company's Scope 1 emissions (the physical carbon emitted directly from their facilities and smokestacks, as reported to the EPA).

### Why We Model This

Financial markets are notoriously slow to price in unprecedented regulatory shifts. By deducting this theoretical tax liability directly from a company's Baseline EBITDA (raw operational profit), we can instantly visualize which companies are operating with dangerously thin margins relative to their carbon footprint.

### Model Assumptions & Limitations

**Zero Cost Pass-Through:** This model represents a worst-case scenario. It assumes the company must absorb 100% of the carbon tax internally. In reality, utility and energy companies would likely pass a portion of this tax burden onto consumers via higher prices.

**Static Operations:** This stress test is a snapshot in time. It assumes the company does not immediately invest in green technology to lower their emissions profile in response to the tax.
"""
            )

        if tax_rate == 0:
            st.info(
                "👈 **No Tax Applied:** Adjust the Carbon Tax slider in the "
                "stress-test configuration above to simulate EBITDA "
                "compression."
            )

        stress_figure = px.treemap(
            filtered_df_sim,
            path=[
                px.Constant("Heavy Industry Universe"),
                "Sector",
                "Treemap_Label",
            ],
            values="EBITDA",
            color="Pct_EBITDA_Lost",
            color_continuous_scale="Purples",
            labels={"Pct_EBITDA_Lost": "% EBITDA Lost"},
            custom_data=[
                "Company",
                "Ticker",
                "Sector",
                "EBITDA",
                "Adjusted_EBITDA",
                "Pct_EBITDA_Lost",
            ],
        )
        stress_figure.update_traces(
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Sector: %{customdata[2]}<br>Baseline EBITDA: $%{customdata[3]:,.0f}<br>Adjusted EBITDA: $%{customdata[4]:,.0f}<br>EBITDA Lost: %{customdata[5]:.2f}%<extra></extra>"
        )
        stress_figure.update_traces(texttemplate="%{label}")
        stress_figure.update_layout(
            margin=dict(t=20, l=20, r=20, b=20),
            clickmode="none",
        )
        stress_figure.update_coloraxes(
            colorbar=dict(orientation="h", y=-0.15)
        )
        stress_figure.update_layout(margin=dict(b=80))
        st.plotly_chart(stress_figure, use_container_width=True)
        st.caption(
            "Simulation Baseline: 2023 Reported EBITDA and Carbon Output."
        )

        st.subheader("📓 Full Universe Impact Ledger")
        ledger = filtered_df_sim[
            [
                "Company",
                "Ticker",
                "Sector",
                "EBITDA",
                "Adjusted_EBITDA",
                "Pct_EBITDA_Lost",
            ]
        ].sort_values("Pct_EBITDA_Lost", ascending=False)
        ledger_display = ledger.rename(
            columns={
                "Adjusted_EBITDA": "Adjusted EBITDA",
                "Pct_EBITDA_Lost": "% EBITDA Lost",
            }
        )
        st.dataframe(
            ledger_display.style.format(
                {
                    "EBITDA": "${:,.0f}",
                    "Adjusted EBITDA": "${:,.0f}",
                    "% EBITDA Lost": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if unprofitable_count:
            failed_companies = simulation.loc[
                simulation["Stress_Status"] == "Negative Earnings",
                ["listed_company_name", "ticker"],
            ]
            company_labels = ", ".join(
                f"{row.listed_company_name} ({row.ticker})"
                for row in failed_companies.itertuples(index=False)
            )
            st.warning(
                f"Negative Earnings under this scenario: {company_labels}"
            )


if __name__ == "__main__":
    main()
