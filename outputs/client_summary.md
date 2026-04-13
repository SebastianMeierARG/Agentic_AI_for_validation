## 1. Model Governance and Oversight
- The governance structure for credit risk management includes the **Risk and Compliance Management** department, which collaborates with the **Credit Risk Management** department to develop and review the financial asset impairment estimation model.
- The model must align with **IFRS 9** standards and is subject to annual reviews or more frequent assessments if significant changes occur.
- Approval processes involve submitting the model and any modifications to the **Board of Directors** for final approval, ensuring that all parameters used in the model are also sanctioned by the Board.

## 2. Data Quality and Sources
- Data inputs for the credit risk models include both **internal** and **external** sources, with a focus on historical performance data spanning a maximum of **60 months**.
- Quality control measures are implemented to ensure the accuracy and reliability of data, including regular audits and validation checks against external benchmarks.
- The methodology emphasizes the importance of using individual exposure cases rather than monetary amounts to enhance the statistical validity of the Probability of Default (PD) estimations.

## 3. Segmentation
- The portfolio is segmented based on risk categories, with specific thresholds established for different client types and products.
- Clients classified as **Stage 2** may include those with a risk category higher than the internal classification, even if their local obligations are current, to incorporate external deterioration signals.
- The segmentation process is designed to improve the consolidated view of risk and is updated to reflect changes in client risk profiles.

## 4. Definition of Default
- Default is defined as an obligation that is **90 days past due** or classified as unlikely to pay, which includes adverse classifications in external credit bureaus.
- Clients in the **Corporate Banking** segment with a classification of **3 or 4** in the financial system are automatically classified as Stage 2, regardless of their payment history with the entity.
- The default status is maintained over time according to established guidelines in the **Cure Operations Process**.

## 5. Risk Contagion
- The contagion rules stipulate that if a client is classified as default in one product, it may affect the default status across other related products or obligations.
- This approach ensures that interconnected risks are adequately captured, allowing for a more comprehensive risk assessment.
- Specific criteria for contagion include adverse classifications in external credit systems and internal risk assessments that exceed established thresholds.

## 6. PD (Probability of Default)
- The methodology for estimating PD is based on historical default frequencies observed in specific segments, with a maximum analysis horizon of **60 months**.
- PD is calculated separately for **12-month** and **lifetime** periods, using a cumulative default curve derived from historical data.
- The average of the last **36 historical curves** is computed to derive a stable PD estimate, ensuring that the model reflects recent trends and conditions.

## 7. LGD (Loss Given Default)
- LGD is estimated using weighted averages based on the balance of defaults, with a specific formula applied to calculate recovery rates.
- A **100% LGD** is enforced for exposures that are **480 days past due** to ensure full provisioning when accounts are moved to off-balance sheet status.
- The treatment of collateral is factored into the LGD calculations, with specific floors or caps applied to ensure conservative estimates.

## 8. EAD / CCF (Exposure at Default / Credit Conversion Factor)
- EAD is determined as the accounting balance at the calculation date, which includes capital, accrued interest, and other debt components.
- For revolving operations, the **Credit Conversion Factor (CCF)** is guided by Basel standards, typically ranging from **10% to 75%** for short-term credit lines and revolving credit products.
- The methodology ensures that both on-balance and off-balance sheet exposures are accurately captured in the EAD calculations.

## 9. Macro Scenarios
- The macroeconomic scenarios utilized for forward-looking adjustments include **Base**, **Adverse**, and **Optimistic** scenarios, although specific weights for each scenario are not documented.
- These scenarios are critical for adjusting PD and LGD estimates to reflect potential future economic conditions.
- The scenarios are reviewed periodically to ensure they remain relevant and reflective of current economic forecasts.

## 10. Forward-Looking Information
- Forward-looking information is modeled by incorporating macroeconomic variables that influence credit risk, although specific variables used are not documented.
- The methodology includes adjustments to PD and LGD based on anticipated economic conditions, ensuring that the Expected Credit Loss (ECL) calculations are proactive rather than reactive.
- The integration of forward-looking information is essential for compliance with IFRS 9 requirements.

## 11. ECL Calculation
- The formula for Expected Credit Loss (ECL) is defined as:
  \[
  \text{ECL} = \text{EAD} \times (\text{PD} \times \text{FL}) \times \text{LGD}
  \]
- The ECL is calculated differently across stages:
  - **Stage 1**: \( \text{ECL} = \text{EAD} \times (\text{PD (12 months)} \times \text{FL}) \times \text{LGD} \)
  - **Stage 2**: \( \text{ECL} = \text{EAD} \times (\text{PD (Lifetime)} \times \text{FL}) \times \text{LGD} \)
  - **Stage 3**: \( \text{ECL} = \text{EAD} \times \text{LGD} \)
- This structured approach ensures that the ECL reflects the risk profile of the asset at each stage of credit deterioration.

## 12. Model Monitoring
- The framework for model monitoring includes backtesting and benchmarking against actual performance metrics, with specific performance indicators tracked regularly.
- Ongoing monitoring is conducted at least annually, or more frequently if significant changes in the portfolio or economic conditions occur.
- The results of monitoring activities are reported to the Board, ensuring transparency and accountability in the risk management process.

## 13. Model Overrides
- The policy for model overrides requires strict governance, with expert judgment adjustments needing approval from senior management and documentation of the rationale behind such decisions.
- Overrides must be justified based on significant changes in market conditions or client circumstances that are not captured by the model.
- Documentation standards are enforced to ensure that all overrides are recorded and reviewed, maintaining the integrity of the risk management framework.