## 1. Model Governance and Oversight
- The governance structure for credit risk models involves multiple layers of management, including the **Risk Management and Compliance Management** and **Credit Risk Management** departments. These departments collaborate to develop and review the model for estimating financial asset impairment, ensuring compliance with **IFRS 9**.
- The model is subject to annual reviews or more frequent assessments if significant events occur that could impact the adopted methodology. The **Board of Directors** must approve the model and any modifications, ensuring a robust oversight mechanism.
- Specific committees are tasked with the approval of credit loss estimates for segments classified as "Individual Treatment," and the organizational structures for managing credit risk are also subject to board approval.

## 2. Data Quality and Sources
- Data inputs for the credit risk models include both **internal** and **external** sources, with a focus on historical performance data spanning a maximum analysis horizon of **60 months**. This timeframe is chosen based on the observation that no portfolios have residual maturities exceeding this limit.
- Quality control measures are implemented to ensure the accuracy and reliability of data used in the models. This includes regular audits and validations of the data sources to prevent biases in the Expected Credit Loss (ECL) estimates.
- The institution emphasizes the importance of using comprehensive and relevant data, including borrower credit ratings and macroeconomic indicators, to enhance the robustness of the credit risk assessment.

## 3. Segmentation
- The portfolio is segmented based on various criteria, including client types and product categories. Specific thresholds are established to differentiate between segments, such as **Corporate Banking** and **Retail Banking**.
- Each segment is analyzed for its unique risk characteristics, with particular attention given to the **Probability of Default (PD)** and **Loss Given Default (LGD)** metrics. This segmentation allows for tailored risk management strategies and more accurate ECL calculations.
- The segmentation process also considers external credit ratings, where clients classified as higher risk in external systems may be assigned to a higher stage in the internal classification, even if they are current on their obligations.

## 4. Definition of Default
- Default is defined as a situation where a borrower is **90 days past due** on any obligation. This threshold is critical for categorizing exposures into different stages of credit risk.
- Additional indicators of default include adverse classifications in external credit systems, where clients with a risk category of **3 or 4** are automatically classified as at least Stage 2, regardless of their payment history with the institution.
- The classification of default is maintained over time according to established guidelines, ensuring consistency in the treatment of defaulted accounts.

## 5. Risk Contagion
- The contagion of default status is governed by specific rules that dictate how default classifications can affect related products or obligations. For instance, if a client is classified as defaulted in one segment, this may trigger a similar classification across other related exposures.
- The institution employs a systematic approach to assess the interconnectedness of client relationships, ensuring that risk assessments reflect the potential for contagion across different products.
- This approach is designed to enhance the overall risk management framework by recognizing the broader implications of a default event within the portfolio.

## 6. PD (Probability of Default)
- The methodology for estimating PD involves analyzing historical default frequencies within specific segments, utilizing a maximum analysis horizon of **60 months**. This is based on the premise that longer horizons do not yield additional relevant data.
- PD is calculated separately for **12-month** and **lifetime** horizons, with a default curve constructed to reflect cumulative probabilities from the analysis date to the end of the portfolio's life. The average of the last **36 historical curves** is used to derive a stable PD estimate.
- The institution emphasizes a case-based approach for PD calculation, focusing on the number of individual exposures rather than monetary amounts, aligning with statistical principles for estimating default probabilities.

## 7. LGD (Loss Given Default)
- LGD is estimated based on recovery rates from defaulted loans, with a specific focus on the treatment of collateral. The institution applies a **100% LGD** for exposures that are **480 days past due**, ensuring full provisioning when accounts are moved to off-balance sheet status.
- The calculation of LGD involves a weighted average of recovery rates, taking into account the specific characteristics of the collateral and the historical performance of similar exposures.
- The institution does not apply LGD floors or caps but ensures that the methodology is robust enough to reflect the true loss potential in default scenarios.

## 8. EAD / CCF (Exposure at Default / Credit Conversion Factor)
- EAD is defined as the accounting balance at the time of calculation, which includes principal, accrued interest, and other debt components. This is crucial for accurately assessing potential losses in default scenarios.
- For revolving credit products, the institution follows Basel guidelines, with CCFs typically ranging from **10% to 75%** for short-term credit lines and revolving products, reflecting the likelihood of drawdowns at the time of default.
- The methodology for determining EAD and CCF is documented and regularly reviewed to ensure compliance with regulatory standards and internal risk management practices.

## 9. Macro Scenarios
- The institution utilizes multiple macroeconomic scenarios for forward-looking adjustments, including **Base**, **Adverse**, and **Optimistic** scenarios. Each scenario is assigned specific weights to reflect its likelihood and potential impact on credit risk.
- The macroeconomic variables considered include GDP growth rates, unemployment rates, and inflation, which are integrated into the ECL calculations to enhance predictive accuracy.
- The institution commits to regularly updating these scenarios based on prevailing economic conditions and forecasts, ensuring that the ECL estimates remain relevant and reflective of current market dynamics.

## 10. Forward-Looking Information
- Forward-looking information is modeled using a combination of macroeconomic indicators and borrower-specific data. This information is incorporated into the ECL calculation to adjust for expected changes in credit risk.
- The institution emphasizes the importance of using reasonable and supportable forward-looking information, ensuring that all relevant variables are considered in the risk assessment process.
- Specific macroeconomic variables, such as interest rates and economic growth forecasts, are monitored and updated regularly to reflect their impact on credit risk profiles.

## 11. ECL Calculation
- The formula for calculating Expected Credit Loss (ECL) is defined as:
  \[
  \text{PCE} = \text{EAD} \times (\text{PD} \times \text{FL}) \times \text{LGD}
  \]
  - Where:
    - **EAD**: Exposure at Default
    - **PD**: Probability of Default
    - **LGD**: Loss Given Default
    - **FL**: Forward Looking
- The ECL is calculated differently across stages:
  - **Stage 1**: \( \text{EAD} \times (\text{PD (12 months)} \times \text{FL}) \times \text{LGD} \)
  - **Stage 2**: \( \text{EAD} \times (\text{PD (Lifetime)} \times \text{FL}) \times \text{LGD} \)
  - **Stage 3**: \( \text{EAD} \times \text{LGD} \)
- This structured approach ensures that the ECL reflects the credit risk associated with each exposure accurately, considering both current and expected future conditions.

## 12. Model Monitoring
- The institution has established a comprehensive framework for model monitoring, which includes backtesting and benchmarking against historical performance metrics. This ensures that the models remain accurate and relevant over time.
- Specific performance metrics tracked include the accuracy of PD and LGD estimates, as well as the overall performance of the ECL models in predicting actual losses.
- Ongoing monitoring occurs at regular intervals, with findings reported to senior management and adjustments made as necessary to improve model performance.

## 13. Model Overrides
- The policy for model overrides is strictly governed, requiring expert judgment to be documented and approved at appropriate governance levels. Overrides must be justified with clear rationale and supported by empirical data.
- The institution mandates that any post-model adjustments undergo a formal review process, ensuring that they align with the overall risk management framework and regulatory requirements.
- Documentation standards for overrides are rigorous, requiring detailed records of the decision-making process and the data used to support any deviations from the model outputs.