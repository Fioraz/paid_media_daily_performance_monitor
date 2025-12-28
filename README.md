# Daily Paid Media Performance Monitor (Ad Level)

A lightweight Python monitor that analyzes **daily paid media performance at the ad level** for an e-commerce brand selling **high-ticket greenhouses** (AOV ≈ **$3,000**). It classifies each ad as **Prospecting** or **Retargeting**, evaluates performance against funnel-stage benchmarks, flags **alerts** (bad/great), and generates a **Slack/email-ready** text report.

## What it does

From a dataset with **one row per ad per day**, the monitor:

1. **Calculates derived metrics** (safe division; missing/zero => `N/A`)
   - `cost_per_atc = spend / atc` (if `atc > 0`)
   - `cost_per_ic = spend / ic` (if `ic > 0`)
   - `cpa = spend / orders` (if `orders > 0`)
   - `aov = revenue / orders` (if `orders > 0`)
   - `atc_rate` only if `landing_page_views` is available; otherwise skipped (no alerts are raised on it)

2. **Classifies each ad** as:
   - **Prospecting** or **Retargeting**
   - Naming patterns are checked first (e.g., `TOF`, `Prospecting`, `Cold`, `P_`, `Retargeting`, `Remarketing`, `MOF`, `BOF`, `R_`)
   - If naming is unclear, uses `frequency_7d` heuristic:
     - `frequency_7d <= 2.5` → **Prospecting**
     - `frequency_7d > 2.5` → **Retargeting**

3. **Evaluates performance using stage-specific benchmarks**
   - Bounce rate, Avg session duration, Cost/ATC, Cost/IC, CPA, AOV
   - If a metric cannot be calculated, it is marked `N/A` and does not trigger alerts

4. **Triggers alerts** (only for meaningful volume)
   - Meaningful volume: **$50+ spend OR ≥ 3 events** (ATC/IC/Orders)
   - **Negative alerts** if metric is in *Bad* range (with additional event minimums for cost metrics / CPA)
   - **Positive alerts** if metric is in *Good* range (with additional minimum volume)

5. **Generates a concise report** suitable for Slack/email:
   - Headline summary
   - Prospecting summary sentence
   - Retargeting summary sentence
   - Key Alerts (Bad)
   - Key Wins (Great)
   - Funnel behavior commentary (3–5 bullets)
   - Optional compact table for ads that triggered alerts
