"""
Daily Paid Media Performance Monitor
=====================================
A production-ready system for monitoring high-ticket greenhouse ad performance.
Designed for integration with Slack, email, or other notification systems.

Implements the task specification:
- Classify ads as Prospecting vs Retargeting (naming first, then frequency_7d heuristic)
- Calculate required derived metrics (safe division; missing/zero => N/A)
- Evaluate metrics against Prospecting vs Retargeting benchmarks
- Trigger negative/positive alerts per specified rules (no alerts on N/A)
- Output a concise Slack/email-friendly daily summary with alerts, wins, commentary,
  and an optional compact table (only ads that triggered alerts)

Notes (per task):
- atc_rate is calculated ONLY if landing_page_views is available. If not, it's skipped.
- If fields are missing, report explicitly what’s missing and proceed with what’s available.

Author: Dinusha Dissanayaka
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import re
from enum import Enum
from sample_data import generate_sample_data

# ------------------------------------------------------- #
# CONFIGURATION - Easily editable benchmarks and settings #
# ------------------------------------------------------- #

@dataclass
class MetricBenchmark:
    """
    Defines Good/OK/Bad thresholds for a metric.
    - If higher_is_better is False: lower values are better (e.g., bounce, costs, CPA).
      Good: value <= good_threshold
      OK:   good_threshold < value <= ok_threshold
      Bad:  value > ok_threshold
    - If higher_is_better is True: higher values are better (e.g., AOV, session duration).
      Good: value >= good_threshold
      OK:   ok_threshold <= value < good_threshold
      Bad:  value < ok_threshold
    """
    good_threshold: float
    ok_threshold: float
    higher_is_better: bool = False  # True only for metrics like AOV, session duration, & atc_rate
    
    def evaluate(self, value: Optional[float]) -> str:
        """Evaluate metric value. Returns Good/OK/Bad/N/A."""
        if value is None or pd.isna(value):
            return "N/A"
        
        if self.higher_is_better:
            if value >= self.good_threshold:
                return "Good"
            elif value >= self.ok_threshold:
                return "OK"
            else:
                return "Bad"
        else:
            if value <= self.good_threshold:
                return "Good"
            elif value <= self.ok_threshold:
                return "OK"
            else:
                return "Bad"


@dataclass(frozen=True)
class FunnelStageBenchmarks:
    """Configuration for a funnel stage (Prospecting or Retargeting)."""
    bounce_rate: MetricBenchmark
    avg_session_duration_seconds: MetricBenchmark
    cost_per_atc: MetricBenchmark
    cost_per_ic: MetricBenchmark
    cpa: MetricBenchmark
    aov: MetricBenchmark


@dataclass(frozen=True)
class AlertThresholds:
    """Minimum volume thresholds to trigger alerts."""
    # "meaningful volume (at least $50+ daily spend or ≥ 3–5 events)"
    min_spend: float = 50.0
    min_events_alt: int = 3  # used only to allow non-spend volume gating for bounce/session

    # Negative alerts
    min_atc_for_bad_cost_per_atc: int = 5
    min_ic_for_bad_cost_per_ic: int = 3
    min_orders_for_bad_cpa: int = 1

    # Positive alerts
    min_atc_for_good_cost_per_atc: int = 10
    min_ic_for_good_cost_per_ic: int = 5
    min_orders_for_good_cpa: int = 2
    min_orders_for_good_aov: int = 2



class Config:
    """Central configuration for the monitor."""  
    # Prospecting benchmarks (Cold)
    PROSPECTING = FunnelStageBenchmarks(
        bounce_rate=MetricBenchmark(good_threshold=65, ok_threshold=75, higher_is_better=False),
        avg_session_duration_seconds=MetricBenchmark(good_threshold=105, ok_threshold=75, higher_is_better=True),
        cost_per_atc=MetricBenchmark(good_threshold=120, ok_threshold=160, higher_is_better=False),
        cost_per_ic=MetricBenchmark(good_threshold=350, ok_threshold=450, higher_is_better=False),
        cpa=MetricBenchmark(good_threshold=1500, ok_threshold=2000, higher_is_better=False),
        aov=MetricBenchmark(good_threshold=2800, ok_threshold=2400, higher_is_better=True),
    )
    
    # Retargeting benchmarks (Warm/Hot)
    RETARGETING = FunnelStageBenchmarks(
        bounce_rate=MetricBenchmark(good_threshold=55, ok_threshold=65, higher_is_better=False),
        avg_session_duration_seconds=MetricBenchmark(good_threshold=120, ok_threshold=90, higher_is_better=True),
        cost_per_atc=MetricBenchmark(good_threshold=80, ok_threshold=120, higher_is_better=False),
        cost_per_ic=MetricBenchmark(good_threshold=200, ok_threshold=300, higher_is_better=False),
        cpa=MetricBenchmark(good_threshold=800, ok_threshold=1200, higher_is_better=False),
        aov=MetricBenchmark(good_threshold=2800, ok_threshold=2400, higher_is_better=True),
    )
    
    ALERTS = AlertThresholds()
    
    # Classification patterns (naming first)
    PROSPECTING_PATTERNS = [
        r"\bTOF\b",
        r"\bProspecting\b",
        r"\bCold\b",
        r"\bP_",
        r"\bP\s*-",
    ]
    RETARGETING_PATTERNS = [
        r"\bRetargeting\b",
        r"\bRemarketing\b",
        r"\bMOF\b",
        r"\bBOF\b",
        r"\bR_",
    ]

    FREQUENCY_THRESHOLD = 2.5  # <= 2.5 Prospecting, > 2.5 Retargeting

    # Minimum input fields per task
    MIN_INPUT_FIELDS = [
        "date",
        "ad_name",
        "campaign_name",
        "frequency_7d",
        "landing_page_url",
        "bounce_rate",
        "avg_session_duration_seconds",
        "spend",
        "atc",
        "ic",
        "orders",
        "revenue",
    ]


# ------------------------------------------------------- #
# DATA MODELS                                             #
# ------------------------------------------------------- #

class FunnelStage(Enum):
    """
    Funnel stage classification (Always output the final classification for each ad as one of:
    Prospecting / Retargeting)
    """
    PROSPECTING = "Prospecting"
    RETARGETING = "Retargeting"


def safe_div(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None or pd.isna(numer) or pd.isna(denom) or denom <= 0:
        return None
    return float(numer) / float(denom)

@dataclass
class AdRow:
    # Raw fields (may be missing)
    date: Optional[str]
    ad_name: Optional[str]
    campaign_name: Optional[str]
    frequency_7d: Optional[float]
    landing_page_url: Optional[str]
    bounce_rate: Optional[float]
    avg_session_duration_seconds: Optional[float]
    spend: Optional[float]
    atc: Optional[int]
    ic: Optional[int]
    orders: Optional[int]
    revenue: Optional[float]

    # Optional field for atc_rate calculation (not required in minimum fields)
    landing_page_views: Optional[int] = None

    # Classification
    stage: FunnelStage = FunnelStage.PROSPECTING
    classification_method: str = "default"

    # Derived metrics
    cost_per_atc: Optional[float] = None
    cost_per_ic: Optional[float] = None
    cpa: Optional[float] = None
    aov: Optional[float] = None
    atc_rate: Optional[float] = None  # computed only if landing_page_views exists

    # Evaluations
    evals: Dict[str, str] = field(default_factory=dict)

    def compute_metrics(self) -> None:
        """Calculate derived metrics with safe division."""
        self.cost_per_atc = safe_div(self.spend, self.atc)
        self.cost_per_ic = safe_div(self.spend, self.ic)
        self.cpa = safe_div(self.spend, self.orders)
        self.aov = safe_div(self.revenue, self.orders)

        # atc_rate only if landing_page_views available
        if self.landing_page_views is not None and self.landing_page_views > 0 and self.atc is not None:
            self.atc_rate = (float(self.atc) / float(self.landing_page_views)) * 100.0
        else:
            self.atc_rate = None

    def evaluate(self) -> None:
        """Evaluate each metric against benchmarks based on funnel stage."""
        bm = Config.PROSPECTING if self.stage == FunnelStage.PROSPECTING else Config.RETARGETING
        self.evals = {
            "bounce_rate": bm.bounce_rate.evaluate(self.bounce_rate),
            "avg_session_duration_seconds": bm.avg_session_duration_seconds.evaluate(self.avg_session_duration_seconds),
            "cost_per_atc": bm.cost_per_atc.evaluate(self.cost_per_atc),
            "cost_per_ic": bm.cost_per_ic.evaluate(self.cost_per_ic),
            "cpa": bm.cpa.evaluate(self.cpa),
            "aov": bm.aov.evaluate(self.aov),
            # atc_rate is computed if possible, but no benchmarks/alerts are defined in the task.
        }

    def display_name(self, max_len: int = 42) -> str:
        """Get ad name for display, with fallback."""
        name = self.ad_name or "Unknown Ad"
        return name if len(name) <= max_len else name[: max_len - 3] + "..."

    def spend_value(self) -> float:
        """Get spend value, defaulting to 0 if missing."""
        return float(self.spend) if self.spend is not None and not pd.isna(self.spend) else 0.0

    def atc_value(self) -> int:
        """Get ATC value, defaulting to 0 if missing."""
        return int(self.atc) if self.atc is not None and not pd.isna(self.atc) else 0

    def ic_value(self) -> int:
        """Get IC value, defaulting to 0 if missing."""
        return int(self.ic) if self.ic is not None and not pd.isna(self.ic) else 0

    def orders_value(self) -> int:
        """Get orders value, defaulting to 0 if missing."""
        return int(self.orders) if self.orders is not None and not pd.isna(self.orders) else 0

    def revenue_value(self) -> float:
        """Get revenue value, defaulting to 0 if missing."""
        return float(self.revenue) if self.revenue is not None and not pd.isna(self.revenue) else 0.0


@dataclass(frozen=True)
class Alert:
    stage: FunnelStage
    ad_name: str
    metric: str
    actual: Optional[float]
    target_text: str
    suggestion: str
    is_positive: bool


# ------------------------------------------------------- #
# Classification                                          #
# ------------------------------------------------------- #

class AdClassifier:
    """
    Classifies ads into Prospecting or Retargeting. 
    1. If the campaign or ad set naming clearly indicates funnel stage, use that first.
    2. If naming is NOT clear, use frequency as heuristic:
       - If frequency_7d <= 2.5 -> classify as PROSPECTING
       - If frequency_7d > 2.5 -> classify as RETARGETING
    """
    @staticmethod
    def classify(ad_name: Optional[str], campaign_name: Optional[str], frequency_7d: Optional[float]) -> Tuple[FunnelStage, str]:
        combined = f"{campaign_name or ''} {ad_name or ''}".strip()

        for pat in Config.PROSPECTING_PATTERNS:
            if re.search(pat, combined, flags=re.IGNORECASE):
                return FunnelStage.PROSPECTING, "naming"

        for pat in Config.RETARGETING_PATTERNS:
            if re.search(pat, combined, flags=re.IGNORECASE):
                return FunnelStage.RETARGETING, "naming"

        if frequency_7d is not None and not pd.isna(frequency_7d):
            return (FunnelStage.PROSPECTING, "frequency") if frequency_7d <= Config.FREQUENCY_THRESHOLD else (FunnelStage.RETARGETING, "frequency")

        return FunnelStage.PROSPECTING, "default"


# ------------------------------------------------------- #
# Alerts                                                  #
# ------------------------------------------------------- #

class AlertGenerator:
    """
    Generates alerts.
    
    NEGATIVE ALERT (performance bad) triggers if:
    - Bounce rate in "Bad" range
    - Avg session duration in "Bad" range
    - Cost per ATC in "Bad" range (and at least 5 ATC events)
    - Cost per IC in "Bad" range (and at least 3 IC events)
    - CPA in "Bad" range (and at least 1 order)
    
    POSITIVE ALERT (performance great) triggers if:
    - Cost per ATC in "Good" range with ATC >= 10
    - Cost per IC in "Good" range with IC >= 5
    - CPA in "Good" range with orders >= 2
    - AOV in "Good" range with orders >= 2
    
    Focus on: Big deviations, ads with meaningful volume ($50+ daily spend or 3-5 events)
    """

    def __init__(self, thresholds: AlertThresholds = Config.ALERTS):
        self.t = thresholds

    def _meaningful_volume(self, ad: AdRow) -> bool:
        """
        Task: "meaningful volume (at least $50+ daily spend or ≥ 3–5 events)"
        Implemented as:
        - spend >= $50 OR
        - at least 3 events among (ATC, IC, Orders)
        """
        if ad.spend is not None and not pd.isna(ad.spend) and ad.spend >= self.t.min_spend:
            return True
        if max(ad.atc_value(), ad.ic_value(), ad.orders_value()) >= self.t.min_events_alt:
            return True
        return False

    def generate(self, ad: AdRow) -> List[Alert]:
        """Generate alerts for an ad."""
        if not self._meaningful_volume(ad):
            return []

        bm = Config.PROSPECTING if ad.stage == FunnelStage.PROSPECTING else Config.RETARGETING
        alerts: List[Alert] = []

        # ----- NEGATIVE ALERTS -----
        # Bounce rate in “Bad” range
        if ad.evals.get("bounce_rate") == "Bad":
            target = f"Target {ad.stage.value} ≤ {bm.bounce_rate.good_threshold:.0f}%"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="Bounce rate",
                actual=ad.bounce_rate,
                target_text=target,
                suggestion="Check landing page relevance vs ad promise; tighten message match.",
                is_positive=False,
            ))

        # Avg session duration in “Bad” range
        if ad.evals.get("avg_session_duration_seconds") == "Bad":
            target = f"Target {ad.stage.value} ≥ {bm.avg_session_duration_seconds.good_threshold:.0f}s"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="Avg session duration",
                actual=ad.avg_session_duration_seconds,
                target_text=target,
                suggestion="Review page load speed and above-the-fold clarity; improve engagement.",
                is_positive=False,
            ))

        # Cost per ATC in “Bad” range (and at least 5 ATC events)
        if ad.evals.get("cost_per_atc") == "Bad" and ad.atc_value() >= self.t.min_atc_for_bad_cost_per_atc:
            target = f"Target {ad.stage.value} ≤ ${bm.cost_per_atc.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="Cost per ATC",
                actual=ad.cost_per_atc,
                target_text=target,
                suggestion="Test a new hook/angle; the creative is not driving strong intent efficiently.",
                is_positive=False,
            ))

        # Cost per IC in “Bad” range (and at least 3 IC events)
        if ad.evals.get("cost_per_ic") == "Bad" and ad.ic_value() >= self.t.min_ic_for_bad_cost_per_ic:
            target = f"Target {ad.stage.value} ≤ ${bm.cost_per_ic.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="Cost per IC",
                actual=ad.cost_per_ic,
                target_text=target,
                suggestion="Look for checkout friction; consider simplifying the path to checkout.",
                is_positive=False,
            ))

        # CPA in “Bad” range (and at least 1 order)
        if ad.evals.get("cpa") == "Bad" and ad.orders_value() >= self.t.min_orders_for_bad_cpa:
            target = f"Target {ad.stage.value} ≤ ${bm.cpa.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="CPA",
                actual=ad.cpa,
                target_text=target,
                suggestion="Investigate audience quality and on-site conversion blockers.",
                is_positive=False,
            ))

        # ----- POSITIVE ALERTS -----
        # Cost per ATC in “Good” range with ATC ≥ 10
        if ad.evals.get("cost_per_atc") == "Good" and ad.atc_value() >= self.t.min_atc_for_good_cost_per_atc:
            target = f"Target {ad.stage.value} ≤ ${bm.cost_per_atc.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="Cost per ATC",
                actual=ad.cost_per_atc,
                target_text=target,
                suggestion="Scale budget (incrementally) and consider cloning to new audiences.",
                is_positive=True,
            ))

        # Cost per IC in “Good” range with IC ≥ 5
        if ad.evals.get("cost_per_ic") == "Good" and ad.ic_value() >= self.t.min_ic_for_good_cost_per_ic:
            target = f"Target {ad.stage.value} ≤ ${bm.cost_per_ic.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="Cost per IC",
                actual=ad.cost_per_ic,
                target_text=target,
                suggestion="Clone and iterate on this concept; keep the core angle.",
                is_positive=True,
            ))

        # CPA in “Good” range with orders ≥ 2
        if ad.evals.get("cpa") == "Good" and ad.orders_value() >= self.t.min_orders_for_good_cpa:
            target = f"Target {ad.stage.value} ≤ ${bm.cpa.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="CPA",
                actual=ad.cpa,
                target_text=target,
                suggestion="Scale budget (incrementally) while watching CPA stability.",
                is_positive=True,
            ))

        # AOV in “Good” range with orders ≥ 2
        if ad.evals.get("aov") == "Good" and ad.orders_value() >= self.t.min_orders_for_good_aov:
            target = f"Target {ad.stage.value} ≥ ${bm.aov.good_threshold:,.0f}"
            alerts.append(Alert(
                stage=ad.stage,
                ad_name=ad.display_name(),
                metric="AOV",
                actual=ad.aov,
                target_text=target,
                suggestion="Protect this traffic quality; keep offer consistency and consider scaling.",
                is_positive=True,
            ))

        return alerts


# ------------------------------------------------------- #
# Monitor + Reporting                                     #
# ------------------------------------------------------- #

@dataclass
class DataQualityReport:
    """Tracks missing and available fields in the input data."""
    total_rows: int
    missing_fields: List[str]
    available_fields: List[str]
    atc_rate_skipped: bool

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        if self.missing_fields:
            lines.append(f"Missing fields: {', '.join(self.missing_fields)}")
        if self.atc_rate_skipped:
            lines.append("landing_page_views not provided; ATC rate was skipped.")
        return lines


class PaidMediaMonitor:
    def __init__(self):
        self.classifier = AdClassifier()
        self.alerts = AlertGenerator()

    @staticmethod
    def _safe_get_str(row: pd.Series, col: str) -> Optional[str]:
        if col not in row.index:
            return None
        v = row[col]
        if pd.isna(v):
            return None
        return str(v)

    @staticmethod
    def _safe_get_float(row: pd.Series, col: str) -> Optional[float]:
        if col not in row.index:
            return None
        v = row[col]
        if pd.isna(v):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_get_int(row: pd.Series, col: str) -> Optional[int]:
        if col not in row.index:
            return None
        v = row[col]
        if pd.isna(v):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def analyze_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        available = [c for c in Config.MIN_INPUT_FIELDS if c in df.columns]
        missing = [c for c in Config.MIN_INPUT_FIELDS if c not in df.columns]

        atc_rate_skipped = "landing_page_views" not in df.columns
        return DataQualityReport(
            total_rows=len(df),
            missing_fields=missing,
            available_fields=available,
            atc_rate_skipped=atc_rate_skipped,
        )

    def process(self, df: pd.DataFrame) -> Tuple[List[AdRow], DataQualityReport]:
        dq = self.analyze_data_quality(df)
        ads: List[AdRow] = []

        if df is None or len(df) == 0:
            return ads, dq

        for _, row in df.iterrows():
            ad_name = self._safe_get_str(row, "ad_name")
            campaign_name = self._safe_get_str(row, "campaign_name")
            freq = self._safe_get_float(row, "frequency_7d")

            stage, method = self.classifier.classify(ad_name, campaign_name, freq)

            ad = AdRow(
                date=self._safe_get_str(row, "date"),
                ad_name=ad_name,
                campaign_name=campaign_name,
                frequency_7d=freq,
                landing_page_url=self._safe_get_str(row, "landing_page_url"),
                bounce_rate=self._safe_get_float(row, "bounce_rate"),
                avg_session_duration_seconds=self._safe_get_float(row, "avg_session_duration_seconds"),
                spend=self._safe_get_float(row, "spend"),
                atc=self._safe_get_int(row, "atc"),
                ic=self._safe_get_int(row, "ic"),
                orders=self._safe_get_int(row, "orders"),
                revenue=self._safe_get_float(row, "revenue"),
                landing_page_views=self._safe_get_int(row, "landing_page_views"),
                stage=stage,
                classification_method=method,
            )

            ad.compute_metrics()
            ad.evaluate()
            ads.append(ad)

        return ads, dq

    @staticmethod
    def _fmt_money(v: Optional[float]) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"${v:,.0f}"

    @staticmethod
    def _fmt_pct(v: Optional[float]) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{v:.1f}%"

    @staticmethod
    def _fmt_seconds(v: Optional[float]) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{v:.0f}s"

    def generate_report(self, ads: List[AdRow], data_quality: DataQualityReport, report_date: Optional[str] = None) -> str:
        if report_date is None:
            # best-effort: use first row's date, else today
            report_date = next((a.date for a in ads if a.date), None) or datetime.now().strftime("%Y-%m-%d")

        # Split stages
        prospecting = [a for a in ads if a.stage == FunnelStage.PROSPECTING]
        retargeting = [a for a in ads if a.stage == FunnelStage.RETARGETING]

        # Alerts
        all_alerts: List[Alert] = []
        for ad in ads:
            all_alerts.extend(self.alerts.generate(ad))

        negative = [x for x in all_alerts if not x.is_positive]
        positive = [x for x in all_alerts if x.is_positive]

        # Day strength (no extra metric rules; derived from alert balance)
        if positive and not negative:
            day_quality = "strong"
        elif negative and not positive:
            day_quality = "weak"
        elif positive and negative:
            day_quality = "average"
        else:
            day_quality = "average"

        lines: List[str] = []

        # ---- Data availability ---- #
        if data_quality.missing_fields or data_quality.atc_rate_skipped:
            for l in data_quality.summary_lines():
                lines.append(f"* Data notes: {l}")
            lines.append("")

        # ---- Headline summary (2–3 sentences) ---- #
        lines.append(f"===================================================== \nDAILY PAID MEDIA PERFORMANCE: {report_date} - {day_quality.upper()} DAY ({len(positive)} WINS, {len(negative)} UNDERPERFORMERS)")

        # Prospecting sentence (intent + on-site behavior first; not purely CPA)
        p_spend = sum(a.spend_value() for a in prospecting)
        p_atc = sum(a.atc_value() for a in prospecting)
        p_ic = sum(a.ic_value() for a in prospecting)
        p_bounce_vals = [a.bounce_rate for a in prospecting if a.bounce_rate is not None and not pd.isna(a.bounce_rate)]
        p_sess_vals = [a.avg_session_duration_seconds for a in prospecting if a.avg_session_duration_seconds is not None and not pd.isna(a.avg_session_duration_seconds)]
        p_cpatc = safe_div(p_spend, p_atc) if p_atc > 0 else None
        p_cpic = safe_div(p_spend, p_ic) if p_ic > 0 else None
        p_avg_bounce = float(np.mean(p_bounce_vals)) if p_bounce_vals else None
        p_avg_sess = float(np.mean(p_sess_vals)) if p_sess_vals else None

        lines.append(
            "PROSPECTING: "
            f"{self._fmt_money(p_spend)} spend, {p_atc} ATCs ({self._fmt_money(p_cpatc)}/ATC), "
            f"{p_ic} ICs ({self._fmt_money(p_cpic)}/IC), "
            f"avg bounce {self._fmt_pct(p_avg_bounce)}, avg session {self._fmt_seconds(p_avg_sess)}."
        )

        # Retargeting sentence (expected to close; CPA/AOV focus)
        r_spend = sum(a.spend_value() for a in retargeting)
        r_orders = sum(a.orders_value() for a in retargeting)
        r_revenue = sum(a.revenue_value() for a in retargeting)
        r_cpa = safe_div(r_spend, r_orders) if r_orders > 0 else None
        r_aov = safe_div(r_revenue, r_orders) if r_orders > 0 else None

        lines.append(
            "RETARGETING: "
            f"{self._fmt_money(r_spend)} spend, {r_orders} orders ({self._fmt_money(r_cpa)} CPA), "
            f"AOV {self._fmt_money(r_aov)}. \n===================================================== \n"
        )
        lines.append("")

        # ---- Classification output for each ad (required) ---- #
        # Keep it compact: one bullet per ad.
        lines.append("Ad classifications \n----------------------------------- \n")
        for ad in ads:
            lines.append(f"- {ad.display_name(60)} — {ad.stage.value}")
        lines.append("")

        # ---- Key Alerts – Underperforming (Bad) ---- #
        lines.append("Key Alerts – Underperforming (Bad) \n----------------------------------- \n")
        if not negative:
            lines.append("- None.")
        else:
            # Group by stage
            for stage in [FunnelStage.PROSPECTING, FunnelStage.RETARGETING]:
                stage_alerts = [a for a in negative if a.stage == stage]
                if not stage_alerts:
                    continue
                lines.append(f"{(stage.value).upper()}:")
                # limit to most important problems (cap 5 per stage)
                for a in stage_alerts[:5]:
                    actual_str = (
                        self._fmt_pct(a.actual) if "bounce" in a.metric.lower()
                        else self._fmt_seconds(a.actual) if "session" in a.metric.lower()
                        else self._fmt_money(a.actual)
                    )
                    lines.append(f"  * {stage.value} - {a.ad_name} \n {a.metric} = {actual_str} ({a.target_text}) \n Suggestion: {a.suggestion} \n")
        lines.append("")

        # ---- Key Wins – Overperforming (Great) ---- #
        lines.append("Key Wins – Overperforming (Great) \n----------------------------------- \n")
        if not positive:
            lines.append("- None.")
        else:
            # top 3–5
            for a in positive[:5]:
                actual_str = self._fmt_money(a.actual)
                lines.append(f"* {a.stage.value} — {a.ad_name} \n {a.metric} = {actual_str} ({a.target_text}) \n Suggestion: {a.suggestion} \n")
        lines.append("")

        # ---- Funnel Behavior Commentary (3–5 bullets) ---- #
        lines.append("Funnel Behavior Commentary (Short) \n----------------------------------- \n")
        commentary = self._funnel_commentary(prospecting, retargeting)
        for c in commentary[:5]:
            lines.append(f"- {c}")
        lines.append("")

        # ---- Optional Table Summary (only ads that triggered alerts) ---- #
        lines.append("Optional Table Summary (Alerts Only) \n----------------------------------- \n")
        table = self._alert_table(ads, all_alerts, max_rows=15)
        lines.append(table if table else "No ads triggered alerts.")
        lines.append("")

        return "\n".join(lines)

    def _funnel_commentary(self, prospecting: List[AdRow], retargeting: List[AdRow]) -> List[str]:
        points: List[str] = []

        # Prospecting: ATC/IC + bounce/session interpretation
        if prospecting:
            p_atc = sum(a.atc_value() for a in prospecting)
            p_ic = sum(a.ic_value() for a in prospecting)

            p_bounce = [a.bounce_rate for a in prospecting if a.bounce_rate is not None and not pd.isna(a.bounce_rate)]
            p_sess = [a.avg_session_duration_seconds for a in prospecting if a.avg_session_duration_seconds is not None and not pd.isna(a.avg_session_duration_seconds)]

            if p_bounce:
                avg_b = float(np.mean(p_bounce))
                points.append(f"Prospecting on-site quality: avg bounce {self._fmt_pct(avg_b)} (target good ≤ 65%).")

            if p_sess:
                avg_s = float(np.mean(p_sess))
                points.append(f"Prospecting engagement: avg session {self._fmt_seconds(avg_s)} (target good ≥ 105s).")

            if p_atc > 0:
                atc_to_ic = (p_ic / p_atc) * 100.0
                points.append(f"Prospecting intent flow: {p_atc} ATCs → {p_ic} ICs ({atc_to_ic:.1f}% ATC→IC).")

        # Retargeting: CPA/AOV closing ability
        if retargeting:
            r_spend = sum(a.spend_value() for a in retargeting)
            r_orders = sum(a.orders_value() for a in retargeting)
            r_revenue = sum(a.revenue_value() for a in retargeting)

            if r_orders > 0:
                r_cpa = r_spend / r_orders
                r_aov = r_revenue / r_orders if r_revenue > 0 else None
                points.append(f"Retargeting closing: {r_orders} orders at {self._fmt_money(r_cpa)} CPA (target good ≤ $800).")
                points.append(f"Retargeting value: AOV {self._fmt_money(r_aov)} (target good ≥ $2,800).")
            else:
                if r_spend > 0:
                    points.append("Retargeting spent today but did not generate orders (expected to close).")

        # Mismatch (simple, task-aligned interpretation)
        if prospecting and retargeting:
            p_atc = sum(a.atc_value() for a in prospecting)
            r_orders = sum(a.orders_value() for a in retargeting)
            if p_atc > 0 and r_orders == 0:
                points.append("Mismatch: Prospecting is generating intent (ATCs), but retargeting is not closing today.")

        if not points:
            points.append("Insufficient data to interpret funnel behavior reliably.")

        return points

    def _alert_table(self, ads: List[AdRow], alerts: List[Alert], max_rows: int = 15) -> str:
        if not alerts:
            return ""

        # Determine per-ad status (required table column)
        # - Bad if any negative alerts
        # - Great if any positive alerts
        # - Mixed if both
        by_ad: Dict[str, Dict[str, bool]] = {}
        for a in alerts:
            by_ad.setdefault(a.ad_name, {"bad": False, "great": False})
            if a.is_positive:
                by_ad[a.ad_name]["great"] = True
            else:
                by_ad[a.ad_name]["bad"] = True

        def status_for(ad_display_name: str) -> str:
            s = by_ad.get(ad_display_name, {"bad": False, "great": False})
            if s["bad"] and s["great"]:
                return "Mixed"
            if s["bad"]:
                return "Bad"
            if s["great"]:
                return "Great"
            return "OK"

        # Only ads that triggered alerts (match by display_name)
        alert_names = set(by_ad.keys())
        table_ads = [a for a in ads if a.display_name() in alert_names][:max_rows]
        if not table_ads:
            return ""

        header = "Ad | Stage | Spend | ATC | $/ATC | IC | $/IC | Orders | CPA | AOV | Bounce% | AvgSession(s) | Status"
        sep = "-" * len(header)
        rows = [header, sep]

        for ad in table_ads:
            row = " | ".join([
                ad.display_name(30),
                ad.stage.value,
                self._fmt_money(ad.spend),
                str(ad.atc_value()) if ad.atc is not None else "N/A",
                self._fmt_money(ad.cost_per_atc),
                str(ad.ic_value()) if ad.ic is not None else "N/A",
                self._fmt_money(ad.cost_per_ic),
                str(ad.orders_value()) if ad.orders is not None else "N/A",
                self._fmt_money(ad.cpa),
                self._fmt_money(ad.aov),
                self._fmt_pct(ad.bounce_rate),
                self._fmt_seconds(ad.avg_session_duration_seconds),
                status_for(ad.display_name()),
            ])
            rows.append(row)

        return "\n".join(rows)


# ------------------------------------------------------- #
# MAIN EXECUTION                                          #
# ------------------------------------------------------- #

def run_monitor(df: Optional[pd.DataFrame] = None, report_date: Optional[str] = None) -> str:
    """
    Main entry point.
    
    Args:
        df: DataFrame with ad data. If None, uses sample data.
        report_date: Report date. If None, uses data date or today.
    
    Returns:
        The generated report string.
    """
    monitor = PaidMediaMonitor()
    
    if df is None:
        df = generate_sample_data()
    
    ads, dq = monitor.process(df)
    return monitor.generate_report(ads, dq, report_date=report_date)


if __name__ == "__main__":
    # Run with sample data
    print(run_monitor())

