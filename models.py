"""
models.py
Defines data structures, Pydantic schemas, and audit log helpers for Lab 27 HITL.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """
    Pydantic schema representing an immutable audit log record
    capturing both agent reasoning and human operator decisions.
    """
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO 8601 timestamp of the decision/action"
    )
    agent_id: str = Field(
        default="churn-risk-agent",
        description="Identifier of the agent that evaluated the customer"
    )
    action: str = Field(
        ...,
        description="The action executed or aborted (e.g. increase_credit_limit, send_email)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed confidence score [0.0 - 1.0]"
    )
    reviewer_id: str = Field(
        ...,
        description="Operator ID (or 'system_auto' if auto-executed)"
    )
    decision: str = Field(
        ...,
        description="Decision outcome: 'approve', 'reject', 'edit', or 'auto_executed'"
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier associated with this audit entry"
    )
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context such as reasoning, edited parameters, execution notes"
    )


class CustomerProfile(BaseModel):
    """
    Customer data model representing banking and churn metrics.
    """
    customer_id: str
    name: str
    toi: float = Field(..., description="Total Operating Income in VND (Annual/Monthly)")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated churn probability")
    current_credit_limit: float = Field(..., description="Current credit limit in VND")
    account_age_months: int = Field(default=24, description="Tenure in months")
    notes: Optional[str] = None


# Sample presets for quick testing and demonstrations
PRESET_CUSTOMERS: Dict[str, CustomerProfile] = {
    "CUST001": CustomerProfile(
        customer_id="CUST001",
        name="Nguyen Van An (VIP - High Churn Risk)",
        toi=150_000_000,
        churn_probability=0.82,
        current_credit_limit=50_000_000,
        account_age_months=36,
        notes="High TOI but recent drop in transaction frequency. Proposes credit limit increase."
    ),
    "CUST002": CustomerProfile(
        customer_id="CUST002",
        name="Tran Thi Bich (Standard - Moderate Risk)",
        toi=35_000_000,
        churn_probability=0.45,
        current_credit_limit=20_000_000,
        account_age_months=18,
        notes="Moderate churn risk. Standard retention email is safe for auto-execution."
    ),
    "CUST003": CustomerProfile(
        customer_id="CUST003",
        name="Le Hoang Nam (Uncertain/Borderline Profile)",
        toi=45_000_000,
        churn_probability=0.58,
        current_credit_limit=25_000_000,
        account_age_months=8,
        notes="Low tenure and conflicting transaction signals. Agent confidence falls below threshold."
    ),
    "CUST004": CustomerProfile(
        customer_id="CUST004",
        name="Pham Minh Duc (Corporate High-Value)",
        toi=300_000_000,
        churn_probability=0.91,
        current_credit_limit=100_000_000,
        account_age_months=48,
        notes="High churn risk with 0.99 confidence. Hard policy must force human review."
    )
}


class AuditLogger:
    """
    Thread-safe and append-friendly helper to persist and retrieve AuditEntry records in audit_log.json.
    """
    def __init__(self, log_filepath: str = "audit_log.json"):
        self.log_filepath = log_filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create empty list JSON file if it doesn't exist or is invalid."""
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
        else:
            try:
                with open(self.log_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        with open(self.log_filepath, "w", encoding="utf-8") as f_out:
                            json.dump([], f_out, indent=2, ensure_ascii=False)
            except Exception:
                with open(self.log_filepath, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2, ensure_ascii=False)

    def log(self, entry: AuditEntry) -> None:
        """Append an AuditEntry to audit_log.json without overwriting history."""
        self._ensure_file_exists()
        try:
            with open(self.log_filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []

        records.append(entry.model_dump())

        with open(self.log_filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all audit records."""
        self._ensure_file_exists()
        try:
            with open(self.log_filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_metrics(self) -> Dict[str, int]:
        """Compute quick metrics for dashboard display."""
        records = self.get_all()
        metrics = {
            "total": len(records),
            "approve": 0,
            "reject": 0,
            "edit": 0,
            "auto_executed": 0
        }
        for rec in records:
            dec = rec.get("decision", "").lower()
            if dec in metrics:
                metrics[dec] += 1
        return metrics

    def clear(self) -> None:
        """Clear audit trail (used primarily for test isolation)."""
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
