"""
graph.py
LangGraph implementation for Lab 27: Human-in-the-Loop (HITL) Churn Risk Assessment.
Includes:
- GraphState schema (TypedDict)
- Agent reasoning node: evaluate_customer
- Conditional edge: route_action (incorporating hard policy rules and confidence routing)
- Execution nodes: execute_low_risk_action, execute_high_risk_action
- Graph compilation with MemorySaver checkpointer and interrupt_before
"""

from typing import Any, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models import AuditEntry, AuditLogger, PRESET_CUSTOMERS

# Global or configurable audit logger
audit_logger = AuditLogger()

# Confidence threshold configuration
CONFIDENCE_THRESHOLD: float = 0.85
HARD_RULE_ACTIONS = {"increase_credit_limit"}


class GraphState(TypedDict):
    """
    Persistent state tracking customer evaluation, agent proposal,
    confidence score, reasoning, human operator feedback, and execution status.
    """
    customer_id: str
    proposed_action: str
    confidence_score: float
    reasoning: str
    human_decision: Optional[str]        # 'approve' | 'reject' | 'edit' | None
    action_details: Optional[Dict[str, Any]]
    execution_status: Optional[str]     # 'pending_approval' | 'auto_executed' | 'executed' | 'aborted'
    customer_data: Optional[Dict[str, Any]]
    reviewer_id: Optional[str]


def evaluate_customer(state: GraphState) -> Dict[str, Any]:
    """
    Agent Reasoning Node:
    Evaluates customer data (TOI, churn probability, tenure) and derives
    a proposed action, confidence score, and explainable reasoning.
    If explicit proposed_action and confidence_score are provided in state, they are respected.
    """
    customer_id = state.get("customer_id", "UNKNOWN")

    # If explicit proposed_action and confidence_score are provided in input state, prioritize them
    if state.get("proposed_action") is not None and state.get("confidence_score") is not None:
        action = state["proposed_action"]
        conf = float(state["confidence_score"])
        reason = state.get("reasoning", f"Evaluation for {action} with confidence {conf:.2f}")
        details = state.get("action_details", {})
        return {
            "proposed_action": action,
            "confidence_score": conf,
            "reasoning": reason,
            "action_details": details,
            "execution_status": "evaluated"
        }

    customer_data = state.get("customer_data")
    # If preset customer ID is provided and no raw data passed, look up preset
    if not customer_data and customer_id in PRESET_CUSTOMERS:
        preset = PRESET_CUSTOMERS[customer_id]
        customer_data = preset.model_dump()

    # Evaluate based on customer data metrics
    churn_prob = customer_data.get("churn_probability", 0.5) if customer_data else 0.5
    toi = customer_data.get("toi", 30_000_000) if customer_data else 30_000_000
    current_limit = customer_data.get("current_credit_limit", 20_000_000) if customer_data else 20_000_000

    # Decision heuristics:
    # 1. High Churn (> 0.70) & High Value (TOI >= 50M) -> Propose credit limit increase (High-Risk)
    if churn_prob >= 0.70:
        new_limit = current_limit + 30_000_000
        proposed_action = "increase_credit_limit"
        # If very high churn and solid TOI, agent confidence is very high (e.g., 0.94 - 0.99)
        confidence_score = 0.96 if churn_prob >= 0.85 else 0.91
        reasoning = (
            f"Customer {customer_id} exhibits critical churn risk ({churn_prob*100:.1f}%) "
            f"despite substantial TOI ({toi:,.0f} VND). Increasing credit limit to {new_limit:,.0f} VND "
            f"is recommended to improve retention and product stickiness."
        )
        action_details = {
            "current_limit": current_limit,
            "proposed_new_limit": new_limit,
            "increase_amount": new_limit - current_limit
        }

    # 2. Moderate Churn (0.35 <= churn_prob < 0.70) with clear tenure -> Send Email (Low-Risk)
    elif churn_prob < 0.50:
        proposed_action = "send_email"
        confidence_score = 0.92
        reasoning = (
            f"Customer {customer_id} has moderate churn probability ({churn_prob*100:.1f}%). "
            f"A proactive loyalty and promotional email campaign is sufficient. No financial risk involved."
        )
        action_details = {
            "email_template": "loyalty_retention_promo_v1",
            "subject": "Ưu đãi đặc quyền dành riêng cho Quý khách",
            "discount_rate": "15%"
        }

    # 3. Borderline / Ambiguous Churn (0.50 <= churn_prob < 0.70) -> Low confidence send_email
    else:
        proposed_action = "send_email"
        confidence_score = 0.78  # Below 0.85 threshold -> triggers escalation
        reasoning = (
            f"Customer {customer_id} has borderline churn probability ({churn_prob*100:.1f}%) "
            f"with conflicting behavioral signals. Proposing retention email with reduced confidence."
        )
        action_details = {
            "email_template": "consultative_checkin_v2",
            "subject": "Chúng tôi có thể hỗ trợ gì thêm cho Quý khách?",
            "requires_custom_message": True
        }

    return {
        "proposed_action": proposed_action,
        "confidence_score": confidence_score,
        "reasoning": reasoning,
        "action_details": action_details,
        "execution_status": "evaluated"
    }


def route_action(state: GraphState) -> str:
    """
    Conditional Routing Function implementing:
    - Rule 1: Hard Policy Override (increase_credit_limit ALWAYS routes to high-risk review)
    - Rule 2: Auto-Execute (low-risk action AND confidence >= 0.85)
    - Rule 3: Escalate/Suggest (confidence < 0.85 forces human review)
    """
    action = state.get("proposed_action", "")
    confidence = state.get("confidence_score", 0.0)

    # Rule 1 - Hard Policy Override: High-risk financial operations ALWAYS require human approval
    if action in HARD_RULE_ACTIONS:
        return "execute_high_risk_action"

    # Rule 2 - Auto-Execute: Low-risk action with high confidence
    if confidence >= CONFIDENCE_THRESHOLD and action == "send_email":
        return "execute_low_risk_action"

    # Rule 3 - Escalate / Human Review: Low confidence or unknown action
    return "execute_high_risk_action"


def execute_low_risk_action(state: GraphState) -> Dict[str, Any]:
    """
    Executes low-risk actions automatically and records the decision in the audit trail.
    """
    action = state.get("proposed_action", "send_email")
    confidence = state.get("confidence_score", 1.0)
    customer_id = state.get("customer_id", "UNKNOWN")
    details = state.get("action_details", {})
    reasoning = state.get("reasoning", "")

    # Record in Audit Log
    entry = AuditEntry(
        agent_id="churn-risk-agent",
        action=action,
        confidence=confidence,
        reviewer_id="system_auto",
        decision="auto_executed",
        customer_id=customer_id,
        details={
            "reasoning": reasoning,
            "action_details": details,
            "routing_rule": "Auto-Execute (Low-Risk + High Confidence >= 0.85)"
        }
    )
    audit_logger.log(entry)

    return {
        "execution_status": "auto_executed",
        "human_decision": None
    }


def execute_high_risk_action(state: GraphState) -> Dict[str, Any]:
    """
    Executes or aborts high-risk actions based on human operator decision (Approve, Reject, Edit)
    and records the final audited decision in the audit trail.
    """
    human_decision = state.get("human_decision", "reject")  # Default safety fallback
    action = state.get("proposed_action", "increase_credit_limit")
    confidence = state.get("confidence_score", 0.0)
    customer_id = state.get("customer_id", "UNKNOWN")
    reviewer_id = state.get("reviewer_id", "operator_01")
    details = state.get("action_details", {})
    reasoning = state.get("reasoning", "")

    status = "pending"
    execution_notes = ""

    if human_decision == "approve":
        status = "executed"
        execution_notes = f"Operator approved action: {action}."
    elif human_decision == "reject":
        status = "aborted"
        execution_notes = f"Operator rejected action: {action}. Action aborted."
    elif human_decision == "edit":
        status = "executed_with_edits"
        execution_notes = f"Operator edited and approved action: {action}."
    else:
        status = "aborted_unknown_decision"
        execution_notes = f"Unknown decision '{human_decision}', aborted for security."

    # Record in Audit Log
    entry = AuditEntry(
        agent_id="churn-risk-agent",
        action=action,
        confidence=confidence,
        reviewer_id=reviewer_id,
        decision=human_decision,
        customer_id=customer_id,
        details={
            "reasoning": reasoning,
            "action_details": details,
            "execution_notes": execution_notes,
            "final_status": status
        }
    )
    audit_logger.log(entry)

    return {
        "execution_status": status
    }


def create_hitl_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Builds and compiles the LangGraph HITL workflow.
    Uses interrupt_before=['execute_high_risk_action'] to pause graph
    execution prior to running any high-risk or escalated actions.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(GraphState)

    # Add Nodes
    builder.add_node("evaluate_customer", evaluate_customer)
    builder.add_node("execute_low_risk_action", execute_low_risk_action)
    builder.add_node("execute_high_risk_action", execute_high_risk_action)

    # Add Edges
    builder.add_edge(START, "evaluate_customer")
    builder.add_conditional_edges(
        "evaluate_customer",
        route_action,
        {
            "execute_low_risk_action": "execute_low_risk_action",
            "execute_high_risk_action": "execute_high_risk_action"
        }
    )
    builder.add_edge("execute_low_risk_action", END)
    builder.add_edge("execute_high_risk_action", END)

    # Compile with MemorySaver and interrupt_before
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_high_risk_action"]
    )
    return graph
