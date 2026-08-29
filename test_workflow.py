"""
test_workflow.py
Comprehensive automated test suite for Lab 27 HITL LangGraph workflow.
Tests:
1. Hard Policy Rule: increase_credit_limit (confidence 0.99) -> must interrupt
2. Auto-Execute: send_email (confidence 0.92 >= 0.85) -> auto executes without interrupt
3. Escalation: send_email (confidence 0.82 < 0.85) -> interrupts for human review
4. Human Approve flow: resumes and executes action
5. Human Reject flow: resumes and aborts action
6. Human Edit flow: updates parameters, resumes, and executes edited action
7. Audit Log integrity: records all decisions without overwriting previous logs
"""

import os
import uuid
import pytest
from langgraph.checkpoint.memory import MemorySaver

from graph import create_hitl_graph, GraphState, audit_logger
from models import AuditEntry, PRESET_CUSTOMERS


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure clean test audit log for each test run."""
    test_log_file = "test_audit_log.json"
    audit_logger.log_filepath = test_log_file
    audit_logger.clear()
    yield
    if os.path.exists(test_log_file):
        os.remove(test_log_file)
    audit_logger.log_filepath = "audit_log.json"


def test_hard_policy_rule_override():
    """
    Test Rule 1: Even with confidence = 0.99, increase_credit_limit MUST NOT auto-execute.
    It must trigger interrupt_before=['execute_high_risk_action'].
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": "CUST004",  # Corporate High-Value
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.99,
        "reasoning": "High churn risk with top tier TOI. Financial retention proposed.",
        "human_decision": None
    }

    # Execute workflow until first interruption
    result = graph.invoke(initial_state, config)

    # Check graph state
    snapshot = graph.get_state(config)
    
    # Verify graph is paused before execute_high_risk_action
    assert snapshot.next == ("execute_high_risk_action",), "Graph did not interrupt before execute_high_risk_action!"
    assert snapshot.values["proposed_action"] == "increase_credit_limit"
    assert snapshot.values["confidence_score"] == 0.99
    assert snapshot.values["customer_id"] == "CUST004"


def test_auto_execute_low_risk_high_confidence():
    """
    Test Rule 2: Low risk action (send_email) with confidence >= 0.85 should auto-execute to END.
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": "CUST002",  # Standard moderate risk
        "proposed_action": "send_email",
        "confidence_score": 0.92,
        "reasoning": "Customer has moderate churn risk. Safe loyalty email.",
        "human_decision": None
    }

    result = graph.invoke(initial_state, config)

    snapshot = graph.get_state(config)
    # Workflow should complete (no pending nodes)
    assert snapshot.next == (), "Workflow should have finished without interruption!"
    assert result["execution_status"] == "auto_executed"

    # Verify audit log entry
    records = audit_logger.get_all()
    assert len(records) == 1
    assert records[0]["action"] == "send_email"
    assert records[0]["decision"] == "auto_executed"
    assert records[0]["reviewer_id"] == "system_auto"


def test_escalation_low_confidence():
    """
    Test Rule 3: Low risk action (send_email) with confidence < 0.85 must escalate to human review.
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": "CUST003",
        "proposed_action": "send_email",
        "confidence_score": 0.82,  # Below 0.85 threshold
        "reasoning": "Customer behavior is ambiguous. Low confidence escalation.",
        "human_decision": None
    }

    result = graph.invoke(initial_state, config)

    snapshot = graph.get_state(config)
    # Should be interrupted for human review
    assert snapshot.next == ("execute_high_risk_action",)
    assert snapshot.values["confidence_score"] == 0.82


def test_human_approve_flow():
    """
    Test Full HITL Flow with APPROVE:
    1. Start workflow -> Paused at interrupt
    2. Human reviews and selects 'approve'
    3. State updated via graph.update_state()
    4. Resume execution with graph.invoke(None, config)
    5. Action executes, status is 'executed', audit log recorded.
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": "CUST001",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.94,
        "reasoning": "VIP customer with high churn probability.",
        "human_decision": None
    }

    # Step 1: Trigger workflow
    graph.invoke(initial_state, config)
    snapshot_before = graph.get_state(config)
    assert snapshot_before.next == ("execute_high_risk_action",)

    # Step 2 & 3: Human Operator Approves
    graph.update_state(
        config,
        {
            "human_decision": "approve",
            "reviewer_id": "operator_alice"
        },
        as_node="evaluate_customer"
    )

    # Step 4: Resume graph
    final_result = graph.invoke(None, config)
    snapshot_after = graph.get_state(config)

    # Step 5: Verify completion and audit
    assert snapshot_after.next == ()
    assert final_result["execution_status"] == "executed"

    records = audit_logger.get_all()
    assert len(records) == 1
    assert records[0]["decision"] == "approve"
    assert records[0]["reviewer_id"] == "operator_alice"
    assert records[0]["customer_id"] == "CUST001"


def test_human_reject_flow():
    """
    Test Full HITL Flow with REJECT:
    1. Start workflow -> Paused at interrupt
    2. Human reviews and selects 'reject'
    3. State updated via graph.update_state()
    4. Resume execution -> action aborted.
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": "CUST001",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.94,
        "reasoning": "VIP customer with high churn probability.",
        "human_decision": None
    }

    # Step 1: Start
    graph.invoke(initial_state, config)

    # Step 2: Human Operator Rejects
    graph.update_state(
        config,
        {
            "human_decision": "reject",
            "reviewer_id": "operator_bob"
        },
        as_node="evaluate_customer"
    )

    # Step 3: Resume
    final_result = graph.invoke(None, config)

    # Step 4: Verify
    assert final_result["execution_status"] == "aborted"
    records = audit_logger.get_all()
    assert len(records) == 1
    assert records[0]["decision"] == "reject"
    assert records[0]["reviewer_id"] == "operator_bob"


def test_human_edit_flow():
    """
    Test Full HITL Flow with EDIT:
    1. Start workflow -> Paused at interrupt
    2. Human edits parameters (e.g. adjusts credit limit increase from 50M to 20M)
    3. State updated with new parameters and 'edit' decision
    4. Resume execution -> executed_with_edits
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": "CUST001",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.94,
        "reasoning": "VIP customer with high churn probability.",
        "action_details": {"current_limit": 50_000_000, "proposed_new_limit": 100_000_000},
        "human_decision": None
    }

    # Step 1: Start
    graph.invoke(initial_state, config)

    # Step 2: Human Operator Edits
    edited_details = {"current_limit": 50_000_000, "proposed_new_limit": 70_000_000, "adjusted_by_human": True}
    graph.update_state(
        config,
        {
            "human_decision": "edit",
            "reviewer_id": "operator_charlie",
            "action_details": edited_details
        },
        as_node="evaluate_customer"
    )

    # Step 3: Resume
    final_result = graph.invoke(None, config)

    # Step 4: Verify
    assert final_result["execution_status"] == "executed_with_edits"
    records = audit_logger.get_all()
    assert len(records) == 1
    assert records[0]["decision"] == "edit"
    assert records[0]["reviewer_id"] == "operator_charlie"
    assert records[0]["details"]["action_details"]["proposed_new_limit"] == 70_000_000


def test_audit_log_multiple_history_integrity():
    """
    Test that audit log keeps a chronological record of multiple actions without data corruption.
    """
    memory = MemorySaver()
    graph = create_hitl_graph(checkpointer=memory)

    # Run 1: Auto-executed email
    config1 = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({
        "customer_id": "CUST002",
        "proposed_action": "send_email",
        "confidence_score": 0.95,
        "reasoning": "Auto test"
    }, config1)

    # Run 2: Approved credit increase
    config2 = {"configurable": {"thread_id": "thread-2"}}
    graph.invoke({
        "customer_id": "CUST001",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.95,
        "reasoning": "High risk test"
    }, config2)
    graph.update_state(config2, {"human_decision": "approve", "reviewer_id": "admin"}, as_node="evaluate_customer")
    graph.invoke(None, config2)

    # Verify audit entries
    records = audit_logger.get_all()
    assert len(records) == 2
    assert records[0]["decision"] == "auto_executed"
    assert records[1]["decision"] == "approve"
