"""
demo_cli.py
Demonstration script executing the LangGraph Human-in-the-Loop workflow via CLI.
Walks through:
1. Auto-execution scenario (Low-risk + High confidence)
2. Escalation scenario (Low confidence -> Human Review)
3. Hard Policy Override scenario (High-risk action -> Interrupt before execution -> Human Review & Resume)
"""

import uuid
from graph import create_hitl_graph, audit_logger
from models import PRESET_CUSTOMERS


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_demo():
    print_separator("LAB 27: HUMAN-IN-THE-LOOP (HITL) WORKFLOW DEMO")

    graph = create_hitl_graph()

    # -------------------------------------------------------------
    # Scenario 1: Auto-Execute (Low-Risk + High Confidence >= 0.85)
    # -------------------------------------------------------------
    print_separator("SCENARIO 1: AUTO-EXECUTE (Low-Risk, Confidence >= 0.85)")
    cust1 = PRESET_CUSTOMERS["CUST002"]
    thread1 = f"thread-{uuid.uuid4().hex[:6]}"
    config1 = {"configurable": {"thread_id": thread1}}

    print(f"Customer: {cust1.name} ({cust1.customer_id})")
    print(f"Churn Probability: {cust1.churn_probability*100:.0f}% | TOI: {cust1.toi:,.0f} VND")
    print("Triggering LangGraph workflow...")

    res1 = graph.invoke({
        "customer_id": cust1.customer_id,
        "customer_data": cust1.model_dump(),
        "human_decision": None
    }, config1)

    snap1 = graph.get_state(config1)
    print(f"Graph Interrupted? {'Yes' if snap1.next else 'No (Finished directly to END)'}")
    print(f"Proposed Action: {snap1.values.get('proposed_action')}")
    print(f"Confidence Score: {snap1.values.get('confidence_score')}")
    print(f"Execution Status: {res1.get('execution_status')}")

    # -------------------------------------------------------------
    # Scenario 2: Hard Policy Override (increase_credit_limit, Conf = 0.96)
    # -------------------------------------------------------------
    print_separator("SCENARIO 2: HARD POLICY OVERRIDE (High-Risk Action -> Interrupt Before)")
    cust2 = PRESET_CUSTOMERS["CUST001"]
    thread2 = f"thread-{uuid.uuid4().hex[:6]}"
    config2 = {"configurable": {"thread_id": thread2}}

    print(f"Customer: {cust2.name} ({cust2.customer_id})")
    print(f"Churn Probability: {cust2.churn_probability*100:.0f}% | Current Limit: {cust2.current_credit_limit:,.0f} VND")
    print("Triggering LangGraph workflow...")

    graph.invoke({
        "customer_id": cust2.customer_id,
        "customer_data": cust2.model_dump(),
        "human_decision": None
    }, config2)

    snap2 = graph.get_state(config2)
    print(f"\n[INTERRUPT TRIGGERED]")
    print(f"Next Node to run: {snap2.next}")
    print(f"Proposed Action: {snap2.values.get('proposed_action')}")
    print(f"Confidence Score: {snap2.values.get('confidence_score')}")
    print(f"Reasoning: {snap2.values.get('reasoning')}")
    print("Action parameters:", snap2.values.get('action_details'))

    print("\n--- Operator reviews proposal on Dashboard ---")
    print("Operator Decision: APPROVE")
    
    # Update state with human decision
    graph.update_state(
        config2,
        {"human_decision": "approve", "reviewer_id": "operator_phuc"},
        as_node="evaluate_customer"
    )

    # Resume graph execution
    print("Resuming graph execution via graph.invoke(None, config)...")
    res2 = graph.invoke(None, config2)
    snap2_after = graph.get_state(config2)

    print(f"Execution Status: {res2.get('execution_status')}")
    print(f"Graph Completed? {'Yes' if not snap2_after.next else 'No'}")

    # -------------------------------------------------------------
    # Audit Trail Summary
    # -------------------------------------------------------------
    print_separator("PERSISTENT AUDIT TRAIL SUMMARY")
    metrics = audit_logger.get_metrics()
    print("Audit Metrics:", metrics)
    print(f"Total Audit Entries in 'audit_log.json': {len(audit_logger.get_all())}")
    for idx, entry in enumerate(audit_logger.get_all()[-3:], 1):
        print(f"\n[{idx}] Time: {entry['timestamp']} | Customer: {entry.get('customer_id')} | Action: {entry['action']} | Decision: {entry['decision']} | Reviewer: {entry['reviewer_id']}")


if __name__ == "__main__":
    run_demo()
