"""
app.py
Streamlit Human Approval Interface for Lab 27: Human-in-the-Loop (HITL) LangGraph Workflow.
Provides:
- Real-time customer evaluation and graph triggering
- Human review Action Card with Approve, Reject, and Edit capabilities
- Live state inspection via graph.get_state(config)
- Resumption of execution via graph.update_state and graph.invoke(None, config)
- Persistent Audit Trail dashboard with filtering, metrics, and JSON export
- In-depth interactive answers to Reflection Questions
"""

from datetime import datetime
import json
import uuid
import streamlit as st
import pandas as pd

from graph import create_hitl_graph, audit_logger, CONFIDENCE_THRESHOLD, HARD_RULE_ACTIONS
from models import PRESET_CUSTOMERS, CustomerProfile, AuditEntry

# Page Configuration
st.set_page_config(
    page_title="HITL Agent Dashboard | Lab 27",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished enterprise UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .status-badge-pending {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-approved {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-rejected {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-auto {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .card-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "checkpointer" not in st.session_state:
    from langgraph.checkpoint.memory import MemorySaver
    st.session_state.checkpointer = MemorySaver()

if "graph" not in st.session_state:
    st.session_state.graph = create_hitl_graph(checkpointer=st.session_state.checkpointer)

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

if "last_execution_result" not in st.session_state:
    st.session_state.last_execution_result = None

if "reviewer_id" not in st.session_state:
    st.session_state.reviewer_id = "operator_01"


# Sidebar: Operator Config & Customer Scenario Selection
with st.sidebar:
    st.title("⚙️ Điều khiển & Cấu hình")
    st.session_state.reviewer_id = st.text_input(
        "Mã Nhân viên Reviewer (Reviewer ID):",
        value=st.session_state.reviewer_id
    )

    st.markdown("---")
    st.subheader("🎯 Chọn hồ sơ khách hàng")
    scenario_type = st.radio(
        "Phương thức chọn:",
        ["Hồ sơ mẫu (Preset)", "Nhập tùy chỉnh (Custom)"]
    )

    selected_customer_data = {}
    cust_id = ""

    if scenario_type == "Hồ sơ mẫu (Preset)":
        preset_key = st.selectbox(
            "Chọn khách hàng:",
            options=list(PRESET_CUSTOMERS.keys()),
            format_func=lambda k: f"{k} - {PRESET_CUSTOMERS[k].name}"
        )
        preset = PRESET_CUSTOMERS[preset_key]
        cust_id = preset.customer_id
        selected_customer_data = preset.model_dump()

        st.info(f"**Ghi chú hồ sơ:**\n{preset.notes}")
        st.caption(
            f"• **TOI:** {preset.toi:,.0f} VND\n"
            f"• **Churn Prob:** {preset.churn_probability*100:.0f}%\n"
            f"• **Hạn mức hiện tại:** {preset.current_credit_limit:,.0f} VND"
        )
    else:
        cust_id = st.text_input("Customer ID:", value="CUST_CUSTOM_01")
        cust_name = st.text_input("Họ tên khách hàng:", value="Trần Thị Khách Hàng")
        cust_toi = st.number_input("Total Operating Income (VND):", value=60_000_000, step=5_000_000)
        cust_churn = st.slider("Xác suất rời bỏ (Churn Probability):", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
        cust_limit = st.number_input("Hạn mức thẻ hiện tại (VND):", value=30_000_000, step=5_000_000)
        cust_age = st.number_input("Thời gian gắn bó (tháng):", value=18, min_value=1)

        selected_customer_data = {
            "customer_id": cust_id,
            "name": cust_name,
            "toi": float(cust_toi),
            "churn_probability": float(cust_churn),
            "current_credit_limit": float(cust_limit),
            "account_age_months": int(cust_age)
        }

    st.markdown("---")
    trigger_btn = st.button("🚀 Chạy Đánh Giá (Trigger Workflow)", type="primary", use_container_width=True)

    if trigger_btn:
        # Create a new unique thread ID for this workflow run
        new_thread = f"thread-{cust_id}-{uuid.uuid4().hex[:6]}"
        st.session_state.active_thread_id = new_thread
        st.session_state.last_execution_result = None

        config = {"configurable": {"thread_id": new_thread}}
        initial_state = {
            "customer_id": cust_id,
            "customer_data": selected_customer_data,
            "human_decision": None,
            "reviewer_id": st.session_state.reviewer_id
        }

        # Invoke workflow until first interrupt or completion
        with st.spinner("Agent đang phân tích hồ sơ và định tuyến chính sách..."):
            result = st.session_state.graph.invoke(initial_state, config)
            st.session_state.last_execution_result = result
        st.rerun()


# Main Dashboard Tabs
tab_workflow, tab_audit, tab_reflection = st.tabs([
    "🛡️ Human Approval & Workflow",
    "📜 Audit Trail (Lịch sử kiểm toán)",
    "💡 Reflection & Kiến trúc HITL"
])


with tab_workflow:
    st.markdown('<div class="main-header">🛡️ Hệ thống Quản Trị Churn Risk - Human in the Loop</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">LangGraph Persistent State • Confidence Routing • Hard Policy Override • Operator Decision Panel</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.active_thread_id:
        st.info("👈 Hãy chọn một hồ sơ khách hàng ở thanh bên trái và bấm **'Chạy Đánh Giá (Trigger Workflow)'** để bắt đầu.")
    else:
        active_thread = st.session_state.active_thread_id
        config = {"configurable": {"thread_id": active_thread}}
        snapshot = st.session_state.graph.get_state(config)
        state_values = snapshot.values

        col_left, col_right = st.columns([1.6, 1.0])

        with col_left:
            st.subheader("📋 Chi tiết Trạng thái Graph (Graph State)")
            st.caption(f"Thread ID đang thực thi: `{active_thread}`")

            # Check if workflow is paused (Pending Human Approval) or finished
            is_interrupted = len(snapshot.next) > 0 and snapshot.next[0] == "execute_high_risk_action"

            if is_interrupted:
                st.warning("⚠️ **GRAPH ĐANG TẠM DỪNG (INTERRUPTED) - CHỜ CON NGƯỜI DUYỆT TRƯỚC KHI THỰC THI**")
                
                proposed_action = state_values.get("proposed_action", "N/A")
                confidence = state_values.get("confidence_score", 0.0)
                reasoning = state_values.get("reasoning", "")
                details = state_values.get("action_details", {})
                cust_name = selected_customer_data.get("name", state_values.get("customer_id"))

                with st.container():
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    st.markdown(f"### 🎴 Action Proposal Card - `{cust_id}`")
                    st.markdown(f"**Khách hàng:** {cust_name}")
                    
                    # Routing badge
                    if proposed_action in HARD_RULE_ACTIONS:
                        st.markdown(
                            f"**Hành động đề xuất:** <span class='status-badge-rejected'>🔴 {proposed_action} (High Risk - Hard Policy Override)</span>",
                            unsafe_allow_html=True
                        )
                        st.caption("🔒 **Quy tắc cứng (Hard Rule):** Hành động tăng hạn mức tín dụng luôn bắt buộc phải có sự phê duyệt của con người dù điểm tin cậy đạt 0.99.")
                    else:
                        st.markdown(
                            f"**Hành động đề xuất:** <span class='status-badge-pending'>🟠 {proposed_action} (Low-Risk nhưng bị Escalated do Confidence < 0.85)</span>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"⚠️ **Quy tắc leo thang (Escalation):** Điểm tự tin của Agent ({confidence*100:.1f}%) thấp hơn ngưỡng an toàn (85%).")

                    # Confidence meter
                    st.markdown(f"**Độ tự tin của Agent (Confidence Score):** `{confidence:.2f}` / `1.00`")
                    st.progress(float(confidence))

                    st.markdown(f"**Lý do đề xuất (Agent Reasoning):**\n> *\"{reasoning}\"*")

                    if details:
                        st.markdown("**Thông số chi tiết hành động:**")
                        st.json(details)

                    st.markdown("---")
                    st.markdown("#### ✍️ Quyết định của Human Reviewer:")

                    btn_col1, btn_col2, btn_col3 = st.columns(3)

                    # APPROVE
                    with btn_col1:
                        if st.button("✅ Phê duyệt (Approve)", type="primary", use_container_width=True):
                            with st.spinner("Đang cập nhật state và resume graph..."):
                                st.session_state.graph.update_state(
                                    config,
                                    {
                                        "human_decision": "approve",
                                        "reviewer_id": st.session_state.reviewer_id
                                    },
                                    as_node="evaluate_customer"
                                )
                                resumed_res = st.session_state.graph.invoke(None, config)
                                st.session_state.last_execution_result = resumed_res
                            st.success("Đã phê duyệt và thực thi hành động thành công!")
                            st.rerun()

                    # REJECT
                    with btn_col2:
                        if st.button("❌ Từ chối (Reject)", use_container_width=True):
                            with st.spinner("Đang hủy hành động và resume graph..."):
                                st.session_state.graph.update_state(
                                    config,
                                    {
                                        "human_decision": "reject",
                                        "reviewer_id": st.session_state.reviewer_id
                                    },
                                    as_node="evaluate_customer"
                                )
                                resumed_res = st.session_state.graph.invoke(None, config)
                                st.session_state.last_execution_result = resumed_res
                            st.error("Đã từ chối hành động. Workflow đã hủy bỏ thao tác an toàn.")
                            st.rerun()

                    # EDIT EXPANDER
                    with btn_col3:
                        edit_expanded = st.checkbox("✏️ Chỉnh sửa (Edit)", value=False)

                    if edit_expanded:
                        st.markdown("##### 🛠️ Chỉnh sửa thông số trước khi duyệt:")
                        current_limit_val = details.get("current_limit", 20_000_000)
                        suggested_limit = details.get("proposed_new_limit", current_limit_val + 20_000_000)
                        
                        edited_action = st.selectbox(
                            "Chọn loại hành động thay thế:",
                            ["increase_credit_limit", "send_email", "offer_special_discount", "assign_relationship_manager"],
                            index=0 if proposed_action == "increase_credit_limit" else 1
                        )
                        
                        edited_limit = st.number_input(
                            "Hạn mức mới sau chỉnh sửa (VND):",
                            value=int(suggested_limit),
                            step=5_000_000
                        )
                        
                        operator_note = st.text_area(
                            "Ghi chú chỉnh sửa của Operator:",
                            value="Điều chỉnh mức tăng hạn mức phù hợp với khả năng thu nhập thực tế của khách hàng."
                        )

                        if st.button("🚀 Xác nhận & Thực thi hành động đã sửa (Confirm Edit)", type="primary"):
                            updated_details = dict(details)
                            updated_details["proposed_new_limit"] = float(edited_limit)
                            updated_details["increase_amount"] = float(edited_limit - current_limit_val)
                            updated_details["operator_note"] = operator_note
                            updated_details["edited_by"] = st.session_state.reviewer_id

                            with st.spinner("Đang cập nhật dữ liệu sửa đổi và tiếp tục graph..."):
                                st.session_state.graph.update_state(
                                    config,
                                    {
                                        "proposed_action": edited_action,
                                        "action_details": updated_details,
                                        "human_decision": "edit",
                                        "reviewer_id": st.session_state.reviewer_id
                                    },
                                    as_node="evaluate_customer"
                                )
                                resumed_res = st.session_state.graph.invoke(None, config)
                                st.session_state.last_execution_result = resumed_res
                            st.success("Đã ghi nhận bản sửa đổi và hoàn tất thực thi!")
                            st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                # Completed workflow (either auto-executed or already resumed)
                status = state_values.get("execution_status", "N/A")
                st.success("✅ **WORKFLOW ĐÃ HOÀN TẤT (GRAPH COMPLETED)**")
                
                with st.container():
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    st.markdown(f"### 🏁 Kết quả thực thi cho `{state_values.get('customer_id')}`")
                    st.markdown(f"• **Trạng thái thực thi:** `{status}`")
                    st.markdown(f"• **Hành động cuối cùng:** `{state_values.get('proposed_action')}`")
                    st.markdown(f"• **Quyết định con người:** `{state_values.get('human_decision', 'N/A (Tự động)')}`")
                    st.markdown(f"• **Độ tin cậy:** `{state_values.get('confidence_score', 0.0):.2f}`")
                    st.markdown(f"• **Lý giải:** {state_values.get('reasoning')}")
                    
                    if state_values.get("action_details"):
                        st.json(state_values.get("action_details"))
                    st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.subheader("🔍 LangGraph State Snapshot")
            st.caption("Dữ liệu trực tiếp lấy từ `graph.get_state(config)`:")
            st.json({
                "values": snapshot.values,
                "next_nodes": snapshot.next,
                "config": snapshot.config
            })


with tab_audit:
    st.subheader("📜 Nhật Ký Kiểm Toán Bất Biến (Audit Trail)")
    st.caption("Dữ liệu được lưu trữ và ghi nối tiếp (append-only) tại `audit_log.json`.")

    metrics = audit_logger.get_metrics()
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Tổng lượt đánh giá", metrics["total"])
    m_col2.metric("Phê duyệt (Approve)", metrics["approve"])
    m_col3.metric("Từ chối (Reject)", metrics["reject"])
    m_col4.metric("Chỉnh sửa (Edit)", metrics["edit"])
    m_col5.metric("Tự động (Auto)", metrics["auto_executed"])

    records = audit_logger.get_all()
    if not records:
        st.info("Chưa có bản ghi kiểm toán nào. Hãy chạy thử một kịch bản ở tab Workflow.")
    else:
        df_records = []
        for r in records:
            df_records.append({
                "Timestamp": r.get("timestamp"),
                "Customer ID": r.get("customer_id", "N/A"),
                "Action": r.get("action"),
                "Confidence": f"{r.get('confidence', 0.0):.2f}",
                "Reviewer": r.get("reviewer_id"),
                "Decision": r.get("decision")
            })
        
        st.dataframe(pd.DataFrame(df_records).iloc[::-1], use_container_width=True)

        st.markdown("#### 📂 Chi tiết bản ghi JSON gần nhất:")
        st.json(records[-1])

        # Download button
        st.download_button(
            label="📥 Tải xuống Audit Log (JSON)",
            data=json.dumps(records, indent=2, ensure_ascii=False),
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


with tab_reflection:
    st.subheader("💡 Giải Đáp Chi Tiết 3 Câu Hỏi Phản Biện (Reflection Questions)")

    with st.expander("❓ Câu 1: interrupt_before vs interrupt_after khi Human cần rewrite retention email", expanded=True):
        st.markdown("""
        **Câu hỏi:**
        > Ở Bước 4, chúng ta đã dùng `interrupt_before=["execute_high_risk_action"]`.
        > Nếu mục tiêu của bạn là để con người rewrite một customer retention email vừa được generate trước khi nó di chuyển đến một routing node, bạn sẽ dùng `interrupt_before` hay `interrupt_after`? Tại sao?

        **Trả lời chi tiết:**
        - **Lựa chọn tối ưu:** Sử dụng **`interrupt_after=["generate_retention_email"]`** (hoặc `interrupt_before=["route_email_node"]`).
        - **Giải thích nguyên lý:**
          1. Node `generate_retention_email` cần phải được **thực thi xong trước** để sinh ra nội dung email nháp (draft content) và lưu vào `GraphState`.
          2. Sau khi node hoàn thành, `interrupt_after` sẽ tạm dừng graph ngay lập tức, cho phép giao diện Streamlit trích xuất bản nháp email từ State để nhân viên chỉnh sửa/viết lại (rewrite).
          3. Nhân viên sau khi rewrite sẽ gọi `graph.update_state(config, {"email_content": rewritten_text})` và resume graph. Khi đó, node kế tiếp (routing node hoặc send node) sẽ nhận chính xác bản email đã được tối ưu hóa bởi con người.
          4. Nếu dùng `interrupt_before=["generate_retention_email"]`, graph sẽ dừng lại khi email **chưa được sinh ra**, khiến con người không có bản nháp để review hoặc rewrite!
        """)

    with st.expander("❓ Câu 2: Giải pháp ngăn chặn Alert Fatigue khi có quá nhiều lượt review thủ công", expanded=True):
        st.markdown("""
        **Câu hỏi:**
        > Giả sử Streamlit UI của bạn hiện đang ép human phải review 500 actions `send_email` mỗi ngày vì confidence của agent bị kẹt ở `0.82` ngay dưới threshold `0.85`.
        > Hãy nêu các thay đổi cụ thể về UI/UX hoặc kiến trúc để ngăn chặn **Alert Fatigue** (Hội chứng mệt mỏi vì cảnh báo)?

        **Trả lời chi tiết:**
        1. **Về mặt Kiến trúc & AI Calibrations (Backend/Architecture):**
           - **Dynamic / Adaptive Thresholding:** Thay vì cố định ngưỡng `0.85` cho mọi nhóm khách hàng, phân tách threshold theo rủi ro phân khúc (ví dụ: với khách hàng phổ thông, rủi ro email thấp thì threshold cho phép auto-execute có thể hạ xuống `0.80`).
           - **Fine-tuning & Few-shot Prompting:** Bổ sung ví dụ mẫu và dữ liệu đánh giá để agent tự tin hơn trong các trường hợp an toàn, nâng confidence trung bình từ `0.82` lên trên `0.88`.
           - **Ensemble Validation Node:** Sử dụng một mô hình phân loại nhẹ (như XGBoost/LightGBM) để xác thực chéo; nếu 2 mô hình đồng thuận thì cho phép bypass manual review.
        
        2. **Về mặt Giao diện & Trải nghiệm (UI/UX):**
           - **Batch Review / Batch Approval (Duyệt hàng loạt):** Cho phép operator tích chọn nhiều email tương đồng và bấm *“Duyệt tất cả 50 email nhóm này”* thay vì phải bấm 500 lần riêng lẻ.
           - **Review by Exception / Diff Highlighting:** Chỉ hiển thị những điểm dị biệt (ví dụ: lý do tại sao confidence thấp), làm nổi bật bằng màu sắc để reviewer chỉ mất 2-3 giây đánh giá.
           - **Snooze & Smart Queue Prioritization:** Tự động xếp các email có giá trị cao hoặc rủi ro cao lên đầu danh sách duyệt, các email rủi ro thấp có thể áp dụng cơ chế *Auto-Approve sau 4 giờ nếu không có phản đối*.
        """)

    with st.expander("❓ Câu 3: Rủi ro khi phụ thuộc LLM self-confidence & Phương pháp Calibrate", expanded=True):
        st.markdown("""
        **Câu hỏi:**
        > Bạn nhận thấy agent thường xuyên tự báo confidence là `0.95` khi đề xuất `increase_credit_limit`, nhưng nó lại thường xuyên sai về thu nhập thực tế của khách hàng.
        > Tại sao việc chỉ phụ thuộc vào sự tự đánh giá confidence của LLM lại nguy hiểm? Và làm thế nào bạn có thể calibrate điểm số này trước bước routing?

        **Trả lời chi tiết:**
        - **Tại sao LLM Self-Confidence nguy hiểm?**
          1. **Ảo giác & Overconfidence (Ảo tưởng tự tin):** LLM là mô hình xác suất sinh ngôn ngữ, xu hướng sinh ra các câu khẳng định chắc nịch và điểm confidence cao dù căn cứ số học bị sai lệch.
          2. **Thiếu khả năng tính toán số học chính xác:** LLM có thể đọc nhầm dòng tiền TOI hoặc tính sai tỷ lệ nợ trên thu nhập (DTI), dẫn đến kết luận tăng hạn mức sai lầm gây tổn thất tín dụng nghiêm trọng.
        
        - **Phương pháp Calibrate điểm số trước bước Routing:**
          1. **Tool-Grounded Deterministic Verification:** Sử dụng một Python Tool / Function Call để truy vấn trực tiếp Core Banking API và tính toán công thức cứng:
             $$\\text{Eligible Limit} = \\min(\\text{TOI} \\times 3, \\text{Max Policy Limit})$$
             Nếu LLM đề xuất vượt quá con số tính toán của tool, tự động gán điểm confidence = `0.0` và kích hoạt Policy Violation.
          2. **Logit-based Confidence / Verbalized Calibration:** Thay vì bảo LLM "hãy đoán số từ 0 đến 1", đo entropy hoặc trích xuất log-probabilities của token sinh ra, kết hợp với kỹ thuật *Platt Scaling* hoặc *Isotonic Regression*.
          3. **Self-Consistency & Multi-agent Cross-Check:** Chạy 3 lần sampling (nhiệt độ $T > 0$) hoặc có thêm một *Critic Agent (Risk Officer Agent)* phản biện độc lập. Điểm confidence sẽ là tỷ lệ đồng thuận giữa các agent.
        """)
