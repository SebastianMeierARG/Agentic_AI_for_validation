from shiny import App, ui, render, reactive, req
import pandas as pd
import json
from pathlib import Path
import os
import shutil
import asyncio

# --- Project Paths ---
root_dir = Path(__file__).parent
src_dir = root_dir / "src"
inputs_dir = root_dir / "inputs"
outputs_dir = root_dir / "outputs"
www_dir = root_dir / "images"

audit_results_path = outputs_dir / "audit_results.json"
validation_report_path = outputs_dir / "validation_comparison_report.csv"
client_summary_path = outputs_dir / "client_summary.md"
input_csv_path = inputs_dir / "rcm_input.csv"
expert_csv_path = inputs_dir / "expert_answer.csv"
docs_dir = inputs_dir / "docs"

# Ensure directories exist
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

# --- Data Loading ---
def load_audit_data():
    try:
        if audit_results_path.exists():
            with open(audit_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading audit results: {e}")
    return pd.DataFrame()

def load_validation_data():
    try:
        if validation_report_path.exists():
            return pd.read_csv(validation_report_path, sep=";")
    except Exception as e:
        print(f"Error loading validation report: {e}")
    return pd.DataFrame()

def load_client_summary():
    try:
        if client_summary_path.exists():
            with open(client_summary_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error loading client summary: {e}")
    return None

def get_choices(df, col_name):
    if df.empty or col_name not in df.columns:
        return ["All"]
    items = sorted([str(x) for x in df[col_name].dropna().unique()])
    return ["All"] + items

# --- UI Definition ---
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Control Center",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("1. Upload Inputs"),
                ui.input_file("upload_docs", "Knowledge Base (PDFs)", multiple=True, accept=".pdf"),
                ui.input_file("upload_rcm", "Risk Control Matrix (CSV)", accept=".csv"),
                ui.input_file("upload_expert", "Expert Answers (CSV - Opt)", accept=".csv"),
                ui.hr(),
                ui.h4("2. Execution"),
                ui.input_action_button("btn_run_summary", "Generate Client Summary", class_="btn-secondary w-100 mb-2"),
                ui.input_action_button("btn_run_audit", "Run Audit Engine", class_="btn-primary w-100 mb-2"),
                ui.input_action_button("btn_run_val", "Run Expert Validation", class_="btn-warning w-100"),
                width=400
            ),
            ui.card(
                ui.card_header("Pipeline Logs & Status"),
                ui.output_text("terminal_logs", inline=False),
                full_screen=True
            )
        )
    ),
    ui.nav_panel(
        "Client Summary",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Client Policy Summary"),
                ui.p(
                    "AI-generated overview of the client's IFRS 9 policies across 13 governance topics.",
                    style="color: gray; font-size: 0.9em;"
                ),
                ui.hr(),
                ui.input_action_button("btn_regen_summary", "Regenerate Summary", class_="btn-secondary w-100"),
                width=300,
            ),
            ui.card(
                ui.card_header("IFRS 9 Policy Summary"),
                ui.output_ui("client_summary_content"),
                full_screen=True,
            )
        )
    ),
    ui.nav_panel(
        "Audit Findings",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Filters"),
                ui.output_ui("audit_filters"),
                ui.hr(),
                ui.download_button("download_audit", "Download Audit JSON", class_="btn-success w-100"),
            ),
            ui.card(
                ui.card_header("Audit Results Data (Select a row for details)"),
                ui.output_data_frame("audit_grid"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Detailed Audit View"),
                ui.output_ui("detail_view")
            )
        )
    ),
    ui.nav_panel(
        "Validation Report",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Score Overview"),
                ui.output_ui("val_kpi_cards"),
                ui.hr(),
                ui.download_button("download_val", "Download Validation CSV", class_="btn-success w-100"),
                width=280,
            ),
            ui.card(
                ui.card_header("Validation Comparison Report (Select a row for details)"),
                ui.output_data_frame("validation_table"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Row Detail"),
                ui.output_ui("val_detail_view")
            )
        )
    ),
    title=ui.span(
        ui.img(src="Logo/GT_Logo_name.png", height="140px", style="margin-right: 20px; margin-bottom: 5px; border-radius: 5px;"),
        "AI Validator Dashboard"
    ),
    id="tabs"
)

# --- Server Logic ---
def server(input, output, session):
    df_audit_state = reactive.Value(load_audit_data())
    df_val_state = reactive.Value(load_validation_data())
    summary_state = reactive.Value(load_client_summary())
    log_stream = reactive.Value("Waiting for execution...")

    # ---------------- File Upload Handlers ----------------
    @reactive.Effect
    @reactive.event(input.upload_docs)
    def handle_docs_upload():
        files = input.upload_docs()
        if not files: return
        for f in os.listdir(docs_dir):
            os.remove(os.path.join(docs_dir, f))
        for file_info in files:
            shutil.copy(file_info["datapath"], docs_dir / file_info["name"])
        log_stream.set(log_stream.get() + f"\n[System] Uploaded {len(files)} PDF document(s).")

    @reactive.Effect
    @reactive.event(input.upload_rcm)
    def handle_rcm_upload():
        files = input.upload_rcm()
        if files:
            shutil.copy(files[0]["datapath"], input_csv_path)
            log_stream.set(log_stream.get() + "\n[System] Replaced Risk Control Matrix input file.")

    @reactive.Effect
    @reactive.event(input.upload_expert)
    def handle_expert_upload():
        files = input.upload_expert()
        if files:
            shutil.copy(files[0]["datapath"], expert_csv_path)
            log_stream.set(log_stream.get() + "\n[System] Replaced Expert Answers input file.")

    # ---------------- Subprocess Execution ----------------
    _BUTTONS = ["btn_run_summary", "btn_run_audit", "btn_run_val", "btn_regen_summary"]

    async def run_subprocess(script_name):
        log_stream.set(f"--- Starting {script_name} ---\n")
        try:
            python_exe = "python"
            if (root_dir / ".venv" / "Scripts" / "python.exe").exists():
                python_exe = str(root_dir / ".venv" / "Scripts" / "python.exe")

            process = await asyncio.create_subprocess_exec(
                python_exe, str(src_dir / script_name),
                cwd=str(root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace').rstrip()
                log_stream.set(log_stream.get() + "\n" + decoded_line)
            await process.wait()
            log_stream.set(log_stream.get() + f"\n--- {script_name} Completed (Exit Code {process.returncode}) ---\n")
        except Exception as e:
            log_stream.set(log_stream.get() + f"\n[Subprocess Error]: {e}")

    async def _run_locked(script_name, refresh_fn=None):
        for btn in _BUTTONS:
            ui.update_action_button(btn, disabled=True)
        await run_subprocess(script_name)
        if refresh_fn:
            refresh_fn()
        for btn in _BUTTONS:
            ui.update_action_button(btn, disabled=False)

    @reactive.Effect
    @reactive.event(input.btn_run_summary)
    async def trigger_summary():
        await _run_locked("run_summary.py", lambda: summary_state.set(load_client_summary()))

    @reactive.Effect
    @reactive.event(input.btn_regen_summary)
    async def trigger_regen_summary():
        await _run_locked("run_summary.py", lambda: summary_state.set(load_client_summary()))

    @reactive.Effect
    @reactive.event(input.btn_run_audit)
    async def trigger_audit():
        await _run_locked("run_audit.py", lambda: df_audit_state.set(load_audit_data()))

    @reactive.Effect
    @reactive.event(input.btn_run_val)
    async def trigger_validation():
        await _run_locked("validate_audit.py", lambda: df_val_state.set(load_validation_data()))

    @render.text
    def terminal_logs():
        return log_stream.get()

    # ---------------- Client Summary ----------------
    @render.ui
    def client_summary_content():
        content = summary_state.get()
        if content is None:
            return ui.div(
                ui.p(
                    "No client summary found. Click 'Generate Client Summary' in the Control Center "
                    "or 'Regenerate Summary' to create one.",
                    style="color: gray; font-style: italic; padding: 20px;"
                )
            )
        return ui.div(ui.markdown(content), style="padding: 10px;")

    # ---------------- Audit Filters & Table ----------------
    @render.ui
    def audit_filters():
        df = df_audit_state.get()
        return ui.TagList(
            ui.input_select("scope_filter", "Scope", choices=get_choices(df, "Scope")),
            ui.input_select("verdict_filter", "Compliance Verdict", choices=get_choices(df, "Compliance_Verdict")),
            ui.input_select("tier_filter", "Tier", choices=get_choices(df, "Tier (1/2/3)")),
        )

    @reactive.calc
    def filtered_audit_df():
        df = df_audit_state.get().copy()
        if df.empty: return df
        if input.scope_filter() and input.scope_filter() != "All" and "Scope" in df.columns:
            df = df[df["Scope"].astype(str) == input.scope_filter()]
        if input.verdict_filter() and input.verdict_filter() != "All" and "Compliance_Verdict" in df.columns:
            df = df[df["Compliance_Verdict"].astype(str) == input.verdict_filter()]
        if input.tier_filter() and input.tier_filter() != "All" and "Tier (1/2/3)" in df.columns:
            df = df[df["Tier (1/2/3)"].astype(str) == input.tier_filter()]
        return df

    @render.data_frame
    def audit_grid():
        df = filtered_audit_df()
        cols_to_drop = [c for c in ["AI_Answer", "Evidence_Sources"] if c in df.columns]
        display_df = df.drop(columns=cols_to_drop) if not df.empty else df
        return render.DataGrid(display_df, selection_mode="row", filters=True)

    @render.ui
    def detail_view():
        df = filtered_audit_df().reset_index(drop=True)
        try:
            selection = audit_grid.cell_selection()
        except AttributeError:
            return ui.p("Data grid selection not supported by your Shiny version.")
        if selection and "rows" in selection and len(selection["rows"]) > 0:
            row_idx = selection["rows"][0]
            try:
                view_df = audit_grid.data_view()
                if row_idx < len(view_df):
                    row = view_df.iloc[row_idx]
                    orig_idx = row.name
                    orig_row = df.loc[orig_idx]
                    return ui.div(
                        ui.h5("AI Answer:"),
                        ui.p(str(orig_row.get("AI_Answer", "N/A"))),
                        ui.br(),
                        ui.h5("Evidence Sources:"),
                        ui.p(str(orig_row.get("Evidence_Sources", "N/A")))
                    )
            except Exception as e:
                return ui.p(f"Error retrieving details: {str(e)}")
        return ui.p("Please select a row in the table above to view full details.", style="color: gray; font-style: italic;")

    # ---------------- Validation KPI Cards ----------------
    @render.ui
    def val_kpi_cards():
        df = df_val_state.get()
        if df.empty:
            return ui.p("No validation data loaded.", style="color: gray; font-style: italic;")

        score_cols = {
            "Cross-Encoder": "Cross_Encoder_Score",
            "LLM Accuracy": "Validation_Score_LLM_as_judge",
            "Stability": "Stability_Score",
            "Drift Resistance": "Drift_Resistance_Score",
            "Guardrails": "Guardrail_Effectiveness_Score",
        }

        def score_color(val):
            if val >= 75: return "#198754"
            if val >= 50: return "#fd7e14"
            return "#dc3545"

        def kpi_card(label, value_str, color):
            return ui.card(
                ui.div(
                    ui.div(label, style=f"font-size: 0.72em; font-weight: 600; color: {color}; text-transform: uppercase; letter-spacing: 0.04em;"),
                    ui.div(value_str, style=f"font-size: 1.5em; font-weight: 700; color: {color};"),
                    style="text-align: center; padding: 6px 4px;"
                ),
                style=f"border-left: 4px solid {color}; margin-bottom: 7px;"
            )

        cards = []
        for label, col in score_cols.items():
            if col in df.columns:
                avg = df[col].mean()
                cards.append(kpi_card(label, f"{avg:.1f}", score_color(avg)))
            else:
                cards.append(kpi_card(label, "N/A", "#6c757d"))

        cards.append(kpi_card("Controls Evaluated", str(len(df)), "#0d6efd"))
        return ui.TagList(*cards)

    @render.data_frame
    def validation_table():
        df = df_val_state.get().copy()
        if df.empty:
            return render.DataTable(df)
        return render.DataGrid(df, filters=True, selection_mode="row")

    @render.ui
    def val_detail_view():
        df = df_val_state.get().copy()
        if df.empty:
            return ui.p("No validation data loaded.", style="color: gray; font-style: italic;")
        try:
            selection = validation_table.cell_selection()
        except AttributeError:
            return ui.p("Select a row to view reasoning.", style="color: gray; font-style: italic;")
        if selection and "rows" in selection and len(selection["rows"]) > 0:
            row_idx = selection["rows"][0]
            try:
                view_df = validation_table.data_view()
                if row_idx < len(view_df):
                    row = view_df.iloc[row_idx]
                    orig_row = df.loc[row.name] if (hasattr(row, 'name') and row.name in df.index) else row
                    parts = [ui.h5(f"Control: {orig_row.get('Control Reference', 'N/A')}")]
                    if "Reasoning_LLM_as_judge" in orig_row:
                        parts += [ui.h6("LLM Judge Reasoning:"), ui.p(str(orig_row.get("Reasoning_LLM_as_judge", "")))]
                    if "Expert_Answer_EN" in orig_row:
                        parts += [ui.h6("Expert Answer (EN):"), ui.p(str(orig_row.get("Expert_Answer_EN", "")))]
                    return ui.div(*parts)
            except Exception as e:
                return ui.p(f"Error: {str(e)}")
        return ui.p("Select a row to view LLM reasoning and expert answer.", style="color: gray; font-style: italic;")

    # ---------------- Downloads ----------------
    @render.download(filename="audit_results.json")
    def download_audit():
        if audit_results_path.exists():
            with open(audit_results_path, "rb") as f:
                yield f.read()

    @render.download(filename="validation_report.csv")
    def download_val():
        if validation_report_path.exists():
            with open(validation_report_path, "rb") as f:
                yield f.read()

app = App(app_ui, server, static_assets=www_dir)
