from shiny import App, ui, render, reactive, req
import pandas as pd
import json
import yaml
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
config_path = root_dir / "config.yaml"

input_csv_path = inputs_dir / "rcm_input.csv"
expert_csv_path = inputs_dir / "rcm_expert_answer.csv"
docs_dir = root_dir / "documents"

# Ensure directories exist
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

# --- Output Discovery (latest timestamped run folder) ---
def find_latest_run_dir():
    """Return the most recently modified timestamped run subfolder under outputs/, or None."""
    subdirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)

def get_output_file(filename_pattern):
    """
    Find a file matching filename_pattern in the latest run dir, then fall back to
    the outputs root. Returns Path or None.
    """
    run_dir = find_latest_run_dir()
    if run_dir:
        matches = list(run_dir.glob(filename_pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    matches = list(outputs_dir.glob(filename_pattern))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    return None

# --- Data Loading ---
def load_audit_data():
    try:
        path = get_output_file("audit_results.json")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading audit results: {e}")
    return pd.DataFrame()

def load_validation_data():
    try:
        path = get_output_file("val_metrics*.csv")
        if path:
            return pd.read_csv(path, sep=";")
    except Exception as e:
        print(f"Error loading validation report: {e}")
    return pd.DataFrame()

def load_flagged_data():
    try:
        path = get_output_file("flagged_for_review.json")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading flagged review data: {e}")
    return []

def load_run_manifest():
    try:
        path = get_output_file("run_manifest.json")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading run manifest: {e}")
    return {}

def load_client_summary():
    try:
        path = get_output_file("client_summary.md")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error loading client summary: {e}")
    return None

# --- Config Helpers ---
def load_config():
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config: {e}")
    return {}

def save_config(cfg):
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

def get_choices(df, col_name):
    if df.empty or col_name not in df.columns:
        return ["All"]
    items = sorted([str(x) for x in df[col_name].dropna().unique()])
    return ["All"] + items

# --- UI Definition ---
app_ui = ui.page_navbar(
    # ------------------------------------------------------------------ #
    # Tab 1 — Control Center
    # ------------------------------------------------------------------ #
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
                width=400,
            ),
            ui.card(
                ui.card_header("Pipeline Logs & Status"),
                ui.output_text("terminal_logs", inline=False),
                full_screen=True,
            ),
        ),
    ),
    # ------------------------------------------------------------------ #
    # Tab 2 — Configuration
    # ------------------------------------------------------------------ #
    ui.nav_panel(
        "Configuration",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Pipeline Configuration"),
                ui.p(
                    "Edit settings below, then click Save to write config.yaml before running.",
                    style="color: gray; font-size: 0.9em;",
                ),
                ui.hr(),
                ui.input_action_button("btn_save_config", "Save Configuration", class_="btn-primary w-100 mb-2"),
                ui.input_action_button("btn_reload_config", "Reload from File", class_="btn-secondary w-100 mb-2"),
                ui.hr(),
                ui.output_ui("config_save_status"),
                width=280,
            ),
            ui.accordion(
                # ---- LLM Settings ----
                ui.accordion_panel(
                    "LLM Settings",
                    ui.layout_columns(
                        ui.input_select("cfg_provider", "Primary Provider", choices=["openai", "google"], selected="openai"),
                        ui.input_numeric("cfg_temperature", "Temperature", value=0.0, min=0.0, max=2.0, step=0.1),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.input_text("cfg_openai_model", "OpenAI Model", value="gpt-4o-mini"),
                        ui.input_text("cfg_openai_embedding", "OpenAI Embedding Model", value="text-embedding-3-small"),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.input_text("cfg_google_model", "Google Model", value="models/gemini-pro-latest"),
                        ui.input_text("cfg_google_embedding", "Google Embedding Model", value="models/embedding-001"),
                        col_widths=[6, 6],
                    ),
                ),
                # ---- RAG Settings ----
                ui.accordion_panel(
                    "RAG Settings",
                    ui.layout_columns(
                        ui.input_numeric("cfg_chunk_size", "Chunk Size", value=1200, min=100, step=100),
                        ui.input_numeric("cfg_chunk_overlap", "Chunk Overlap", value=300, min=0, step=50),
                        ui.input_text("cfg_doc_language", "Document Language", value="Spanish"),
                        col_widths=[4, 4, 4],
                    ),
                    ui.layout_columns(
                        ui.input_numeric("cfg_retrieval_threshold", "L2 Retrieval Threshold", value=1.8, min=0.5, max=3.0, step=0.1),
                        ui.input_numeric("cfg_client_top_k", "Client Top-K", value=6, min=1, max=20),
                        ui.input_numeric("cfg_regs_top_k", "Regs Top-K", value=2, min=0, max=10),
                        ui.input_numeric("cfg_rerank_top_k", "Rerank Top-K", value=7, min=1, max=20),
                        col_widths=[3, 3, 3, 3],
                    ),
                    ui.layout_columns(
                        ui.input_checkbox("cfg_use_regulations", "Use Regulations Index", value=False),
                        ui.input_checkbox("cfg_use_query_decomp", "Use Query Decomposition", value=False),
                        col_widths=[6, 6],
                    ),
                    ui.input_text(
                        "cfg_reranker_model",
                        "Reranker Model",
                        value="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                    ),
                ),
                # ---- Judge LLM ----
                ui.accordion_panel(
                    "Judge LLM",
                    ui.layout_columns(
                        ui.input_select(
                            "cfg_judge_provider",
                            "Judge Provider",
                            choices=["auto", "groq", "ollama", "together", "openai", "google"],
                            selected="auto",
                        ),
                        col_widths=[6],
                    ),
                    ui.input_checkbox_group(
                        "cfg_judge_auto_order",
                        "Auto Priority (providers to try when provider = 'auto')",
                        choices={"groq": "Groq", "ollama": "Ollama (local)", "together": "Together AI"},
                        selected=["groq", "ollama", "together"],
                    ),
                    ui.layout_columns(
                        ui.input_text("cfg_judge_groq_model", "Groq Model", value="llama-3.3-70b-versatile"),
                        ui.input_text(
                            "cfg_judge_together_model",
                            "Together AI Model",
                            value="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.input_text("cfg_judge_ollama_model", "Ollama Model", value="llama3.1:8b"),
                        ui.input_text("cfg_judge_ollama_url", "Ollama Base URL", value="http://localhost:11434"),
                        col_widths=[6, 6],
                    ),
                ),
                # ---- Validation ----
                ui.accordion_panel(
                    "Validation",
                    ui.layout_columns(
                        ui.input_checkbox("cfg_self_critique", "Enable Self-Critique", value=False),
                        ui.input_checkbox("cfg_cross_llm", "Enable Cross-LLM Critique", value=False),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.input_numeric(
                            "cfg_confidence_threshold", "Confidence Threshold (0-100)", value=60.0, min=0, max=100, step=5
                        ),
                        ui.input_numeric(
                            "cfg_retry_threshold", "Self-Critique Retry Threshold (0-10)", value=6, min=0, max=10
                        ),
                        ui.input_numeric("cfg_max_revisions", "Max Revision Attempts", value=2, min=1, max=5),
                        col_widths=[4, 4, 4],
                    ),
                    ui.input_select("cfg_critique_llm", "Critique LLM", choices=["primary", "judge"], selected="primary"),
                ),
                # ---- Audit Trail & Filtering ----
                ui.accordion_panel(
                    "Audit Trail & Filtering",
                    ui.layout_columns(
                        ui.input_checkbox("cfg_audit_trail", "Enable Audit Trail", value=True),
                        ui.input_checkbox("cfg_run_validation", "Run Validation After Audit", value=False),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.input_numeric("cfg_flag_score_threshold", "Flag Score Threshold (0-10)", value=6, min=0, max=10),
                        ui.input_select("cfg_tier", "Tier Filter", choices=["all", "1", "2", "3"], selected="1"),
                        col_widths=[6, 6],
                    ),
                    ui.hr(),
                    ui.h6("Tier Weights (risk-weighted compliance scoring)"),
                    ui.layout_columns(
                        ui.input_numeric("cfg_tier1_weight", "Tier 1", value=3, min=1),
                        ui.input_numeric("cfg_tier2_weight", "Tier 2", value=2, min=1),
                        ui.input_numeric("cfg_tier3_weight", "Tier 3", value=1, min=1),
                        col_widths=[4, 4, 4],
                    ),
                ),
                open="LLM Settings",
                id="config_accordion",
            ),
        ),
    ),
    # ------------------------------------------------------------------ #
    # Tab 3 — Client Summary
    # ------------------------------------------------------------------ #
    ui.nav_panel(
        "Client Summary",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Client Policy Summary"),
                ui.p(
                    "AI-generated overview of the client's IFRS 9 policies across 13 governance topics.",
                    style="color: gray; font-size: 0.9em;",
                ),
                ui.hr(),
                ui.input_action_button("btn_regen_summary", "Regenerate Summary", class_="btn-secondary w-100"),
                width=300,
            ),
            ui.card(
                ui.card_header("IFRS 9 Policy Summary"),
                ui.output_ui("client_summary_content"),
                full_screen=True,
            ),
        ),
    ),
    # ------------------------------------------------------------------ #
    # Tab 4 — Audit Findings
    # ------------------------------------------------------------------ #
    ui.nav_panel(
        "Audit Findings",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Filters"),
                ui.output_ui("audit_filters"),
                ui.hr(),
                ui.download_button("download_audit", "Download Audit JSON", class_="btn-success w-100"),
            ),
            ui.output_ui("flagged_banner"),
            ui.card(
                ui.card_header("Audit Results Data (Select a row for details)"),
                ui.output_data_frame("audit_grid"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Detailed Audit View"),
                ui.output_ui("detail_view"),
            ),
        ),
    ),
    # ------------------------------------------------------------------ #
    # Tab 5 — Validation Report
    # ------------------------------------------------------------------ #
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
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Row Detail"),
                ui.output_ui("val_detail_view"),
            ),
        ),
    ),
    title=ui.span(
        ui.img(src="Logo/GT_Logo_name.png", height="140px", style="margin-right: 20px; margin-bottom: 5px; border-radius: 5px;"),
        "AI Validator Dashboard",
    ),
    id="tabs",
)


# --- Server Logic ---
def server(input, output, session):
    df_audit_state = reactive.Value(load_audit_data())
    df_val_state = reactive.Value(load_validation_data())
    summary_state = reactive.Value(load_client_summary())
    flagged_state = reactive.Value(load_flagged_data())
    manifest_state = reactive.Value(load_run_manifest())
    log_stream = reactive.Value("Waiting for execution...")
    config_save_msg = reactive.Value("")

    # ------------------------------------------------------------------ #
    # Config helpers
    # ------------------------------------------------------------------ #
    def _apply_config_to_ui(cfg):
        """Push all config values into the matching UI inputs."""
        llm = cfg.get("llm_settings", {})
        ui.update_select("cfg_provider", selected=str(llm.get("provider", "openai")))
        ui.update_numeric("cfg_temperature", value=float(llm.get("temperature", 0.0)))
        oa = llm.get("openai", {})
        ui.update_text("cfg_openai_model", value=str(oa.get("model", "gpt-4o-mini")))
        ui.update_text("cfg_openai_embedding", value=str(oa.get("embedding_model", "text-embedding-3-small")))
        gc = llm.get("google", {})
        ui.update_text("cfg_google_model", value=str(gc.get("model", "models/gemini-pro-latest")))
        ui.update_text("cfg_google_embedding", value=str(gc.get("embedding_model", "models/embedding-001")))

        rag = cfg.get("rag_settings", {})
        ui.update_numeric("cfg_chunk_size", value=int(rag.get("chunk_size", 1200)))
        ui.update_numeric("cfg_chunk_overlap", value=int(rag.get("chunk_overlap", 300)))
        ui.update_text("cfg_doc_language", value=str(rag.get("document_language", "Spanish")))
        ui.update_numeric("cfg_retrieval_threshold", value=float(rag.get("retrieval_score_threshold", 1.8)))
        ui.update_numeric("cfg_client_top_k", value=int(rag.get("client_top_k", 6)))
        ui.update_numeric("cfg_regs_top_k", value=int(rag.get("regs_top_k", 2)))
        ui.update_numeric("cfg_rerank_top_k", value=int(rag.get("rerank_top_k", 7)))
        ui.update_checkbox("cfg_use_regulations", value=bool(rag.get("use_regulations", False)))
        ui.update_checkbox("cfg_use_query_decomp", value=bool(rag.get("use_query_decomposition", False)))
        ui.update_text("cfg_reranker_model", value=str(rag.get("reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")))

        judge = cfg.get("judge_llm", {})
        ui.update_select("cfg_judge_provider", selected=str(judge.get("provider", "auto")))
        ui.update_checkbox_group("cfg_judge_auto_order", selected=list(judge.get("auto_order", ["groq", "ollama", "together"])))
        ui.update_text("cfg_judge_groq_model", value=str(judge.get("model", "llama-3.3-70b-versatile")))
        ui.update_text("cfg_judge_together_model", value=str(judge.get("together_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")))
        ui.update_text("cfg_judge_ollama_model", value=str(judge.get("ollama_model", "llama3.1:8b")))
        ui.update_text("cfg_judge_ollama_url", value=str(judge.get("ollama_base_url", "http://localhost:11434")))

        val = cfg.get("validation", {})
        ui.update_checkbox("cfg_self_critique", value=bool(val.get("enable_self_critique", False)))
        ui.update_checkbox("cfg_cross_llm", value=bool(val.get("enable_cross_llm_critique", False)))
        ui.update_numeric("cfg_confidence_threshold", value=float(val.get("confidence_threshold", 60.0)))
        ui.update_numeric("cfg_retry_threshold", value=int(val.get("self_critique_retry_threshold", 6)))
        ui.update_numeric("cfg_max_revisions", value=int(val.get("max_revision_attempts", 2)))
        ui.update_select("cfg_critique_llm", selected=str(val.get("critique_llm", "primary")))

        trail = cfg.get("audit_trail", {})
        ui.update_checkbox("cfg_audit_trail", value=bool(trail.get("enabled", True)))
        ui.update_checkbox("cfg_run_validation", value=bool(trail.get("run_validation", False)))
        ui.update_numeric("cfg_flag_score_threshold", value=int(trail.get("flag_score_threshold", 6)))

        filt = cfg.get("filtering", {})
        ui.update_select("cfg_tier", selected=str(filt.get("tier", "1")))
        weights = filt.get("tier_weights", {"1": 3, "2": 2, "3": 1})
        ui.update_numeric("cfg_tier1_weight", value=int(weights.get("1", 3)))
        ui.update_numeric("cfg_tier2_weight", value=int(weights.get("2", 2)))
        ui.update_numeric("cfg_tier3_weight", value=int(weights.get("3", 1)))

    # Populate inputs from config.yaml on session start
    @reactive.Effect
    def _init_config():
        _apply_config_to_ui(load_config())

    @reactive.Effect
    @reactive.event(input.btn_reload_config)
    def handle_reload_config():
        _apply_config_to_ui(load_config())
        config_save_msg.set("Config reloaded from file.")

    @reactive.Effect
    @reactive.event(input.btn_save_config)
    def handle_save_config():
        try:
            cfg = load_config()  # start from file to preserve untouched keys (e.g. paths)

            # LLM settings
            cfg.setdefault("llm_settings", {})
            cfg["llm_settings"]["provider"] = input.cfg_provider()
            cfg["llm_settings"]["temperature"] = float(input.cfg_temperature())
            cfg["llm_settings"].setdefault("openai", {})
            cfg["llm_settings"]["openai"]["model"] = input.cfg_openai_model()
            cfg["llm_settings"]["openai"]["embedding_model"] = input.cfg_openai_embedding()
            cfg["llm_settings"].setdefault("google", {})
            cfg["llm_settings"]["google"]["model"] = input.cfg_google_model()
            cfg["llm_settings"]["google"]["embedding_model"] = input.cfg_google_embedding()

            # RAG settings
            cfg.setdefault("rag_settings", {})
            cfg["rag_settings"]["chunk_size"] = int(input.cfg_chunk_size())
            cfg["rag_settings"]["chunk_overlap"] = int(input.cfg_chunk_overlap())
            cfg["rag_settings"]["document_language"] = input.cfg_doc_language()
            cfg["rag_settings"]["retrieval_score_threshold"] = float(input.cfg_retrieval_threshold())
            cfg["rag_settings"]["client_top_k"] = int(input.cfg_client_top_k())
            cfg["rag_settings"]["regs_top_k"] = int(input.cfg_regs_top_k())
            cfg["rag_settings"]["rerank_top_k"] = int(input.cfg_rerank_top_k())
            cfg["rag_settings"]["use_regulations"] = bool(input.cfg_use_regulations())
            cfg["rag_settings"]["use_query_decomposition"] = bool(input.cfg_use_query_decomp())
            cfg["rag_settings"]["reranker_model"] = input.cfg_reranker_model()

            # Judge LLM
            cfg.setdefault("judge_llm", {})
            cfg["judge_llm"]["provider"] = input.cfg_judge_provider()
            cfg["judge_llm"]["auto_order"] = list(input.cfg_judge_auto_order() or [])
            cfg["judge_llm"]["model"] = input.cfg_judge_groq_model()
            cfg["judge_llm"]["together_model"] = input.cfg_judge_together_model()
            cfg["judge_llm"]["ollama_model"] = input.cfg_judge_ollama_model()
            cfg["judge_llm"]["ollama_base_url"] = input.cfg_judge_ollama_url()

            # Validation
            cfg.setdefault("validation", {})
            cfg["validation"]["enable_self_critique"] = bool(input.cfg_self_critique())
            cfg["validation"]["enable_cross_llm_critique"] = bool(input.cfg_cross_llm())
            cfg["validation"]["confidence_threshold"] = float(input.cfg_confidence_threshold())
            cfg["validation"]["self_critique_retry_threshold"] = int(input.cfg_retry_threshold())
            cfg["validation"]["max_revision_attempts"] = int(input.cfg_max_revisions())
            cfg["validation"]["critique_llm"] = input.cfg_critique_llm()

            # Audit trail
            cfg.setdefault("audit_trail", {})
            cfg["audit_trail"]["enabled"] = bool(input.cfg_audit_trail())
            cfg["audit_trail"]["flag_score_threshold"] = int(input.cfg_flag_score_threshold())
            cfg["audit_trail"]["run_validation"] = bool(input.cfg_run_validation())

            # Filtering
            cfg.setdefault("filtering", {})
            cfg["filtering"]["tier"] = input.cfg_tier()
            cfg["filtering"]["tier_weights"] = {
                "1": int(input.cfg_tier1_weight()),
                "2": int(input.cfg_tier2_weight()),
                "3": int(input.cfg_tier3_weight()),
            }

            save_config(cfg)
            config_save_msg.set("Configuration saved to config.yaml.")
        except Exception as e:
            config_save_msg.set(f"Error saving config: {e}")

    @render.ui
    def config_save_status():
        msg = config_save_msg.get()
        if not msg:
            return ui.div()
        color = "#198754" if "saved" in msg.lower() or "reloaded" in msg.lower() else "#dc3545"
        return ui.p(msg, style=f"color: {color}; font-size: 0.88em; font-weight: 600; margin: 0;")

    # ------------------------------------------------------------------ #
    # File Upload Handlers
    # ------------------------------------------------------------------ #
    @reactive.Effect
    @reactive.event(input.upload_docs)
    def handle_docs_upload():
        files = input.upload_docs()
        if not files:
            return
        for f in os.listdir(docs_dir):
            fp = os.path.join(docs_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
        for file_info in files:
            shutil.copy(file_info["datapath"], docs_dir / file_info["name"])
        log_stream.set(log_stream.get() + f"\n[System] Uploaded {len(files)} PDF document(s) to {docs_dir}.")

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

    # ------------------------------------------------------------------ #
    # Subprocess Execution
    # ------------------------------------------------------------------ #
    _BUTTONS = ["btn_run_summary", "btn_run_audit", "btn_run_val", "btn_regen_summary"]

    async def run_subprocess(script_name):
        log_stream.set(f"--- Starting {script_name} ---\n")
        try:
            python_exe = "python"
            if (root_dir / ".venv" / "Scripts" / "python.exe").exists():
                python_exe = str(root_dir / ".venv" / "Scripts" / "python.exe")

            # Force UTF-8 stdout so Unicode characters in print() don't raise
            # UnicodeEncodeError on Windows (default cp1252 pipe encoding).
            subprocess_env = os.environ.copy()
            subprocess_env["PYTHONIOENCODING"] = "utf-8"

            process = await asyncio.create_subprocess_exec(
                python_exe, str(src_dir / script_name),
                cwd=str(root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=subprocess_env,
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode("utf-8", errors="replace").rstrip()
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
        def refresh_audit():
            df_audit_state.set(load_audit_data())
            flagged_state.set(load_flagged_data())
            manifest_state.set(load_run_manifest())
        await _run_locked("run_audit.py", refresh_audit)

    @reactive.Effect
    @reactive.event(input.btn_run_val)
    async def trigger_validation():
        await _run_locked("validate_audit.py", lambda: df_val_state.set(load_validation_data()))

    @render.text
    def terminal_logs():
        return log_stream.get()

    # ------------------------------------------------------------------ #
    # Client Summary
    # ------------------------------------------------------------------ #
    @render.ui
    def client_summary_content():
        content = summary_state.get()
        if content is None:
            return ui.div(
                ui.p(
                    "No client summary found. Click 'Generate Client Summary' in the Control Center "
                    "or 'Regenerate Summary' to create one.",
                    style="color: gray; font-style: italic; padding: 20px;",
                )
            )
        return ui.div(ui.markdown(content), style="padding: 10px;")

    # ------------------------------------------------------------------ #
    # Flagged Banner
    # ------------------------------------------------------------------ #
    @render.ui
    def flagged_banner():
        flagged = flagged_state.get()
        manifest = manifest_state.get()
        if not flagged:
            return ui.div()

        n = len(flagged)
        run_id = manifest.get("run_id", "unknown")
        total = manifest.get("total_controls_processed", "?")
        refs = [f.get("control_reference", "?") for f in flagged[:5]]
        more = f" … and {n - 5} more" if n > 5 else ""
        ref_list = ", ".join(refs) + more

        return ui.div(
            ui.div(
                ui.strong(f"⚠ {n} of {total} controls flagged for human review"),
                ui.span(f"  (Run: {run_id})", style="font-size: 0.85em; color: #856404; margin-left: 8px;"),
                ui.br(),
                ui.span(f"Controls: {ref_list}", style="font-size: 0.85em;"),
                style=(
                    "background-color: #fff3cd; border: 1px solid #ffc107; "
                    "border-left: 5px solid #ffc107; border-radius: 4px; "
                    "padding: 10px 14px; margin-bottom: 12px; color: #856404;"
                ),
            )
        )

    # ------------------------------------------------------------------ #
    # Audit Filters & Table
    # ------------------------------------------------------------------ #
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
        if df.empty:
            return df
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
                        ui.p(str(orig_row.get("Evidence_Sources", "N/A"))),
                    )
            except Exception as e:
                return ui.p(f"Error retrieving details: {str(e)}")
        return ui.p("Please select a row in the table above to view full details.", style="color: gray; font-style: italic;")

    # ------------------------------------------------------------------ #
    # Validation KPI Cards
    # ------------------------------------------------------------------ #
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
            if val >= 75:
                return "#198754"
            if val >= 50:
                return "#fd7e14"
            return "#dc3545"

        def kpi_card(label, value_str, color):
            return ui.card(
                ui.div(
                    ui.div(label, style=f"font-size: 0.72em; font-weight: 600; color: {color}; text-transform: uppercase; letter-spacing: 0.04em;"),
                    ui.div(value_str, style=f"font-size: 1.5em; font-weight: 700; color: {color};"),
                    style="text-align: center; padding: 6px 4px;",
                ),
                style=f"border-left: 4px solid {color}; margin-bottom: 7px;",
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
                    orig_row = df.loc[row.name] if (hasattr(row, "name") and row.name in df.index) else row
                    parts = [ui.h5(f"Control: {orig_row.get('Control Reference', 'N/A')}")]
                    if "Reasoning_LLM_as_judge" in orig_row:
                        parts += [ui.h6("LLM Judge Reasoning:"), ui.p(str(orig_row.get("Reasoning_LLM_as_judge", "")))]
                    if "Expert_Answer_EN" in orig_row:
                        parts += [ui.h6("Expert Answer (EN):"), ui.p(str(orig_row.get("Expert_Answer_EN", "")))]
                    return ui.div(*parts)
            except Exception as e:
                return ui.p(f"Error: {str(e)}")
        return ui.p("Select a row to view LLM reasoning and expert answer.", style="color: gray; font-style: italic;")

    # ------------------------------------------------------------------ #
    # Downloads
    # ------------------------------------------------------------------ #
    @render.download(filename="audit_results.json")
    def download_audit():
        path = get_output_file("audit_results.json")
        if path:
            with open(path, "rb") as f:
                yield f.read()

    @render.download(filename="validation_report.csv")
    def download_val():
        path = get_output_file("val_metrics*.csv")
        if path:
            with open(path, "rb") as f:
                yield f.read()


app = App(app_ui, server, static_assets=www_dir)
