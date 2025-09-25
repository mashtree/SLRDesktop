"""
SLRDesk — MVP Desktop App for SLR & Bibliometrics (PyQt6)
---------------------------------------------------------

What you get (MVP):
- Tab 1: Ingest CSV (Scopus), quick stats (row count, NA% per column), preview table
- Tab 2: Filter & Highlight by keywords (counts, split by cutoff), export filtered
- Tab 3: Bibliometrics (placeholder): show simple charts (docs per year) with Matplotlib
- Tab 4: Jupyter Console (embedded) to explore the current DataFrame `df`
- Sidebar: PDF Viewer (if Qt PDF is available) and Error Log dock
- Persistent local store: `~/.slrdesk/` for config, cache, logs, and a local SQLite database

Project placement (per your convention):
- Place this file under `frontend/desktop/main.py`.
- Put core logic you refactor later under `app/core/` and storage under `app/storage/`.

Dependencies:
  pip install PyQt6 qtconsole ipykernel jupyter_client pandas matplotlib
  # optional (Qt PDF widget for PDF preview)
  pip install PyQt6-Qt6 PyQt6-Qt6-Dbus  # usually installed with PyQt6

Run:
  python main.py

Build (Windows example):
  pyinstaller --name SLRDesk --windowed main.py \
    --collect-all qtconsole --collect-all jupyter_client --collect-all ipykernel

Notes:
- PDF viewer requires QtPdf (PyQt6.QtPdf and PyQt6.QtPdfWidgets). If not present, the app disables PDF pane gracefully.
- For persistence, this MVP sets up an app home `~/.slrdesk/` with `slrdesk.db`, `logs/app.log`, `cache/`.
- CSVs are loaded into memory as pandas DataFrame and optionally saved as Parquet/PKL in cache.
"""
from __future__ import annotations
import os
import sys
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt  # NEW
import re, html

import pandas as pd
import sqlite3  # NEW
# === NEW: optional deps (wrap in try/except agar app tetap jalan bila belum terpasang) ===
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import pycountry
except Exception:
    pycountry = None

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import squarify  # NEW
except Exception:
    squarify = None

# --- Optional: Qt WebEngine untuk tampilan Plotly interaktif ---
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBVIEW_AVAILABLE = True
except Exception:
    WEBVIEW_AVAILABLE = False


from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, QSize, QUrl  # untuk fallback open HTML
from PyQt6.QtGui import QAction, QIcon, QTextOption, QDesktopServices
from PyQt6.QtWidgets import (  # TAMBAH QComboBox
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QSplitter, QLineEdit, QSpinBox, QCheckBox,
    QTableView, QGroupBox, QTextEdit, QProgressBar, QDockWidget, QMessageBox,
    QToolBar, QComboBox, QMenu, QDialog, QFormLayout, QDialogButtonBox
)

# Optional PDF support
try:
    from PyQt6.QtPdf import QPdfDocument
    from PyQt6.QtPdfWidgets import QPdfView
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# Matplotlib (for quick charts)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# qtconsole (embedded Jupyter console)
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

APP_NAME = "SLRDesk"
HOME = Path.home() / ".slrdesk"
DB_PATH = HOME / "slrdesk.db"
LOG_PATH = HOME / "logs" / "app.log"
CACHE_DIR = HOME / "cache"
TITLE_COL_CANDIDATES = ["title", "document title", "article title"]
ABSTRACT_COL_CANDIDATES = ["abstract", "description"]
KEYWORDS_COL_CANDIDATES = ["author keywords", "index keywords", "keywords"]
SHAPEFILE_RELATIVE = Path(__file__).parent / "110m" / "ne_110m_admin_0_countries.shp"
SHAPEFILE_CACHE    = CACHE_DIR / "ne_110m_admin_0_countries.shp"

# ------------------------- Utilities & Persistence -------------------------

def ensure_dirs():
    (HOME / "logs").mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # DB creation would go here if/when using sqlite3/SQLAlchemy

class ErrorLogger:
    def __init__(self, log_path: Path):
        self.path = log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, msg: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log_exception(self, e: Exception):
        tb = "".join(traceback.format_exception(e))
        self.write(tb)
        return tb

# ------------------------- DataFrame <-> Qt Model --------------------------

class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._full_df = df  # untuk paging
        self._df = df
        self._keywords_colors: dict[str, object] = {}
        self._target_cols: list[str] = []  # kolom yang discan highlight

    # Paging
    def page(self, start: int, length: int):
        self.beginResetModel()
        self._df = self._full_df.iloc[start:start+length].copy()
        self.endResetModel()

    def set_keywords_with_colors(self, mapping: dict[str, object]):
        self._keywords_colors = {k: v for k, v in mapping.items() if k}
        self.layoutChanged.emit()

    def set_target_columns(self, cols: list[str]):
        self._target_cols = cols
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return "" if pd.isna(value) else str(value)
        if role == Qt.ItemDataRole.ForegroundRole and self._keywords_colors:
            colname = str(self._df.columns[index.column()])
            if self._target_cols and colname not in self._target_cols:
                return QVariant()
            text = "" if pd.isna(value) else str(value)
            from PyQt6.QtGui import QBrush, QColor
            for w, color in self._keywords_colors.items():
                if w.lower() in text.lower():
                    qcolor = QColor(color) if not isinstance(color, QColor) else color
                    return QBrush(qcolor)
        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section)

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def full_dataframe(self) -> pd.DataFrame:
        """Kembalikan seluruh data (semua halaman)."""
        return self._full_df

# ------------------------------ Matplotlib UI ------------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

# ------------------------------- Jupyter tab -------------------------------

class JupyterConsole(RichJupyterWidget):
    def __init__(self, ns=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.km = QtInProcessKernelManager()
        self.km.start_kernel(show_banner=False)
        self.km.kernel.gui = 'qt'
        self.kc = self.km.client()
        self.kc.start_channels()
        # Initialize matplotlib inline without relying on widget.execute (compat across qtconsole versions)
        try:
            self.kc.execute("%matplotlib inline")
        except Exception:
            # Fallback to a safe import; inline may be set by backend anyway
            self.kc.execute("import matplotlib.pyplot as plt")
        if ns:
            self.push_variables(ns)

    def push_variables(self, ns: dict):
        self.km.kernel.shell.push(ns)

    def shutdown(self):
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel()
        except Exception:
            pass
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel()
        except Exception:
            pass

# ------------------------------- SQLite -------------------------------
# NEW: Simple SQLite manager for articles and PDF paths

class DB:
    """Minimal SQLite manager: articles + pdf path (MVP)."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                abstract TEXT,
                year INTEGER,
                doi TEXT,
                venue TEXT,
                keywords TEXT,
                pdf_path TEXT,
                UNIQUE(doi) ON CONFLICT IGNORE
            );
            """
        )
        self.conn.commit()

    def upsert_article(self, rec: dict):
        cur = self.conn.cursor()
        if rec.get("doi"):
            cur.execute("SELECT id FROM articles WHERE doi=?", (rec["doi"],))
            row = cur.fetchone()
        else:
            cur.execute("SELECT id FROM articles WHERE title=?", (rec.get("title",""),))
            row = cur.fetchone()
        if row:
            _id = row[0]
            cur.execute(
                "UPDATE articles SET title=?, abstract=?, year=?, venue=?, keywords=? WHERE id=?",
                (rec.get("title"), rec.get("abstract"), rec.get("year"), rec.get("venue"), rec.get("keywords"), _id)
            )
        else:
            cur.execute(
                "INSERT INTO articles(title, abstract, year, doi, venue, keywords, pdf_path) VALUES(?,?,?,?,?,?,?)",
                (
                    rec.get("title"), rec.get("abstract"), rec.get("year"), rec.get("doi"),
                    rec.get("venue"), rec.get("keywords"), rec.get("pdf_path")
                )
            )
            _id = cur.lastrowid
        self.conn.commit()
        return _id

    def attach_pdf(self, identifier: dict, pdf_path: str) -> bool:
        cur = self.conn.cursor()
        if identifier.get("doi"):
            cur.execute("UPDATE articles SET pdf_path=? WHERE doi=?", (pdf_path, identifier["doi"]))
        else:
            cur.execute("UPDATE articles SET pdf_path=? WHERE title=?", (pdf_path, identifier.get("title","")))
        self.conn.commit()
        return cur.rowcount > 0

class FilterDialog(QDialog):
    def __init__(self, parent=None, columns: list[str] | None=None, current: dict | None=None):
        super().__init__(parent)
        self.setWindowTitle("Keyword Filter Settings")
        self.setModal(True)
        self.resize(560, 280)

        lay = QVBoxLayout(self)

        # Keywords + warna (format yang sudah kamu pakai)
        self.ed_keywords = QLineEdit()
        self.ed_keywords.setPlaceholderText("contoh: recovery:#d00000, blockchain:#008000, fault tolerance")

        # Cutoff & case
        self.sp_cutoff = QSpinBox(); self.sp_cutoff.setRange(1, 9999); self.sp_cutoff.setValue(1)
        self.cb_case = QCheckBox("Case sensitive")

        # Target columns (comma)
        self.ed_cols = QLineEdit()
        self.ed_cols.setPlaceholderText("Title, Abstract, Author Keywords, Index Keywords")

        form = QFormLayout()
        form.addRow("Keywords:", self.ed_keywords)
        form.addRow("Cutoff:", self.sp_cutoff)
        form.addRow("", self.cb_case)
        form.addRow("Target columns:", self.ed_cols)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        # isi default
        if current:
            self.ed_keywords.setText(current.get("raw", ""))
            self.sp_cutoff.setValue(current.get("cutoff", 1))
            self.cb_case.setChecked(bool(current.get("case", False)))
            self.ed_cols.setText(", ".join(current.get("cols", [])))

    def value(self):
        return {
            "raw": self.ed_keywords.text().strip(),
            "cutoff": self.sp_cutoff.value(),
            "case": self.cb_case.isChecked(),
            "cols": [c.strip() for c in self.ed_cols.text().split(",") if c.strip()],
        }


# ------------------------------- Main Window -------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.logger = ErrorLogger(LOG_PATH)
        self.setWindowTitle(f"{APP_NAME} — SLR & Bibliometrics")
        self.resize(1200, 800)

        # State
        self.df: pd.DataFrame | None = None
        self.model: PandasModel | None = None
        self.current_csv_path: Path | None = None

        # Layout: Tabs + PDF dock + Error dock
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.filtered_table = None  # sebelum _init_ingest_tab()

        # self._init_toolbar()
        self._init_menubar()
        self._init_ingest_tab()
        self._init_filter_tab()
        self._init_biblio_tab()
        self._init_console_tab()
        self._init_pdf_dock()
        # hide PDF dock by default; show only when user clicks "View PDF of Selected"
        self.pdf_dock.setVisible(False)
        self._init_plotly_dock()
        self._init_error_dock()

        self.tabs.currentChanged.connect(self._on_tab_changed)
        # set visibilitas awal
        self._on_tab_changed(self.tabs.currentIndex())

        self.db = DB(DB_PATH)  # NEW

        self.filtered_df: pd.DataFrame | None = None  # NEW
        self.page_size = 50  # NEW
        self.page_index = 0  # NEW
        self.target_cols: list[str] = []  # NEW


    # -------------------------- UI Construction --------------------------
    def _init_toolbar(self):
        tb = QToolBar("Main")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        act_open = QAction("Open CSV", self)
        act_open.triggered.connect(self.open_csv)
        tb.addAction(act_open)

        act_save_cache = QAction("Cache DF", self)
        act_save_cache.triggered.connect(self.cache_dataframe)
        tb.addAction(act_save_cache)

        act_export = QAction("Export Filtered CSV", self)
        act_export.triggered.connect(self.export_filtered)
        tb.addAction(act_export)
        act_import_db = QAction("Import CSV → DB", self)
        act_import_db.triggered.connect(self.import_csv_to_db)
        tb.addAction(act_import_db)

        act_attach_pdf = QAction("Attach PDF to Selected", self)
        act_attach_pdf.triggered.connect(self.attach_pdf_to_selected)
        tb.addAction(act_attach_pdf)

        act_view_pdf = QAction("View PDF of Selected", self)
        act_view_pdf.triggered.connect(self.view_pdf_of_selected)
        tb.addAction(act_view_pdf)
    
    def _init_menubar(self):
        mb = self.menuBar()

        # ---- File ----
        m_file = mb.addMenu("&File")
        act_open = QAction("Open CSV…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_csv)
        m_file.addAction(act_open)

        m_file.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # ---- Data ----
        m_data = mb.addMenu("&Data")
        act_cache = QAction("Cache DF", self)
        act_cache.triggered.connect(self.cache_dataframe)
        m_data.addAction(act_cache)

        act_export = QAction("Export Filtered CSV", self)
        act_export.triggered.connect(self.export_filtered)
        m_data.addAction(act_export)

        act_import_db = QAction("Import CSV → DB", self)
        act_import_db.triggered.connect(self.import_csv_to_db)
        m_data.addAction(act_import_db)

        # ---- Tools (placeholder untuk nanti) ----
        self.m_tools = mb.addMenu("&Tools")
        act_filter_dlg = QAction("Filter Settings…", self)
        act_filter_dlg.setShortcut("Ctrl+F")
        act_filter_dlg.triggered.connect(self.open_filter_dialog)
        self.m_tools.addAction(act_filter_dlg)

    def _init_ingest_tab(self):
        from PyQt6.QtWidgets import QSplitter
        w = QWidget(); l = QVBoxLayout(w)

        # info "Loaded" dengan tinggi tetap
        info = QLabel("No dataset loaded")
        info.setWordWrap(False)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setMaximumHeight(24)                     # <- tinggi fix
        info.setStyleSheet("QLabel{padding:2px;}")

        # ===== Outer: Kiri–Kanan =====
        outer = QSplitter(Qt.Orientation.Horizontal)

        # ===== KIRI: Preview (atas) + Quick Stats (bawah) =====
        left_split = QSplitter(Qt.Orientation.Vertical)

        # Preview table (atas)
        left_top = QWidget(); lt = QVBoxLayout(left_top); lt.setContentsMargins(0,0,0,0)
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self._install_table_context(self.table)
        lt.addWidget(QLabel("Preview:"))
        lt.addWidget(self.table)

        # Quick stats (bawah)
        left_bottom = QWidget(); lb = QVBoxLayout(left_bottom); lb.setContentsMargins(0,0,0,0)
        self.stats_view = QTextEdit(); self.stats_view.setReadOnly(True)
        self.stats_view.setMinimumHeight(160)
        lb.addWidget(QLabel("Quick Stats:"))
        lb.addWidget(self.stats_view)

        left_split.addWidget(left_top)
        left_split.addWidget(left_bottom)
        left_split.setSizes([800, 300])

        # ===== KANAN: Record Details =====
        right = QWidget(); r = QVBoxLayout(right); r.setContentsMargins(0,0,0,0)
        self.details_view = QTextEdit(); self.details_view.setReadOnly(True)
        self.details_view.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.details_view.setMinimumWidth(380)
        r.addWidget(QLabel("Record Details:"))
        r.addWidget(self.details_view)

        outer.addWidget(left_split)
        outer.addWidget(right)
        outer.setSizes([1100, 500])

        # susun
        l.addWidget(info)
        l.addWidget(outer)

        self.ingest_info_label = info
        self.tab_w_ingest = w
        self.tabs.addTab(w, "Ingest")

    def _connect_table_selection(self, view: QTableView):
        # disconnect lama biar gak dobel (aman dipanggil berulang)
        try:
            view.selectionModel().selectionChanged.disconnect()
        except Exception:
            pass
        view.selectionModel().selectionChanged.connect(
            lambda *_: self._on_row_selected(view)
        )

    def _on_row_selected(self, view: QTableView):
        try:
            idx = view.currentIndex()
            if not idx.isValid():
                self.details_view.clear(); return
            model = view.model()
            if not isinstance(model, PandasModel):
                self.details_view.clear(); return
            row = model.dataframe().iloc[idx.row()]
            html = self._format_record_details(row)
            self.details_view.setHtml(html)
        except Exception as e:
            self._show_error(e)

    def _format_record_details(self, row: pd.Series) -> str:
        """Bentuk ringkas: field-field penting dulu, lalu semua kolom di bawahnya."""
        def gv(*cands):  # get value by candidates (pakai helper kolom yang sudah ada)
            col = self._get_col(*cands)
            return "" if not col else str(row.get(col, "") if col in row.index else "")

        title   = gv("Title", "Document Title", "Article Title")
        year    = gv("Year", "Publication Year")
        source  = gv("Source title", "Journal", "Conference name", "Venue")
        authors = gv("Authors", "Author Names", "Author full names")
        doi     = gv("DOI", "doi")
        abs_    = gv("Abstract", "Description")
        ak      = gv("Author Keywords", "Author_Keywords", "Keywords")
        aff     = gv("Affiliations", "Authors with affiliations", "Author Affiliations")

        # header ringkas
        parts = []
        if title:   parts.append(f"<h3 style='margin:0 0 6px 0'>{title}</h3>")
        meta = []
        if authors: meta.append(authors)
        if source:  meta.append(source)
        if year:    meta.append(str(year))
        if doi:     meta.append(f"DOI: {doi}")
        if meta:
            parts.append("<div style='color:#555;margin-bottom:8px'>" + " • ".join(meta) + "</div>")

        if ak:
            parts.append(f"<b>Author Keywords:</b> {ak}<br>")
        if abs_:
            parts.append(f"<b>Abstract:</b><br><div style='white-space:pre-wrap'>{abs_}</div><br>")
        if aff:
            parts.append(f"<b>Affiliations:</b> {aff}<br>")

        # fallback: tampilkan semua kolom di bawah jika perlu
        parts.append("<hr style='margin:8px 0'>")
        parts.append("<b>All fields:</b><br>")
        for c in row.index:
            val = row[c]
            if pd.isna(val): continue
            parts.append(f"<b>{c}</b>: {val}<br>")

        return "<div style='font-family:Segoe UI,Arial,sans-serif;font-size:12px'>" + "".join(parts) + "</div>"

    def _init_filter_tab(self):
        from PyQt6.QtWidgets import QSplitter
        w = QWidget(); outer = QVBoxLayout(w)

        # Top controls: page + actions
        top_bar = QHBoxLayout()
        self.btn_filter_dlg = QPushButton("Filter Settings…")
        self.btn_filter_dlg.clicked.connect(self.open_filter_dialog)

        self.page_size_spin = QSpinBox(); self.page_size_spin.setRange(10, 1000); self.page_size_spin.setValue(50)
        self.page_index_spin = QSpinBox(); self.page_index_spin.setRange(0, 999999); self.page_index_spin.setValue(0)

        self.btn_first = QPushButton("⏮ First");  self.btn_first.clicked.connect(self.first_page)
        btn_prev = QPushButton("← Prev Page"); btn_prev.clicked.connect(self.prev_page)
        btn_next = QPushButton("Next Page →"); btn_next.clicked.connect(self.next_page)
        self.btn_last = QPushButton("Last ⏭");     self.btn_last.clicked.connect(self.last_page)

        self.lbl_pageinfo = QLabel("Page 1 / 1")

        self.btn_save_filter = QPushButton("Save Filter…"); self.btn_save_filter.clicked.connect(self.save_filtered_records)
        self.btn_sort_hits = QPushButton("Sort by hits (desc)"); self.btn_sort_hits.setCheckable(True); self.btn_sort_hits.setChecked(True)
        self.btn_sort_hits.toggled.connect(lambda _: self._resort_filtered_by_hits())

        self.chk_show_only = QCheckBox("Show only filtered"); self.chk_show_only.setChecked(True)

        top_bar.addWidget(self.btn_filter_dlg)
        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("Page size:")); top_bar.addWidget(self.page_size_spin)
        top_bar.addWidget(QLabel("Page index:")); top_bar.addWidget(self.page_index_spin)
        top_bar.addWidget(self.btn_first); top_bar.addWidget(btn_prev)
        top_bar.addWidget(btn_next); top_bar.addWidget(self.btn_last)
        top_bar.addWidget(self.lbl_pageinfo)
        top_bar.addStretch(1)
        top_bar.addWidget(self.btn_sort_hits)
        top_bar.addWidget(self.chk_show_only)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.btn_save_filter)

        # ==== Split kiri-kanan ====
        split_lr = QSplitter(Qt.Orientation.Horizontal)

        # KIRI: Preview (atas) + Filter Stats (bawah)
        split_left = QSplitter(Qt.Orientation.Vertical)

        # Preview table
        left_top = QWidget(); lt = QVBoxLayout(left_top); lt.setContentsMargins(0,0,0,0)
        self.filtered_table = QTableView(); self.filtered_table.setAlternatingRowColors(True)
        self.filtered_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._install_table_context(self.filtered_table)
        lt.addWidget(QLabel("Filtered Preview (paged):"))
        lt.addWidget(self.filtered_table)

        # Filter stats
        left_bottom = QWidget(); lb = QVBoxLayout(left_bottom); lb.setContentsMargins(0,0,0,0)
        self.filter_stats = QTextEdit(); self.filter_stats.setReadOnly(True)
        self.filter_stats.setMinimumHeight(140)
        lb.addWidget(QLabel("Filter Stats:"))
        lb.addWidget(self.filter_stats)

        split_left.addWidget(left_top)
        split_left.addWidget(left_bottom)
        split_left.setSizes([800, 250])

        # KANAN: Record Details (HTML with highlight)
        right = QWidget(); r = QVBoxLayout(right); r.setContentsMargins(0,0,0,0)
        self.details_view_filter = QTextEdit(); self.details_view_filter.setReadOnly(True)
        self.details_view_filter.setAcceptRichText(True)
        r.addWidget(QLabel("Record Details:"))
        r.addWidget(self.details_view_filter)

        split_lr.addWidget(split_left)
        split_lr.addWidget(right)
        split_lr.setSizes([1100, 520])

        # Assemble
        outer.addLayout(top_bar)
        outer.addWidget(split_lr)

        self.tab_w_filter = w 
        self.tabs.addTab(w, "Filter_Highlight")
        # self.open_filter_dialog()

    # def _init_biblio_tab(self):
    #     w = QWidget(); l = QVBoxLayout(w)
    #     self.canvas = MplCanvas()
    #     l.addWidget(QLabel("Bibliometric: Documents per Year (example)"))
    #     l.addWidget(self.canvas)
    #     btn = QPushButton("Render Chart")
    #     btn.clicked.connect(self.render_docs_per_year)
    #     l.addWidget(btn)
    #     self.tabs.addTab(w, "Bibliometrics")

    def _init_biblio_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        # Selector
        from PyQt6.QtWidgets import QHBoxLayout, QComboBox
        top = QHBoxLayout()
        self.chart_selector = QComboBox()
        self.chart_selector.addItems([
            "1. Distribution of Publications per Year",
            "2. Document Type Distribution (Pie)",
            "3. Publications per Year by Document Type",
            "4. Publication Trends per Year by Document Type",
            "5. Top 10 Sources (Journals/Conferences)",
            "6. WordCloud Keyword (Author_Keywords)",
            "7. WordCloud Keyword (Abstract)",
            "8. Author Collaboration Network (Top 30)",
            "9. Treemap of Keywords (Top 30, Normalized Recovery Terms)",
            "10. Institution Type Distribution (Academic vs Non-Academic)",
            "11. Documents by Country (Top 20)",
            "12. World Map Choropleth (Docs per Country)",
            "13. World Map (GeoPandas) (Docs per Country)",
            "14. Collaboration Scope by Country (Top 20) — Horizontal Stacked",
            "15. Collaboration Scope by Country (Top 20) — % by Category",
            "16. Distribution of Institutional Collaboration Types",
        ])
        btn_render = QPushButton("Render")
        btn_render.clicked.connect(self.render_selected_chart)
        btn_export = QPushButton("Export PNG")
        btn_export.clicked.connect(self.export_current_figure)
        top.addWidget(QLabel("Select chart:"))
        top.addWidget(self.chart_selector, 1)
        top.addWidget(btn_render)
        top.addWidget(btn_export)

        # Canvas
        self.canvas = MplCanvas(width=7, height=4, dpi=100)

        l.addLayout(top)
        l.addWidget(self.canvas)
        self.tab_w_biblio = w
        self.tabs.addTab(w, "Bibliometrics")


    def _init_console_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        self.console = JupyterConsole(ns={"pd": pd})
        l.addWidget(QLabel("Embedded Jupyter Console. Current df is in variable `df` when loaded."))
        l.addWidget(self.console)
        self.tab_w_console = w
        self.tabs.addTab(w, "Console")

    def _init_pdf_dock(self):
        self.pdf_dock = QDockWidget("PDF Viewer", self)
        self.pdf_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.pdf_dock.hide()
        # self.plotly_dock.hide()
        if PDF_AVAILABLE:
            self.pdf_doc = QPdfDocument(self)
            self.pdf_view = QPdfView(self.pdf_dock)
            self.pdf_view.setDocument(self.pdf_doc)
            # NEW: supaya multi-page dan bisa scroll
            try:
                self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
                self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
            except Exception:
                pass
            self.pdf_dock.setWidget(self.pdf_view)
        else:
            stub = QLabel("Qt PDF module not available. Install PyQt6 with QtPdf to enable.")
            stub.setWordWrap(True)
            self.pdf_dock.setWidget(stub)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.pdf_dock)
    
    def _init_plotly_dock(self):
        self.plotly_dock = QDockWidget("Interactive Plot", self)
        # self.pdf_dock.hide()
        self.plotly_dock.hide()
        if WEBVIEW_AVAILABLE:
            self.plotly_view = QWebEngineView(self.plotly_dock)
            self.plotly_dock.setWidget(self.plotly_view)
        else:
            stub = QLabel("Qt WebEngine tidak tersedia.\n"
                        "Saya akan menyimpan plot ke HTML dan membuka di browser.")
            stub.setWordWrap(True)
            self.plotly_dock.setWidget(stub)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.plotly_dock)

    def _init_error_dock(self):
        self.err_dock = QDockWidget("Errors", self)
        self.err_view = QTextEdit(); self.err_view.setReadOnly(True)
        self.err_dock.setWidget(self.err_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.err_dock)

# ---------- Helpers: column resolver & country series ----------
    # === NEW: helper ambil nama kolom by candidates (case-insensitive) ===
    def _get_col(self, *cands):
        """Case-insensitive exact match lalu contains-match."""
        if self.df is None:
            return None
        low = {str(c).lower(): c for c in self.df.columns}
        # exact
        for cand in cands:
            if cand and cand.lower() in low:
                return low[cand.lower()]
        # contains
        for c in self.df.columns:
            for cand in cands:
                if cand and cand.lower() in str(c).lower():
                    return c
        return None

    def _get_country_series(self) -> pd.Series | None:
        """
        Prefer kolom 'Country' (atau variasinya) sebagai string.
        Kalau tak ada, fallback ekstrak kasar dari 'Affiliations':
        ambil segmen terakhir setelah koma dari tiap afiliasi.
        """
        if self.df is None:
            return None
        col_country = self._get_col("Country", "Countries", "Country/Region")
        if col_country:
            ser = self.df[col_country].fillna("").astype(str).str.strip()
            # normalisasi entri multi (pisah ; , |)
            ser = ser.apply(lambda s: s.split(";")[0].strip() if ";" in s else s)
            return ser

        # Fallback dari Affiliations (kasar, tapi cukup untuk jalan dulu)
        col_aff = self._get_col("affiliations", "Affiliations", "Authors with affiliations", "Author Affiliations")
        if not col_aff:
            return None

        def _extract_main_country(text: str) -> str:
            if not isinstance(text, str):
                text = "" if pd.isna(text) else str(text)
            parts = [x.strip() for x in text.split(";") if x.strip()]
            if not parts:
                return "Unknown"
            # ambil negara dari afiliasi pertama: segmen terakhir setelah koma
            last_segment = parts[0].split(",")[-1].strip()
            return last_segment or "Unknown"

        return self.df[col_aff].fillna("").astype(str).apply(_extract_main_country)

    def _series_clean(self, s):
        return s.fillna("").astype(str)

    def _clean_source_series(self, s: pd.Series) -> pd.Series:
        """Bersihkan kolom Source: strip, gabungkan variasi kosong ke 'Unknown'."""
        ser = s.astype(str).fillna("").str.strip()

        # Anggap berbagai placeholder sebagai kosong
        placeholders = {"", "-", "n/a", "na", "none", "null"}
        ser = ser.apply(lambda x: "" if x.lower() in placeholders else x)

        # Jadikan kosong -> 'Unknown'
        ser = ser.replace(r"^\s*$", "Unknown", regex=True)

        # (Opsional) Normalisasi unicode & spasi ganda
        ser = ser.str.lower().str.replace(r"\s+", " ", regex=True)

        return ser

    def _split_semicolon(self, s):
        return [p.strip() for p in str(s).replace("|",";").replace(",", ";").split(";") if p.strip()]

    def _year_col(self):
        return self._get_col("Year", "Publication Year")

    def _doc_type(self):
        return self._get_col("Document Type", "doctype", "Type")

    def _source_col(self):
        # return self._get_col("Source title", "Journal", "Conference name", "Venue")
        return self._get_col(
            "Source title", "Source Title", "Publication Name",
            "Journal", "Conference name", "Conference Name", "Venue"
        )

    def _author_kw_col(self):
        return self._get_col("Author Keywords", "Author_Keywords", "Keywords")

    def _abstract_col(self):
        return self._get_col("Abstract", "Description")

    # Country helpers (stub aman sementara; biar tidak crash)
    def _count_by_country(self):
        return {}

    def _norm_country(self, name):  # biar kompatibel dengan peta nanti
        return name

    def _affil_col(self):
        # cari nama kolom afiliasi (case-insensitive, beberapa variasi umum)
        return self._get_col("affiliations", "Affiliations", "Authors with affiliations", "Author Affiliations")

    def _extract_countries_from_affil(self, text: str) -> str:
        """
        Ambil 'negara' dari setiap segmen afiliasi: pakai segmen terakhir setelah koma.
        Contoh segmen: 'Dept. X, Univ Y, Indonesia' -> ambil 'Indonesia'
        Pisah antar afiliasi pakai ';' (format Scopus umum).
        """
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        parts = [x.strip() for x in text.split(";") if x.strip()]
        countries = set()
        for part in parts:
            last_segment = part.split(",")[-1].strip()
            if len(last_segment) >= 3:
                countries.add(last_segment)
        return "; ".join(sorted(countries))

    def _detect_collab_scope(self, country_string: str) -> str:
        countries = [c.strip() for c in str(country_string).split(";") if c.strip()]
        if len(set(countries)) > 1:
            return "International"
        elif len(countries) == 1:
            return "Domestic"
        else:
            return "Unknown"

    def _build_collab_scope_table(self) -> pd.DataFrame | None:
        """
        Return DataFrame index=Main Country (Top 20), columns=['Domestic','International'] (Unknown di-drop).
        Nilai = jumlah dokumen.
        """
        if self.df is None:
            return None
        col_aff = self._affil_col()
        if not col_aff or col_aff not in self.df.columns:
            return None

        ser_aff = self.df[col_aff].fillna("").astype(str)

        # kolom intermediate seperti di notebook-mu
        extracted = ser_aff.apply(self._extract_countries_from_affil)
        scope = extracted.apply(self._detect_collab_scope)
        main_country = extracted.apply(lambda x: x.split(";")[0].strip() if x else "Unknown")

        tmp = pd.DataFrame({
            "Main Country": main_country,
            "Collab Scope": scope
        })
        scope_count = tmp.groupby(["Main Country", "Collab Scope"]).size().unstack(fill_value=0)

        # ambil hanya Domestic & International; buang kolom lain (Unknown) agar rapi
        for col in list(scope_count.columns):
            if col not in ("Domestic", "International"):
                scope_count.drop(columns=[col], inplace=True)

        # pastikan dua kolom selalu ada
        for needed in ("Domestic", "International"):
            if needed not in scope_count.columns:
                scope_count[needed] = 0

        # Top 20 by total
        totals = scope_count.sum(axis=1)
        if totals.empty:
            return None
        top_idx = totals.nlargest(20).index
        data = scope_count.loc[top_idx].copy()

        # Urutkan NA I K berdasarkan TOTAL (naik) agar barh enak dibaca dari bawah -> atas
        data["__total__"] = data.sum(axis=1)
        data = data.sort_values(by="__total__", ascending=False).drop(columns="__total__")
        return data

    def _install_table_context(self, view: QTableView):
        if view is None:
            return
        try:
            view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            view.customContextMenuRequested.connect(lambda p, v=view: self._on_table_context(v, p))
        except Exception:
            pass

    def _on_table_context(self, view: QTableView, point):
        idx = view.indexAt(point)
        menu = QMenu(view)

        act_attach = QAction("Attach PDF to Selected", self)
        act_view   = QAction("View PDF of Selected", self)

        act_attach.triggered.connect(self.attach_pdf_to_selected)
        act_view.triggered.connect(self.view_pdf_of_selected)

        if idx.isValid():
            # pilih baris yang diklik agar handler pakai baris ini
            view.selectRow(idx.row())
            menu.addAction(act_attach)
            menu.addAction(act_view)
        else:
            # kalau tidak klik di baris, disable
            act_attach.setEnabled(False); act_view.setEnabled(False)
            menu.addAction(act_attach); menu.addAction(act_view)

        menu.exec(view.viewport().mapToGlobal(point))

    def _parse_keyword_mapping(self, raw: str):
        mapping, words = {}, []
        if raw:
            for token in raw.split(','):
                token = token.strip()
                if not token:
                    continue
                if ':' in token:
                    k, c = token.split(':', 1)
                    mapping[k.strip()] = c.strip()
                    words.append(k.strip())
                else:
                    mapping[token] = '#c00000'
                    words.append(token)
        return mapping, words

    def apply_filter_from_cfg(self, cfg: dict):
        if self.df is None:
            QMessageBox.warning(self, APP_NAME, "Load a CSV first.")
            return

        df = self.df.copy()
        mapping, words = self._parse_keyword_mapping(cfg.get("raw",""))
        case = bool(cfg.get("case", False))
        cutoff = int(cfg.get("cutoff", 1))

        # target cols
        target_cols = [c for c in cfg.get("cols", []) if c in df.columns]
        if not target_cols:
            target_cols = self._guess_text_columns(df)

        # hit counter
        def count_hits_cell(cell: str) -> int:
            if pd.isna(cell):
                return 0
            s = str(cell)
            return sum(s.count(w) if case else s.lower().count(w.lower()) for w in words)

        if words and target_cols:
            hit_counts = df[target_cols].applymap(count_hits_cell).sum(axis=1)
        else:
            hit_counts = pd.Series([0]*len(df))

        df["__hits__"] = hit_counts
        kept = df[df["__hits__"] >= cutoff].copy()

        # optional: sort by hits (kalau kamu punya tombol self.btn_sort_hits)
        if hasattr(self, "btn_sort_hits") and self.btn_sort_hits.isChecked():
            kept = kept.sort_values("__hits__", ascending=False)

        # simpan & tampilkan
        self.filtered_df = kept.drop(columns=["__hits__"], errors="ignore")
        model = PandasModel(self.filtered_df)
        model.set_target_columns(target_cols)
        model.set_keywords_with_colors(mapping)
        self.filtered_table.setModel(model)

        # simpan kata kunci terakhir untuk fallback
        self._last_filter_words = list(words) if 'words' in locals() else list(mapping.keys())

        # connect selection -> detail panel + simpan base stats
        if self.filtered_table.selectionModel():
            self.filtered_table.selectionModel().selectionChanged.connect(self._on_filtered_row_selected)
        self._filter_stats_base = self.filter_stats.toPlainText()
        # pilih baris pertama agar detail langsung tampil
        try:
            if model.rowCount() > 0:
                self.filtered_table.selectRow(0)
        except Exception:
            pass
        self.model = model

        # stats text sederhana
        stats = [
            f"Keywords: {list(mapping.keys())}",
            f"Cutoff: {cutoff}",
            f"Target cols: {target_cols}",
            f"Total rows: {len(self.df):,}",
            f"Rows kept: {len(self.filtered_df):,}",
        ]
        self.filter_stats.setPlainText("\n".join(stats))
        self._filter_stats_base = self.filter_stats.toPlainText()  # <— simpan base

        # paging
        if hasattr(self, "page_size_spin") and hasattr(self, "page_index_spin"):
            self.page_size = self.page_size_spin.value()
            self.page_index = self.page_index_spin.value()
        self.refresh_page()

    def _resort_filtered_by_hits(self):
        """Dipanggil saat tombol sort toggle berubah; resort filtered_df dan refresh."""
        if self.filtered_df is None or '__hits__' not in self.filtered_df.columns:
            return
        desc = self.btn_sort_hits.isChecked()
        self.filtered_df = self.filtered_df.sort_values('__hits__', ascending=not desc)
        self.refresh_page()

    import html, re

    def _html_escape(self, s: str) -> str:
        return html.escape("" if pd.isna(s) else str(s))

    def _highlight_text(self, text: str, mapping: dict[str,str], case: bool) -> tuple[str, dict[str,int]]:
        """Return (html, counts_per_keyword)."""
        if not text:
            return "", {}
        counts = {k:0 for k in mapping.keys()}
        esc = self._html_escape(text)
        if not mapping:
            return esc, counts

        # siapkan regex alternatif dengan prioritas kata terpanjang dulu
        keys = sorted(mapping.keys(), key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(k) for k in keys) + ")"
        flags = 0 if case else re.IGNORECASE

        def repl(m):
            word = m.group(0)
            # cari key canonical yg match (case-insens)
            key = next((k for k in keys if (word == k if case else word.lower()==k.lower())), keys[0])
            counts[key] += 1
            color = mapping.get(key, "#c00000")
            return f"<b><span style='color:{color}'>{html.escape(word)}</span></b>"

        # Karena kita sudah escape HTML dulu (esc), perlu regex di text asli.
        # Untuk sederhana: jalankan regex di text asli lalu rebuild manual dengan span.
        # Alternatif: gunakan re.sub pada text asli lalu escape non-match — lebih singkat:
        s = str("" if pd.isna(text) else text)
        out = re.sub(pattern, repl, s, flags=flags)
        return html.escape(out).replace("&lt;b&gt;","<b>").replace("&lt;/b&gt;","</b>").replace(
            "&lt;span","<span").replace("span&gt;","span>"), counts

    def _update_details_for_row(self, cur_index):
        try:
            if not cur_index.isValid():
                self.details_view_filter.clear(); return
            model = self.filtered_table.model()
            if not isinstance(model, PandasModel):
                return
            dfrow = model.dataframe().iloc[cur_index.row()]
            # ambil kolom umum
            title_c = self._get_col(*TITLE_COL_CANDIDATES)
            abs_c   = self._get_col(*ABSTRACT_COL_CANDIDATES)
            ak_c    = self._author_kw_col()

            title = dfrow.get(title_c, "")
            abstr = dfrow.get(abs_c, "")
            akeys = dfrow.get(ak_c, "")

            # highlight
            mapping = getattr(self, "keyword_mapping", {})
            case = bool(getattr(self, "filter_cfg", {}).get("case", False))

            title_html, c1 = self._highlight_text(str(title), mapping, case)
            abs_html,   c2 = self._highlight_text(str(abstr), mapping, case)
            kw_html,    c3 = self._highlight_text(str(akeys), mapping, case)

            # frekuensi total per keyword
            freq = {}
            for d in (c1, c2, c3):
                for k, v in d.items():
                    freq[k] = freq.get(k, 0) + v

            # build details html
            lines = []
            lines.append(f"<h3 style='margin:0 0 6px 0'>{title_html or '(no title)'} </h3>")
            meta = []
            ycol = self._year_col(); scol = self._source_col(); dcol = self._get_col('DOI','doi')
            if ycol: meta.append(f"{self._html_escape(dfrow.get(ycol,''))}")
            if scol: meta.append(self._html_escape(dfrow.get(scol,'')))
            if dcol: meta.append(f"DOI: {self._html_escape(dfrow.get(dcol,''))}")
            if meta: lines.append("<div style='color:#666'>" + " • ".join(meta) + "</div><hr>")

            if ak_c:
                lines.append(f"<b>Author Keywords:</b> {kw_html or '-'}<br>")
            if abs_c:
                lines.append(f"<b>Abstract:</b><br><div style='text-align:justify'>{abs_html or '-'}</div>")

            # stats per keyword
            if mapping:
                lines.append("<hr><b>Matched terms & frequencies:</b><ul style='margin-top:4px'>")
                for k in mapping.keys():
                    if freq.get(k,0)>0:
                        lines.append(f"<li>{html.escape(k)} : {freq.get(k,0)}</li>")
                lines.append("</ul>")

            self.details_view_filter.setHtml("\n".join(lines))

            # tampilkan juga ringkasan freq di panel Filter Stats
            if mapping:
                stat_lines = ["Per-record term counts:"]
                for k in mapping.keys():
                    if freq.get(k,0)>0:
                        stat_lines.append(f"- {k}: {freq[k]}")
                self.filter_stats.append("\n" + "\n".join(stat_lines))

        except Exception as e:
            self._show_error(e)

    def save_filtered_records(self):
        if self.filtered_df is None or self.filtered_df.empty:
            QMessageBox.information(self, APP_NAME, "No filtered data to save.")
            return
        # opsi format
        path, _ = QFileDialog.getSaveFileName(
            self, "Save filtered records", str(Path.home()/ "filtered"),
            "CSV (*.csv);;Excel (*.xlsx);;Parquet (*.parquet)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".xlsx"):
                self.filtered_df.drop(columns=['__hits__'], errors='ignore').to_excel(path, index=False)
            elif path.lower().endswith(".parquet"):
                self.filtered_df.drop(columns=['__hits__'], errors='ignore').to_parquet(path, index=False)
            else:
                if not path.lower().endswith(".csv"): path += ".csv"
                self.filtered_df.drop(columns=['__hits__'], errors='ignore').to_csv(path, index=False)
            QMessageBox.information(self, APP_NAME, f"Saved: {path}")
        except Exception as e:
            self._show_error(e)
    def _toggle_show_only(self, on):
        if on or self.filtered_df is None:
            return
        # tampilkan full df (tanpa highlight)
        model = PandasModel(self.df.copy())
        self.filtered_table.setModel(model)
        self.refresh_page()
    # wire:
    # self.chk_show_only.toggled.connect(self._toggle_show_only)

    def _build_keyword_color_map(self):
        """Dari self.filter_cfg → dict keyword→color. Default biru bila warna tak ditentukan."""
        cfg = getattr(self, "filter_cfg", {}) or {}
        raw = cfg.get("raw", "")
        mapping, words = self._parse_keyword_mapping(raw)  # kamu sudah punya _parse_keyword_mapping
        # default biru untuk setiap kata yang tidak diberi warna
        color_map = {}
        for w in words:
            color_map[w.lower()] = mapping.get(w, "#1f77b4")  # biru default; boleh pakai nama warna umum juga
        return color_map, [w.lower() for w in words]

    def _highlight_keywords_html(self, text: str, kw_color_map: dict[str,str], bold=True) -> str:
        if not isinstance(text, str) or not kw_color_map:
            return "" if pd.isna(text) else str(text)
        # regex \bword\b untuk tiap keyword (case-insensitive)
        pattern = re.compile(r'\\b(' + "|".join(re.escape(k) for k in kw_color_map.keys()) + r')\\b', re.IGNORECASE)
        def replacer(m):
            word = m.group(0)
            color = kw_color_map.get(word.lower(), "#1f77b4")
            fw = 'bold' if bold else 'normal'
            return f"<span style='color:{color}; font-weight:{fw}'>{word}</span>"
        return pattern.sub(replacer, text)

    def _count_occ_in_text(self, text: str, words_lower: list[str]) -> dict[str,int]:
        counts = {w: 0 for w in words_lower}
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        lower = text.lower()
        for w in words_lower:
            counts[w] += len(re.findall(r'\\b' + re.escape(w) + r'\\b', lower))
        return counts

    def _join_counts(self, *dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                out[k] = out.get(k, 0) + int(v)
        return out

    def _on_filtered_row_selected(self, *_):
        try:
            sm = self.filtered_table.selectionModel() if hasattr(self, "filtered_table") else None
            if not sm or not sm.hasSelection():
                return
            row = sm.selectedRows()[0].row()
            m: PandasModel = self.filtered_table.model()
            series = m.dataframe().iloc[row]

            # siapkan highlight
            color_map, words_lower = self._build_keyword_color_map()

            # ambil nama kolom penting (supaya tahu mana yang di-highlight)
            title_c  = self._get_col("Title","Document Title","Article Title")
            abs_c    = self._abstract_col()
            ak_c     = self._author_kw_col()

            # rakit HTML semua kolom
            rows_html = []
            for col in series.index:
                val = series[col]
                col_name = html.escape(str(col))
                if col == title_c or col == abs_c or col == ak_c:
                    hv = self._highlight_keywords_html(val, color_map, bold=True)
                else:
                    hv = html.escape("" if pd.isna(val) else str(val))
                rows_html.append(f"<tr><td style='padding:2px 8px;vertical-align:top;white-space:nowrap;'><b>{col_name}</b></td>"
                                f"<td style='padding:2px 8px;'>{hv}</td></tr>")

            html_doc = (
                "<div style='font-family:Arial, sans-serif;'>"
                "<h3 style='margin:0 0 8px 0;'>Record Details</h3>"
                "<table border='0' cellspacing='0' cellpadding='0' style='border-collapse:collapse;'>"
                + "".join(rows_html) +
                "</table>"
                "</div>"
            )
            self.details_view_filter.setHtml(html_doc)

            # --- stats: frekuensi keyword pada row terpilih (Title + Abstract + Author Keywords)
            title = series.get(title_c, "")
            abstr = series.get(abs_c, "")
            akeys = series.get(ak_c, "")
            c1 = self._count_occ_in_text(title, words_lower)
            c2 = self._count_occ_in_text(abstr, words_lower)
            c3 = self._count_occ_in_text(akeys, words_lower)
            occ = self._join_counts(c1, c2, c3)

            lines = [(self._filter_stats_base or "").rstrip(), "", "Matches in selected row:"]
            for w in sorted(occ.keys()):
                lines.append(f" - {w}: {occ[w]}")
            self.filter_stats.setPlainText("\n".join(lines).strip())

        except Exception as e:
            self._show_error(e)

    # --- KEYWORD COLOR MAP & HIGHLIGHT (case-insensitive) ----------------------

    def _build_keyword_color_map(self):
        """
        Ambil keyword->color dari filter settings (self.filter_cfg).
        Jika warna tidak ditentukan, pakai biru '#1f77b4'.
        Kembalikan: (color_map{lower_kw:str->color}, words_lower[list[str]])
        """
        # sumber kata: dari filter dialog (mapping) ATAU dari filter terakhir
        raw = (getattr(self, "filter_cfg", {}) or {}).get("raw", "")  # contoh: "blockchain:red, traceability:gold"
        mapping, words = self._parse_keyword_mapping(raw) if hasattr(self, "_parse_keyword_mapping") else ({}, [])
        if not words and hasattr(self, "_last_filter_words"):
            words = list(self._last_filter_words)  # fallback

        # normalisasi lower
        color_map = {}
        for w in words:
            lw = str(w).strip().lower()
            if not lw:
                continue
            color = mapping.get(w, None)
            if color is None:
                color = "#1f77b4"  # default biru
            color_map[lw] = color
        return color_map, list(color_map.keys())

    def _highlight_keywords_html(self, text: str, kw_color_map: dict[str,str], bold=True) -> str:
        """Highlight kata kunci (case-insensitive). Menerima hex atau nama warna ('red', 'gold', dst.)."""
        if text is None:
            text = ""
        s = "" if pd.isna(text) else str(text)
        if not kw_color_map:
            return html.escape(s)

        # gunakan \bword\b, ignore case
        patt = re.compile(r'\b(' + "|".join(re.escape(k) for k in kw_color_map.keys()) + r')\b', re.IGNORECASE)
        def repl(m):
            w = m.group(0)
            color = kw_color_map.get(w.lower(), "#1f77b4")
            fw = 'bold' if bold else 'normal'
            return f"<span style='color:{color}; font-weight:{fw}'>{html.escape(w)}</span>"
        # untuk bagian NON match, tetap di-escape
        # caranya: split dan rakit kembali
        out = []
        last = 0
        for mm in patt.finditer(s):
            out.append(html.escape(s[last:mm.start()]))
            out.append(repl(mm))
            last = mm.end()
        out.append(html.escape(s[last:]))
        return "".join(out)

    def _count_occ_in_text(self, text: str, words_lower: list[str]) -> dict[str,int]:
        """Hitung kemunculan kata (case-insensitive) — cocok juga untuk 'Blockchain' atau 'blockchain-based'."""
        if text is None:
            text = ""
        s = "" if pd.isna(text) else str(text)
        lower = s.lower()
        out = {w: 0 for w in words_lower}
        for w in words_lower:
            # \b cocok di sekitar huruf/angka vs non-word (hyphen dihitung boundary)
            out[w] += len(re.findall(r'\b' + re.escape(w) + r'\b', lower))
        return out

    def _join_counts(self, *dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                out[k] = out.get(k, 0) + int(v)
        return out

    # ------------------------------ End Helpers --------------------------------

    # ------------------------------ Actions --------------------------------

    def open_csv(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open Scopus CSV", str(Path.home()), "CSV Files (*.csv)")
            if not path:
                return
            self.load_csv(Path(path))
        except Exception as e:
            self._show_error(e)

    def load_csv(self, path: Path):
        try:
            df = pd.read_csv(path)
            self.set_dataframe(df)
            self.current_csv_path = path
            self.ingest_info_label.setText(f"Loaded: {path.name} ({len(df):,} rows × {len(df.columns):,} cols)")
            self.ingest_info_label.setToolTip(str(path))     # path penuh di tooltip
            self.update_quick_stats()
            # Push into console
            self.console.push_variables({"df": self.df})
            # Saran kolom target (title/abstract/keywords terdeteksi otomatis)
            # self.columns_combo.setEditText(", ".join(self._guess_text_columns(self.df)))
            # Saran kolom target untuk filter:
            suggested_cols = self._guess_text_columns(self.df)

            # Jika UI lama (columns_combo) masih ada, isi; kalau tidak, simpan ke filter_cfg
            if hasattr(self, "columns_combo") and self.columns_combo is not None:
                self.columns_combo.setEditText(", ".join(suggested_cols))
            else:
                # siapkan default filter_cfg supaya dialog terisi otomatis
                cur = getattr(self, "filter_cfg", {})
                self.filter_cfg = {
                    "raw": cur.get("raw", ""),
                    "cutoff": cur.get("cutoff", 1),
                    "case": cur.get("case", False),
                    "cols": suggested_cols,
                }
        except Exception as e:
            self._show_error(e)

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
        self.model = PandasModel(df)
        self.table.setModel(self.model)
        self.filtered_table.setModel(self.model)  # start as same; will be replaced after filter
        # hubungkan seleksi tabel preview ke panel details
        self._connect_table_selection(self.table)
        self._connect_table_selection(self.filtered_table)
        self.filtered_df = df.copy()  # NEW
        self._install_table_context(self.table)
        if self.filtered_table is not None:
            self._install_table_context(self.filtered_table)

    def update_quick_stats(self):
        if self.df is None:
            self.stats_view.setPlainText("No data.")
            return
        try:
            n_rows, n_cols = self.df.shape
            na_counts = self.df.isna().sum()
            na_pct = (na_counts / max(n_rows, 1) * 100).round(2)
            lines = [f"Rows: {n_rows:,} | Cols: {n_cols:,}", "", "NA per column:"]
            for col in self.df.columns:
                lines.append(f"- {col}: {int(na_counts[col])} ({na_pct[col]}%)")
            self.stats_view.setPlainText("\n".join(lines))
        except Exception as e:
            self._show_error(e)

    def apply_filter(self):
        if self.df is None:
            QMessageBox.warning(self, APP_NAME, "Load a CSV first.")
            return

        # kalau belum ada filter_cfg (mis. user belum buka dialog), ambil dari UI lama bila ada,
        # atau pakai default sederhana
        cfg = getattr(self, "filter_cfg", None)
        if not cfg:
            raw = ""
            cutoff = 1
            case = False
            # fallback: kalau masih ada keywords_edit/cutoff_spin/case_sensitive_chk
            if hasattr(self, "keywords_edit"):
                raw = self.keywords_edit.text().strip()
            if hasattr(self, "cutoff_spin"):
                cutoff = self.cutoff_spin.value()
            if hasattr(self, "case_sensitive_chk"):
                case = self.case_sensitive_chk.isChecked()
            cfg = {
                "raw": raw,
                "cutoff": cutoff,
                "case": case,
                "cols": self._guess_text_columns(self.df),
            }
            self.filter_cfg = cfg

        self.apply_filter_from_cfg(cfg)

    def _get_col(self, *cands):
        """Ambil nama kolom pertama yang cocok (case-insensitive)."""
        if self.df is None: return None
        low = {c.lower(): c for c in self.df.columns}
        for c in cands:
            if c and c.lower() in low: return low[c.lower()]
        # fallback by contains
        for c in self.df.columns:
            for cand in cands:
                if cand and cand.lower() in str(c).lower():
                    return c
        return None

    def _series_clean(self, s):
        return s.fillna("").astype(str)

    def _split_semicolon(self, s):
        # Scopus kadang pakai ';' atau ',' tergantung field
        return [p.strip() for p in s.replace("|", ";").replace(",", ";").split(";") if p.strip()]

    def _count_by_country(self):
        # Cari kolom country; fallback via Affiliations
        c_country = self._get_col("Country", "Countries/Regions", "Correspondence Address")
        if c_country:
            ser = self._series_clean(self.df[c_country])
            countries = ser.apply(lambda t: self._split_semicolon(t))
        else:
            c_aff = self._get_col("Affiliations")
            if not c_aff: return {}
            ser = self._series_clean(self.df[c_aff])
            countries = ser.apply(lambda t: [seg.strip() for seg in t.split(",") if seg.strip()])  # kasar
        counts = {}
        for lst in countries:
            if isinstance(lst, str): lst = [lst]
            for c in lst:
                k = c.strip()
                if not k: continue
                counts[k] = counts.get(k, 0) + 1
        return counts

    def _norm_country(self, name):
        if not pycountry: return name
        try:
            # coba cocokkan lewat alpha_2/name
            c = pycountry.countries.get(name=name)
            if c: return c.name
            # fuzzy pendek
            for x in pycountry.countries:
                if name.lower() in x.name.lower():
                    return x.name
        except Exception:
            pass
        return name

    def _author_list(self, row):
        c_auth = self._get_col("Authors", "Author Names")
        if not c_auth: return []
        return self._split_semicolon(str(row.get(c_auth, "")))

    def _doc_type(self):
        return self._get_col("Document Type", "doctype", "Type")

    def _year_col(self):
        return self._get_col("Year", "Publication Year")

    def _source_col(self):
        return self._get_col("Source title", "Journal", "Conference name", "Venue")

    def _author_kw_col(self):
        return self._get_col("Author Keywords", "Author_Keywords", "Keywords")

    def _abstract_col(self):
        return self._get_col("Abstract", "Description")

    def _institution_type(self, aff_text):
        # Heuristik sederhana; bisa kita refine nanti dengan kamus
        t = str(aff_text).lower()
        flags_academic = ["university", "universitas", "institute of technology", "polytechnic", "school of", "faculty", "departemen", "department"]
        if any(k in t for k in flags_academic):
            return "Academic"
        # Non-academic: industry, government, ngo, hospital, etc.
        return "Non-Academic"

    def render_docs_per_year(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        try:
            # Try robust extraction of year: Scopus exports often have 'Year' or 'Publication Year'
            candidates = [c for c in self.df.columns if 'year' in c.lower()]
            if not candidates:
                QMessageBox.warning(self, APP_NAME, "No 'Year' column found.")
                return
            year_col = candidates[0]
            ser = pd.to_numeric(self.df[year_col], errors='coerce').dropna().astype(int)
            counts = ser.value_counts().sort_index()
            self.canvas.ax.clear()
            self.canvas.ax.bar(counts.index.astype(str), counts.values)
            self.canvas.ax.set_xlabel("Year")
            self.canvas.ax.set_ylabel("Documents")
            self.canvas.ax.set_title("Documents per Year")
            self.canvas.draw()
        except Exception as e:
            self._show_error(e)
    
    # ---------- Helpers: keywords ----------
    def _find_keyword_columns(self) -> tuple[str | None, str | None]:
        """
        Cari kolom author_keywords dan keywords (case-insensitive, beberapa varian umum).
        Return: (author_keywords_col, keywords_col)
        """
        if self.df is None: 
            return (None, None)
        cols = {str(c).lower(): c for c in self.df.columns}

        def pick(cands):
            for k in cands:
                lk = k.lower()
                if lk in cols:
                    return cols[lk]
            # fallback: contains
            for lc, orig in cols.items():
                if any(k.lower() in lc for k in cands):
                    return orig
            return None

        author_kw = pick(["author_keywords", "author keywords", "author_keywords;"])
        # beberapa ekspor Scopus kadang "Index Keywords" dipakai sebagai 'keywords' umum
        kw = pick(["keywords", "index keywords"])
        return (author_kw, kw)

    def _build_keyword_freq(self) -> pd.DataFrame | None:
        """
        Gabungkan author_keywords + keywords -> split ';' -> lowercase/strip -> count.
        Return DataFrame: columns=['Keyword','Frequency'] urut desc.
        """
        if self.df is None:
            return None
        a_col, k_col = self._find_keyword_columns()
        if not a_col and not k_col:
            return None

        parts = []
        if a_col: parts.append(self.df[a_col].dropna().astype(str))
        if k_col: parts.append(self.df[k_col].dropna().astype(str))
        if not parts:
            return None

        combined = pd.concat(parts, ignore_index=True)
        # split & flatten
        all_keywords = []
        for s in combined:
            for kw in s.split(';'):
                kw = kw.strip().lower()
                if kw:
                    all_keywords.append(kw)
        if not all_keywords:
            return None

        freq = pd.Series(all_keywords).value_counts().reset_index()
        freq.columns = ['Keyword', 'Frequency']
        return freq


    def plot_world_map_from_shapefile(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        if gpd is None:
            QMessageBox.information(self, APP_NAME, "GeoPandas not installed.\nInstall: pip install geopandas pyproj shapely fiona")
            return
        try:
            # --- Preferred shapefile locations (static first) ---
            # 1) project-relative: <repo>/110m/ne_110m_admin_0_countries.shp
            default_shp = (Path(__file__).parent / "110m" / "ne_110m_admin_0_countries.shp").resolve()

            # 2) cache: ~/.slrdesk/cache/ne_110m_admin_0_countries.shp
            cache_shp = (CACHE_DIR / "ne_110m_admin_0_countries.shp").resolve()

            # Decide shp_path
            if default_shp.exists():
                shp_path = str(default_shp)
            elif cache_shp.exists():
                shp_path = str(cache_shp)
            else:
                # 3) fallback: open file dialog
                shp_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select world shapefile (.shp)",
                    str(Path.home()),
                    "Shapefile (*.shp)"
                )
                if not shp_path:
                    QMessageBox.information(self, APP_NAME, "Shapefile not selected.")
                    return
            
            # After user-picked shp_path and before reading:
            try:
                # copy .shp + teman-temannya (.shx, .dbf, .prj, .cpg, dsb.) ke CACHE_DIR
                src_dir = Path(shp_path).parent
                stem = Path(shp_path).stem  # ne_110m_admin_0_countries
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    src = src_dir / f"{stem}{ext}"
                    if src.exists():
                        dst = CACHE_DIR / src.name
                        if not dst.exists():
                            dst.write_bytes(src.read_bytes())
            except Exception as e:
                self._show_error(e)  # non-fatal

            # Baca shapefile
            world = gpd.read_file(shp_path)

            # Siapkan distribusi dokumen per negara
            ser_country = self._get_country_series()
            self.canvas.ax.clear()
            if ser_country is None or ser_country.eq("").all():
                self.canvas.ax.text(0.5, 0.5, "No country information found", ha="center", va="center")
                self.canvas.draw()
                return

            country_dist = ser_country.value_counts().reset_index()
            country_dist.columns = ["Country", "Document Count"]
            country_dist["Country"] = country_dist["Country"].str.strip()

            # Cocokkan kolom nama negara di shapefile
            # Natural Earth 110m biasanya punya kolom ADMIN (nama negara)
            name_col = None
            for cand in ("ADMIN", "admin", "NAME", "name"):
                if cand in world.columns:
                    name_col = cand
                    break
            if not name_col:
                self.canvas.ax.text(0.5, 0.5, "Shapefile needs a 'ADMIN'/'NAME' country column", ha="center", va="center")
                self.canvas.draw()
                return

            world = world.copy()
            world["name"] = world[name_col].astype(str).str.strip()

            # Merge
            merged = world.merge(country_dist, how="left", left_on="name", right_on="Country")

            # Buang colorbar axes lama (kalau ada) agar tidak menumpuk
            for a in list(self.canvas.figure.axes):
                if a is not self.canvas.ax:
                    try:
                        a.remove()
                    except Exception:
                        pass

            # Plot
            # Bersihkan seluruh figure (termasuk colorbar-colorbar lama)
            self.canvas.figure.clf()
            self.canvas.ax = self.canvas.figure.add_subplot(111)
            ax = self.canvas.ax
            merged.plot(
                column="Document Count",
                cmap="viridis",
                linewidth=0.8,
                ax=ax,
                edgecolor="0.8",
                legend=True,
                missing_kwds={"color": "lightgrey", "label": "No data"}
            )
            ax.set_title("Distribusi Dokumen per Negara (World Map, shapefile)", fontsize=12)
            ax.axis("off")
            self.canvas.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self._show_error(e)
            self.canvas.ax.clear()
            self.canvas.ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            self.canvas.draw()

    def _update_paging_ui(self):
        total = len(self.filtered_df) if isinstance(self.filtered_df, pd.DataFrame) else 0
        ps = max(1, int(getattr(self, "page_size", self.page_size_spin.value() if hasattr(self,"page_size_spin") else 50)))
        tot_pages = max(1, (total + ps - 1) // ps)
        cur = min(max(0, int(self.page_index_spin.value() if hasattr(self,"page_index_spin") else 0)), tot_pages-1)
        self.page_index_spin.blockSignals(True)
        self.page_index_spin.setMaximum(max(0, tot_pages-1))
        self.page_index_spin.setValue(cur)
        self.page_index_spin.blockSignals(False)
        self.lbl_pageinfo.setText(f"Page {cur+1} / {tot_pages}")
        self.btn_first.setEnabled(cur > 0)
        self.btn_last.setEnabled(cur < tot_pages-1)

    def first_page(self):
        if hasattr(self, "page_index_spin"):
            self.page_index_spin.setValue(0)
            self.refresh_page()

    def last_page(self):
        if hasattr(self, "page_index_spin") and isinstance(self.filtered_df, pd.DataFrame):
            ps = max(1, self.page_size_spin.value())
            tot_pages = max(1, (len(self.filtered_df) + ps - 1)//ps)
            self.page_index_spin.setValue(tot_pages-1)
            self.refresh_page()
    
    # === NEW: klasifikasi tipe institusi (adaptasi dari kode Jupyter kamu) ===
    def _classify_institution_types(self, affil_str: str):
        if not isinstance(affil_str, str):
            affil_str = "" if pd.isna(affil_str) else str(affil_str)
        affils = [a.strip().lower() for a in affil_str.split(";") if a.strip()]
        types = set()
        for affil in affils:
            if any(k in affil for k in ["univ", "college", "faculty", "school of"]):
                types.add("Academic")
            elif any(k in affil for k in ["research", "academy", "laboratory", "institute"]):
                types.add("Research")
            elif any(k in affil for k in ["corp", "co.", "inc", "ltd", "tech", "company"]):
                types.add("Industry")
            elif any(k in affil for k in ["gov", "ministry", "bureau", "department"]):
                types.add("Government")
            else:
                types.add("Unknown")
        return types

    def _categorize_institution_combination(self, affil_str: str):
        types = self._classify_institution_types(affil_str)
        # buang Unknown biar sesuai logika kamu
        types = types - {"Unknown"}

        # hitung jumlah afiliasi unik (untuk 'Same' vs 'Different Academic')
        unique_affil_count = len(set([a.strip().lower() for a in str(affil_str).split(";") if a.strip()]))

        if types == {"Academic"}:
            return "Same Academic/University" if unique_affil_count <= 1 else "Different Academic"
        elif types == {"Research"}:
            return "Research only"
        elif types == {"Industry"}:
            return "Industry only"
        elif types == {"Government"}:
            return "Government only"
        elif "Academic" in types and "Research" in types and len(types) == 2:
            return "Academic and Research"
        elif "Academic" in types and "Industry" in types and len(types) == 2:
            return "Academic and Industry"
        elif "Academic" in types and "Government" in types and len(types) == 2:
            return "Academic and Government"
        else:
            return "Mixed Collaboration"

    # === NEW: plotting function (Matplotlib-only) ===
    def plot_institution_collab_types(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        try:
            # cari kolom afiliasi: coba 'affiliations' dulu, lalu fallback variasi umum
            col_aff = self._get_col("affiliations", "Affiliations", "Authors with affiliations", "Author Affiliations")
            if not col_aff:
                self.canvas.ax.clear()
                self.canvas.ax.text(0.5, 0.5, "Affiliations column not found", ha="center", va="center")
                self.canvas.draw()
                return

            ser_aff = self.df[col_aff].fillna("").astype(str)

            # kategorisasi per baris
            cats = ser_aff.apply(self._categorize_institution_combination)
            counts = cats.value_counts().sort_values(ascending=True)  # ascending biar barh dari bawah ke atas

            # render
            # Bersihkan seluruh figure (termasuk colorbar-colorbar lama)
            self.canvas.figure.clf()
            self.canvas.ax = self.canvas.figure.add_subplot(111)
            self.canvas.ax.clear()
            self.canvas.ax.barh(counts.index.tolist(), counts.values.tolist())
            self.canvas.ax.set_title("Distribution of Institutional Collaboration Types")
            self.canvas.ax.set_xlabel("Number of Papers")
            self.canvas.ax.set_ylabel("Institutional Composition")
            self.canvas.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self._show_error(e)
            # tampilkan di kanvas juga biar user lihat error
            self.canvas.ax.clear()
            self.canvas.ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            self.canvas.draw()

    def cache_dataframe(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Nothing to cache.")
            return
        try:
            # Save as Parquet and Pickle for speed; JSON is heavy for large tables
            base = CACHE_DIR / (self.current_csv_path.stem if self.current_csv_path else "dataset")
            pq = base.with_suffix('.parquet')
            pkl = base.with_suffix('.pkl')
            self.df.to_parquet(pq, index=False)
            self.df.to_pickle(pkl)
            QMessageBox.information(self, APP_NAME, f"Cached to:\n{pq}\n{pkl}")
        except Exception as e:
            self._show_error(e)

    def export_filtered(self):
        try:
            # Prioritas: seluruh hasil filter jika ada
            df_to_save = None

            view_model = self.filtered_table.model() if hasattr(self, "filtered_table") else None
            if isinstance(view_model, PandasModel):
                # ambil all pages dari model
                try:
                    df_to_save = view_model.full_dataframe().copy()
                except Exception:
                    pass

            # fallback: variabel penampung hasil filter
            if df_to_save is None and getattr(self, "filtered_df", None) is not None:
                df_to_save = self.filtered_df.copy()

            # fallback terakhir: seluruh dataset
            if df_to_save is None and self.df is not None:
                df_to_save = self.df.copy()

            if df_to_save is None or df_to_save.empty:
                QMessageBox.information(self, APP_NAME, "Tidak ada data untuk diekspor.")
                return

            path, _ = QFileDialog.getSaveFileName(
                self, "Save Filtered CSV",
                str(Path.home() / "filtered.csv"),
                "CSV (*.csv)"
            )
            if not path:
                return

            df_to_save.to_csv(path, index=False)
            QMessageBox.information(self, APP_NAME, f"Saved: {path}")
        except Exception as e:
            self._show_error(e)

    # ------------------------------- PDF API -------------------------------

    def open_pdf(self, path: Path):
        if not PDF_AVAILABLE:
            QMessageBox.warning(self, APP_NAME, "PDF module not available.")
            return
        try:
            # Pastikan tampak hanya di Filter & Highlight
            self.tabs.setCurrentWidget(self.tab_w_filter)
            self.pdf_dock.setVisible(True)
            self.pdf_dock.raise_()
            self.pdf_doc.load(str(path))
            try:
                self.pdf_view.setPage(0)  # mulai dari halaman pertama
            except Exception:
                pass
        except Exception as e:
            self._show_error(e)

    # ------------------------------ Errors ---------------------------------

    def _show_error(self, e: Exception):
        tb = self.logger.log_exception(e)
        self.err_view.append(tb)
        QMessageBox.critical(self, APP_NAME, f"Error:\n{e}")

    def closeEvent(self, ev):
        try:
            self.console.shutdown()
        except Exception:
            pass
        super().closeEvent(ev)

#------------------------------ Paging ------------------------------
    def _guess_text_columns(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in df.columns]
        lower_map = {c.lower(): c for c in cols}
        picked = []
        for k in TITLE_COL_CANDIDATES + ABSTRACT_COL_CANDIDATES + KEYWORDS_COL_CANDIDATES:
            if k in lower_map:
                picked.append(lower_map[k])
        return picked or df.select_dtypes(include=['object']).columns.tolist()

    def refresh_page(self):
        if not isinstance(self.filtered_table.model(), PandasModel):
            return
        m: PandasModel = self.filtered_table.model()
        total = len(self.filtered_df) if self.filtered_df is not None else len(m.dataframe())
        self.page_size = self.page_size_spin.value()
        self.page_index = max(0, self.page_index_spin.value())
        start = self.page_index * self.page_size
        if start >= total and total > 0:
            self.page_index = 0
            self.page_index_spin.setValue(0)
            start = 0
        m._full_df = self.filtered_df if self.filtered_df is not None else m._full_df
        m.page(start, self.page_size)
        self._update_paging_ui()
        try:
            if m.rowCount() > 0:
                self.filtered_table.selectRow(0)
        except Exception:
            pass
        self._update_paging_ui()
        # auto-select row 0 setiap ganti halaman -> supaya detail kanan update
        try:
            m: PandasModel = self.filtered_table.model()
            if m and m.rowCount() > 0:
                self.filtered_table.selectRow(0)
        except Exception:
            pass

    def next_page(self):
        self.page_index_spin.setValue(self.page_index_spin.value() + 1)
        self.refresh_page()

    def prev_page(self):
        if self.page_index_spin.value() > 0:
            self.page_index_spin.setValue(self.page_index_spin.value() - 1)
            self.refresh_page()

    def import_csv_to_db(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load a CSV first.")
            return
        try:
            df = self.df
            cols = {c.lower(): c for c in df.columns}
            def pick(cands):
                for k in cands:
                    if k in cols:
                        return cols[k]
                return None
            title_c = pick(TITLE_COL_CANDIDATES)
            abs_c = pick(ABSTRACT_COL_CANDIDATES)
            kw_c = pick(KEYWORDS_COL_CANDIDATES)
            year_c = next((cols[c] for c in cols if 'year' in c), None)
            doi_c = next((cols[c] for c in cols if 'doi' in c), None)
            venue_c = next((cols[c] for c in cols if 'source title' in c or 'venue' in c), None)

            imported = 0
            for _, row in df.iterrows():
                rec = {
                    'title': row.get(title_c) if title_c else None,
                    'abstract': row.get(abs_c) if abs_c else None,
                    'year': pd.to_numeric(row.get(year_c), errors='coerce') if year_c else None,
                    'doi': str(row.get(doi_c)).strip() if doi_c and not pd.isna(row.get(doi_c)) else None,
                    'venue': row.get(venue_c) if venue_c else None,
                    'keywords': row.get(kw_c) if kw_c else None,
                    'pdf_path': None,
                }
                self.db.upsert_article(rec)
                imported += 1
            QMessageBox.information(self, APP_NAME, f"Imported {imported:,} records into {DB_PATH}")
        except Exception as e:
            self._show_error(e)

    def attach_pdf_to_selected(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load a CSV first.")
            return
        try:
            view = self.filtered_table if self.filtered_table.model() else self.table
            idx = view.currentIndex()
            if not idx.isValid():
                QMessageBox.information(self, APP_NAME, "Pilih baris terlebih dahulu.")
                return
            row = idx.row()
            model = view.model()
            if not isinstance(model, PandasModel):
                QMessageBox.information(self, APP_NAME, "Model tidak dikenal.")
                return
            dfrow = model.dataframe().iloc[row]
            # Identify by DOI or fallback to Title
            cols = {c.lower(): c for c in dfrow.index}
            doi = None
            for k in cols:
                if 'doi' in k:
                    val = dfrow[cols[k]]
                    doi = None if pd.isna(val) else str(val)
                    break
            title = None
            for k in TITLE_COL_CANDIDATES:
                if k in cols:
                    title = dfrow[cols[k]]
                    break
            pdf_path, _ = QFileDialog.getOpenFileName(self, "Select PDF", str(Path.home()), "PDF Files (*.pdf)")
            if not pdf_path:
                return
            ok = self.db.attach_pdf({"doi": doi, "title": title}, pdf_path)
            if ok and PDF_AVAILABLE:
                self.open_pdf(Path(pdf_path))
            QMessageBox.information(self, APP_NAME, "PDF attached to record in DB." if ok else "Record not found in DB; import CSV → DB dulu.")
        except Exception as e:
            self._show_error(e)

    def view_pdf_of_selected(self):
        try:
            view = self.filtered_table if self.filtered_table.model() else self.table
            idx = view.currentIndex()
            if not idx.isValid():
                QMessageBox.information(self, APP_NAME, "Pilih baris dulu.")
                return
            dfrow = view.model().dataframe().iloc[idx.row()]
            # Buka jika ada kolom pdf_path di DF (jika kamu pernah merge dari DB)
            for c in dfrow.index:
                if str(c).lower() == 'pdf_path' and not pd.isna(dfrow[c]):
                    self.open_pdf(Path(str(dfrow[c])))
                    return
            QMessageBox.information(self, APP_NAME, "Tidak ada kolom pdf_path di dataset ini. Gunakan Attach PDF untuk menyimpan ke DB.")
        except Exception as e:
            self._show_error(e)

    def render_selected_chart(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        sel = self.chart_selector.currentText().split(".", 1)[0].strip()
        try:
            idx = int(sel)
        except Exception:
            idx = -1

        self.canvas.ax.clear()
        try:
            if   idx == 1:  self._plot_docs_per_year()
            elif idx == 2:  self._plot_doc_type_pie()
            elif idx == 3:  self._plot_year_by_doctype()
            elif idx == 4:  self._plot_trend_year_by_doctype()
            elif idx == 5:  self._plot_top_sources()
            elif idx == 6:  self._wordcloud_author_keywords()
            elif idx == 7:  self._wordcloud_abstract()
            elif idx == 8:  self._author_collab_network()
            elif idx == 9:  self.plot_treemap_keywords()
            elif idx == 10: self._plot_institution_type_dist()
            elif idx == 11: self._plot_docs_by_country_top20()
            elif idx == 12: self.plot_plotly_choropleth()
            elif idx == 13: self.plot_world_map_from_shapefile()
            elif idx == 14: self._plot_collab_scope_by_country_stacked()
            elif idx == 15: self._plot_collab_scope_by_country_percent()
            elif idx == 16: self.plot_institution_collab_types()   # <— pakai logika notebook-mu
            else:
                self.canvas.ax.text(0.5,0.5,"Unknown selection", ha="center", va="center")
        except Exception as e:
            self._show_error(e)
            self.canvas.ax.text(0.5,0.5,f"Error: {e}", ha="center", va="center")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def export_current_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save chart as PNG", str(Path.home()/ "chart.png"), "PNG (*.png)")
        if not path: return
        try:
            self.canvas.figure.savefig(path, dpi=150, bbox_inches="tight")
            QMessageBox.information(self, APP_NAME, f"Saved: {path}")
        except Exception as e:
            self._show_error(e)

    def plot_plotly_choropleth(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        if px is None or pycountry is None:
            QMessageBox.information(
                self, APP_NAME,
                "Plotly/pycountry belum terpasang.\nInstall:\n  pip install plotly pycountry"
            )
            return
        try:
            ser = self._get_country_series()
            if ser is None or ser.eq("").all():
                QMessageBox.information(self, APP_NAME, "Kolom Country tidak ditemukan / kosong.")
                return

            country_dist = ser.value_counts().reset_index()
            country_dist.columns = ['Country', 'Document Count']

            def get_country_code(name):
                try:
                    return pycountry.countries.lookup(name).alpha_3
                except Exception:
                    return None

            country_dist['iso_alpha'] = country_dist['Country'].apply(get_country_code)
            country_dist = country_dist.dropna(subset=['iso_alpha'])

            if country_dist.empty:
                QMessageBox.information(self, APP_NAME, "Tidak ada negara yang terpetakan ke ISO alpha-3.")
                return

            fig = px.choropleth(
                country_dist,
                locations='iso_alpha',
                color='Document Count',
                hover_name='Country',
                color_continuous_scale='viridis',
                title='Distribusi Dokumen per Negara (World Map Choropleth)'
            )
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True),
                width=1000,
                height=700
            )

            # Tampilkan
            html = fig.to_html(include_plotlyjs='cdn', full_html=True)
            html_path = CACHE_DIR / "plotly_choropleth.html"
            html_path.write_text(html, encoding="utf-8")

            # Tampilkan dock hanya di Bibliometrics
            self.tabs.setCurrentWidget(self.tab_w_biblio)
            self.plotly_dock.setVisible(True)
            self.plotly_dock.raise_()

            if WEBVIEW_AVAILABLE:
                # Render langsung di dock
                self.plotly_view.setHtml(html)  # Html inline; alternatif: load(QUrl.fromLocalFile(...))
                self.plotly_dock.raise_()
            else:
                # Fallback buka di browser
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))

        except Exception as e:
            self._show_error(e)

    def _on_tab_changed(self, idx: int):
        try:
            # aman kalau tab belum dibuat
            idx_filter = self.tabs.indexOf(self.tab_w_filter) if hasattr(self, "tab_w_filter") else -1
            idx_ingest = self.tabs.indexOf(self.tab_w_ingest) if hasattr(self, "tab_w_ingest") else -1
            idx_biblio = self.tabs.indexOf(self.tab_w_biblio)

            show_pdf = (idx == idx_filter)
            show_plotly = (idx == idx_biblio)

            if hasattr(self, "pdf_dock"):
                self.pdf_dock.setVisible(show_pdf)
                if show_pdf:
                    self.pdf_dock.raise_()

            if hasattr(self, "plotly_dock"):
                self.plotly_dock.setVisible(show_plotly)
                if show_plotly:
                    self.plotly_dock.raise_()
        except Exception as e:
            self._show_error(e)
    
    def open_filter_dialog(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load a CSV first.")
            return
        cur = getattr(self, "filter_cfg", {"raw":"", "cutoff":1, "case":False, "cols": self._guess_text_columns(self.df)})
        dlg = FilterDialog(self, list(self.df.columns), cur)
        if dlg.exec():
            self.filter_cfg = dlg.value()
            # jalankan apply_filter dengan cfg ini
            self.apply_filter_from_cfg(self.filter_cfg)

    def _reset_canvas(self, top_pad=0.86):
        """Reset figure/axes dengan layout stabil untuk Matplotlib+Qt."""
        fig = self.canvas.figure
        # bersihkan semua artists/legend/colorbar
        fig.clf()
        # Matplotlib 3.7+: pastikan constrained_layout nonaktif jika sebelumnya pernah aktif
        try:
            fig.set_constrained_layout(False)
        except Exception:
            pass
        ax = fig.add_subplot(111)
        self.canvas.ax = ax
        # sisakan ruang untuk judul/legend di atas
        fig.subplots_adjust(top=top_pad)
        return ax

    # ------------------ Various Bibliometric Plots ------------------
    def _load_world_gdf(self):
        if gpd is None:
            return None
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
        zip_path = CACHE_DIR / "ne_110m_admin_0_countries.zip"
        try:
            if not zip_path.exists():
                try:
                    import urllib.request
                    zip_path.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(url, str(zip_path))
                except Exception:
                    # jika offline, beri tahu user untuk menaruh file secara manual
                    QMessageBox.information(self, APP_NAME,
                        f"Tidak bisa mengunduh Natural Earth.\n"
                        f"Unduh manual ZIP ini:\n{url}\n"
                        f"lalu simpan ke:\n{zip_path}")
                    return None
            # baca shapefile dari dalam ZIP
            return gpd.read_file(f"zip://{zip_path}!ne_110m_admin_0_countries.shp")
        except Exception as e:
            self._show_error(e)
            return None

    # 1. Distribution of Publications per Year
    def _plot_docs_per_year(self):
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        y = self._year_col()
        if not y: 
            self.canvas.ax.text(0.5,0.5,"Year column not found", ha='center'); return
        ser = pd.to_numeric(self.df[y], errors='coerce').dropna().astype(int)
        counts = ser.value_counts().sort_index()
        self.canvas.ax.bar(counts.index.astype(str), counts.values)
        self.canvas.ax.set_xlabel("Year"); 
        self.canvas.ax.set_ylabel("Documents")
        self.canvas.ax.set_title("Publications per Year")
        self.canvas.ax.tick_params(axis='x', rotation=45)

    # 2. Document Type Distribution (Pie Chart)
    def _plot_doc_type_pie(self):
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        c = self._doc_type()
        if not c:
            self.canvas.ax.text(0.5,0.5,"Document Type column not found", ha='center'); return
        ser = self._series_clean(self.df[c])
        counts = ser.value_counts().head(10)
        self.canvas.ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        self.canvas.ax.set_title("Document Type Distribution (Top 10)")

    # 3. Publications per Year by Document Type (clustered bars)
    def _plot_year_by_doctype(self):
        y = self._year_col(); d = self._doc_type()
        if not y or not d:
            self.canvas.ax.text(0.5,0.5,"Year or Document Type missing", ha='center'); self.canvas.draw(); return

        dfp = self.df.copy()
        dfp[y] = pd.to_numeric(dfp[y], errors='coerce').astype('Int64')

        piv = (dfp.dropna(subset=[y, d])
                .groupby([y, d])
                .size()
                .unstack(fill_value=0)
                .sort_index())

        ax = self._reset_canvas(top_pad=0.83)
        if piv.empty:
            ax.text(0.5,0.5,"No data after grouping", ha='center'); self.canvas.draw(); return

        import numpy as np
        years   = piv.index.to_numpy()
        n_years = len(years)
        n_types = len(piv.columns)

        # skala lebar bar proporsional jumlah tipe (tanpa mengubah ukuran figure)
        x = np.arange(n_years)
        group_width = 0.84
        bar_w = max(0.08, min(0.28, group_width / max(1, n_types)))

        # offset sehingga setiap grup terpusat di x
        left_edge = x - (group_width/2) + bar_w/2
        for i, col in enumerate(piv.columns):
            ax.bar(left_edge + i*bar_w, piv[col].to_numpy(), width=bar_w, label=str(col))

        # xtick—batasi supaya tidak terlalu rapat
        max_xticks = 12
        step = max(1, int(np.ceil(n_years / max_xticks)))
        shown_idx = np.arange(0, n_years, step)

        ax.set_xticks(x[shown_idx], labels=[str(int(y)) for y in years[shown_idx]], rotation=45, ha='right')
        ax.set_ylabel("Documents")
        ax.set_title("Publications per Year by Document Type", pad=20)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

        # legend di atas, tidak menutupi judul
        ncol = min(n_types, 4) if n_types > 1 else 1
        ax.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False)

        self.canvas.figure.tight_layout(rect=[0,0,1,0.94])  # sisakan ruang atas utk legend
        self.canvas.draw()

    # 4. Publication Trends per Year by Document Type (lines)
    def _plot_trend_year_by_doctype(self):
        y = self._year_col(); d = self._doc_type()
        if not y or not d:
            self.canvas.ax.text(0.5,0.5,"Year or Document Type missing", ha='center'); self.canvas.draw(); return

        dfp = self.df.copy()
        dfp[y] = pd.to_numeric(dfp[y], errors='coerce').astype('Int64')

        piv = (dfp.dropna(subset=[y, d])
                .groupby([y, d])
                .size()
                .unstack(fill_value=0)
                .sort_index())

        ax = self._reset_canvas(top_pad=0.83)
        if piv.empty:
            ax.text(0.5,0.5,"No data after grouping", ha='center'); self.canvas.draw(); return

        import numpy as np
        years   = piv.index.to_numpy()
        n_years = len(years)
        n_types = len(piv.columns)

        # plot semua tipe
        for col in piv.columns:
            ax.plot(years, piv[col].to_numpy(), marker='o', linewidth=1.8, label=str(col))

        # xtick: tampilkan secukupnya
        max_xticks = 12
        step = max(1, int(np.ceil(n_years / max_xticks)))
        shown_years = years[::step]
        ax.set_xticks(shown_years)
        ax.set_xticklabels([str(int(y)) for y in shown_years], rotation=45, ha='right')

        ax.set_xlabel("Year")
        ax.set_ylabel("Documents")
        ax.set_title("Publication Trends per Year by Document Type", pad=20)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

        ax.legend(ncol=min(n_types, 4), loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False)

        self.canvas.figure.tight_layout(rect=[0,0,1,0.94])
        self.canvas.draw()

    # 5. Top 10 Sources (Journals/Conferences)
    def _plot_top_sources(self):
        s = self._source_col()
        if not s:
            self.canvas.ax.text(0.5,0.5,"Source title column not found", ha='center'); return

        # Bersihkan & hitung
        ser = self._clean_source_series(self.df[s])
        counts = ser.value_counts()

        if counts.empty:
            self.canvas.ax.text(0.5,0.5,"No source titles", ha='center'); return

        # Ambil top 10 (urut naik untuk barh)
        counts = counts.head(10).sort_values()

        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        bars = self.canvas.ax.barh(counts.index.tolist(), counts.values.tolist())

        # Label angka di ujung bar
        for bar, val in zip(bars, counts.values):
            self.canvas.ax.text(bar.get_width() + max(counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                                str(int(val)), va='center', ha='left', fontsize=9)

        self.canvas.ax.set_xlabel("Documents")
        self.canvas.ax.set_title("Top 10 Sources")
        self.canvas.ax.grid(axis='x', linestyle=':', alpha=0.3)

        # Tambah margin kiri kalau label panjang
        self.canvas.figure.subplots_adjust(left=0.35)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # 6. WordCloud Author Keywords
    def _wordcloud_author_keywords(self):
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        if WordCloud is None:
            self.canvas.ax.text(0.5,0.5,"Install 'wordcloud' to enable", ha='center'); return
        k = self._author_kw_col()
        if not k:
            self.canvas.ax.text(0.5,0.5,"Author Keywords column not found", ha='center'); return
        text = " ".join(sum([self._split_semicolon(s) for s in self._series_clean(self.df[k])], []))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        self.canvas.ax.imshow(wc); self.canvas.ax.axis("off"); self.canvas.ax.set_title("WordCloud - Author Keywords")

    # 7. WordCloud Abstract
    def _wordcloud_abstract(self):
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        if WordCloud is None:
            self.canvas.ax.text(0.5,0.5,"Install 'wordcloud' to enable", ha='center'); return
        a = self._abstract_col()
        if not a:
            self.canvas.ax.text(0.5,0.5,"Abstract column not found", ha='center'); return
        text = " ".join(self._series_clean(self.df[a]).tolist())
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        self.canvas.ax.imshow(wc); self.canvas.ax.axis("off"); self.canvas.ax.set_title("WordCloud - Abstract")

    # 8. Author Collaboration Network (Top 30)
    def _author_collab_network(self):
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        if nx is None:
            self.canvas.ax.text(0.5,0.5,"Install 'networkx' to enable", ha='center'); return
        # bangun graf co-authorship
        G = nx.Graph()
        for _, row in self.df.iterrows():
            authors = self._author_list(row)
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a, b = authors[i], authors[j]
                    if not a or not b: continue
                    if G.has_edge(a,b):
                        G[a][b]["w"] += 1
                    else:
                        G.add_edge(a, b, w=1)
        # ambil top degree 30
        if G.number_of_nodes() == 0:
            self.canvas.ax.text(0.5,0.5,"No authors found", ha='center'); return
        deg = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]
        nodes = [n for n,_ in deg]
        H = G.subgraph(nodes).copy()
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx_nodes(H, pos, ax=self.canvas.ax)
        nx.draw_networkx_edges(H, pos, ax=self.canvas.ax, width=[1+H[u][v]["w"]*0.2 for u,v in H.edges()])
        nx.draw_networkx_labels(H, pos, ax=self.canvas.ax, font_size=8)
        self.canvas.ax.set_title("Author Collaboration Network (Top 30)"); self.canvas.ax.axis("off")

    # 9. Treemap of Keywords (Top 30, Normalized Recovery Terms)
    def _treemap_keywords_normalized(self):
        # Kita buat treemap sederhananya sebagai bar area (rectangles) tanpa lib tambahan.
        # Normalisasi istilah 'recovery' dsb. kamu bisa pasok kamus mapping di sini.
        mapping = {
            "data recovery": ["data recovery", "recovery", "rollback", "restore"],
            "fault tolerance": ["fault tolerance","resilience","robust"],
            "blockchain": ["blockchain","dlt","distributed ledger"],
            # tambahkan sesuai kebutuhanmu
        }
        k = self._author_kw_col()
        if not k:
            self.canvas.ax.text(0.5,0.5,"Author Keywords column not found", ha='center'); return
        bag = {}
        for s in self._series_clean(self.df[k]).tolist():
            for kw in self._split_semicolon(s):
                base = kw.lower()
                norm = None
                for cat, keys in mapping.items():
                    if any(sub in base for sub in keys):
                        norm = cat; break
                bag[norm or kw] = bag.get(norm or kw, 0) + 1
        top = sorted(bag.items(), key=lambda x: x[1], reverse=True)[:30]
        if not top:
            self.canvas.ax.text(0.5,0.5,"No keywords", ha='center'); return
        # treemap kasar: plot rectangles proporsional secara grid
        total = sum(v for _,v in top)
        x, y, w, h = 0.0, 0.0, 1.0, 1.0
        # simple slice-and-dice
        curx = 0.0
        for label, val in top:
            width = w * (val/total)
            self.canvas.ax.add_patch(plt.Rectangle((curx, y), width, h, fill=False))
            self.canvas.ax.text(curx+width/2, y+h/2, f"{label}\n{val}", ha='center', va='center', fontsize=8)
            curx += width
        self.canvas.ax.set_xlim(0,1); self.canvas.ax.set_ylim(0,1)
        self.canvas.ax.axis("off"); self.canvas.ax.set_title("Treemap of Keywords (Top 30, Normalized)")

    def plot_treemap_keywords(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        try:
            freq = self._build_keyword_freq()
            self.canvas.ax.clear()
            self.canvas.figure.clf()
            self.canvas.ax = self.canvas.figure.add_subplot(111)
            if freq is None or freq.empty:
                self.canvas.ax.text(0.5, 0.5, "No keyword columns found or empty.", ha="center", va="center")
                self.canvas.draw()
                return

            top30 = freq.head(30).copy()

            # --- Prefer: Plotly (interaktif) ---
            if px is not None:
                fig = px.treemap(
                    top30,
                    path=['Keyword'],
                    values='Frequency',
                    color='Frequency',
                    color_continuous_scale='Tealgrn',
                    title='Treemap of Keywords (Top 30)'
                )
                fig.update_layout(width=800, height=800)

                # Tampilkan: kalau WebEngine ada, render inline; kalau tidak, simpan HTML & buka di browser
                html = fig.to_html(include_plotlyjs='cdn', full_html=True)
                html_path = CACHE_DIR / "treemap_keywords.html"
                html_path.write_text(html, encoding="utf-8")

                if WEBVIEW_AVAILABLE:
                    # Render di dalam dock sementara: gunakan browser default (simple)
                    # Jika kamu sudah punya QWebEngineView dock, kamu bisa setHtml() di sana.
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))

                # beri info kecil di canvas
                self.canvas.ax.text(0.5, 0.5, "Treemap opened in browser (Plotly).", ha="center", va="center")
                self.canvas.draw()
                return

            # --- Fallback: Matplotlib (tanpa lib tambahan) ---
            # Treemap sederhana (slice-and-dice) agar tidak perlu 'squarify'
            self.canvas.ax.set_title("Treemap of Keywords (Top 30)")
            total = top30['Frequency'].sum()
            x, y, w, h = 0.0, 0.0, 1.0, 1.0
            curx = 0.0
            for _, row in top30.iterrows():
                frac = (row['Frequency'] / total) if total else 0
                width = w * frac
                # kotak
                rect = plt.Rectangle((curx, y), width, h, fill=False)
                self.canvas.ax.add_patch(rect)
                # label
                self.canvas.ax.text(
                    curx + width/2, y + h/2,
                    f"{row['Keyword']}\n{int(row['Frequency'])}",
                    ha='center', va='center', fontsize=8
                )
                curx += width

            self.canvas.ax.set_xlim(0, 1); self.canvas.ax.set_ylim(0, 1)
            self.canvas.ax.axis("off")
            self.canvas.draw()

        except Exception as e:
            self._show_error(e)
            self.canvas.ax.clear()
            self.canvas.ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            self.canvas.draw()

    def plot_treemap_keywords_squarify(self):
        if self.df is None:
            QMessageBox.information(self, APP_NAME, "Load data first.")
            return
        try:
            freq = self._build_keyword_freq()
            self.canvas.ax.clear()
            self.canvas.figure.clf()
            self.canvas.ax = self.canvas.figure.add_subplot(111)
            if freq is None or freq.empty:
                self.canvas.ax.text(0.5, 0.5, "No keyword columns found or empty.", ha="center", va="center")
                self.canvas.draw()
                return

            top30 = freq.head(30).copy()

            # --- PRIORITAS: squarify di Matplotlib canvas (jika tersedia) ---
            if squarify is not None:
                sizes = top30["Frequency"].tolist()
                # label: "keyword\nfreq"
                labels = [f"{k}\n{int(v)}" for k, v in zip(top30["Keyword"], top30["Frequency"])]
                # biar teks lebih kebaca, kita plot tanpa warna khusus (default) -> user bebas ganti nanti
                squarify.plot(sizes=sizes, label=labels, ax=self.canvas.ax, pad=True)
                self.canvas.ax.set_title("Treemap of Keywords (Top 30)")
                self.canvas.ax.axis("off")
                self.canvas.figure.tight_layout()
                self.canvas.draw()
                return

            # --- Kedua: Plotly interaktif (punyamu sebelumnya) ---
            if px is not None:
                fig = px.treemap(
                    top30,
                    path=['Keyword'],
                    values='Frequency',
                    color='Frequency',
                    color_continuous_scale='Tealgrn',
                    title='Treemap of Keywords (Top 30)'
                )
                fig.update_layout(width=800, height=800)

                html = fig.to_html(include_plotlyjs='cdn', full_html=True)
                html_path = CACHE_DIR / "treemap_keywords.html"
                html_path.write_text(html, encoding="utf-8")

                if WEBVIEW_AVAILABLE:
                    self.plotly_view.setHtml(html)
                    self.plotly_dock.raise_()
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))

                # info kecil di canvas
                self.canvas.ax.text(0.5, 0.5, "Treemap opened in browser (Plotly).", ha="center", va="center")
                self.canvas.draw()
                return

            # --- Fallback: slice-and-dice Matplotlib tanpa dependensi ---
            self.canvas.ax.set_title("Treemap of Keywords (Top 30)")
            total = float(top30['Frequency'].sum()) or 1.0
            x, y, w, h = 0.0, 0.0, 1.0, 1.0
            curx = 0.0
            for _, row in top30.iterrows():
                frac = row['Frequency'] / total
                width = w * frac
                rect = plt.Rectangle((curx, y), width, h, fill=False)
                self.canvas.ax.add_patch(rect)
                self.canvas.ax.text(
                    curx + width/2, y + h/2,
                    f"{row['Keyword']}\n{int(row['Frequency'])}",
                    ha='center', va='center', fontsize=8
                )
                curx += width
            self.canvas.ax.set_xlim(0, 1); self.canvas.ax.set_ylim(0, 1)
            self.canvas.ax.axis("off")
            self.canvas.draw()

        except Exception as e:
            self._show_error(e)
            self.canvas.ax.clear()
            self.canvas.ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            self.canvas.draw()

    # 10. Institution Type Distribution (Academic vs Non-Academic)
    def _plot_institution_type_dist(self):
        c_aff = self._get_col("affiliations", "Affiliations", "Authors with affiliations", "Author Affiliations")
        if not c_aff:
            self.canvas.ax.text(0.5, 0.5, "Affiliations column not found", ha='center'); 
            self.canvas.draw(); 
            return

        def simple_type(s: str) -> str:
            t = str(s).lower()
            if any(k in t for k in ["univ","university","college","faculty","school of","school","üniversitesi","instituto","faculté","institut","uniwersytet", "nyu", "unsw", "ucla", "mit", "harvard", "stanford", "cambridge", "oxford", "eth", "epfl", "tudelft", "nus", "nanyang", "uio", "uio", "uottawa", "utoronto"]):
                return "Academic"
            if any(k in t for k in ["research","academy","laboratory","lab","institute","recherche","centre","center","observatory","observatorium","observatoire","researcher","researchers"]):
                return "Research"
            if any(k in t for k in ["corp","co.","inc","ltd","tech","company","pt ","tbk","industry","bank","consult","firm","gmbh","sa","ag","llc","pvt","private","enterprise","enterprises"]):
                return "Industry"
            if any(k in t for k in ["gov","government","ministry","bureau","department","kementerian","parliament","commission","interministérielle","agency","agencies","national","state","municipal","city","cities","council","councils","public","police","defense","defence","army","navy","air force","military","hospital","hôpital","health","clinic","clinique"]):
                return "Government"
            return "Unknown"

        ser = self._series_clean(self.df[c_aff]).apply(simple_type).value_counts()

        # — Reset kanvas dengan layout stabil (tidak ubah ukuran figure) —
        ax = self._reset_canvas(top_pad=0.88)

        # urut naik agar barh enak dibaca
        ser = ser.sort_values(ascending=True)

        if ser.empty:
            ax.text(0.5, 0.5, "No data", ha='center'); 
            self.canvas.draw(); 
            return

        bars = ax.barh(ser.index.tolist(), ser.values.tolist())

        # anotasi nilai di ujung bar
        xmax = float(ser.values.max())
        ax.set_xlim(0, xmax * 1.10)
        for bar, val in zip(bars, ser.values):
            ax.text(bar.get_width() + xmax * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    str(int(val)),
                    va='center', ha='left', fontsize=9)

        ax.set_xlabel("Documents")
        ax.set_title("Institution Type Distribution", pad=16)
        ax.grid(axis='x', linestyle=':', alpha=0.35)

        # ruang kiri lebih besar untuk label panjang; jangan pakai subplots_adjust + tight_layout barengan
        self.canvas.figure.tight_layout(rect=[0.25, 0.06, 0.98, 0.94])
        self.canvas.draw()

    # 11. Documents by Country (Top 20)
    def _plot_docs_by_country_top20(self):
        # Pakai helper yang sudah “negara-only”
        ser_country = self._get_country_series()
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        if ser_country is None or ser_country.eq("").all():
            self.canvas.ax.text(0.5, 0.5, "No country info", ha='center', va='center')
            self.canvas.draw()
            return

        # Hitung top-20 negara
        counts = (ser_country
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts())

        if counts.empty:
            self.canvas.ax.text(0.5, 0.5, "No country info", ha='center', va='center')
            self.canvas.draw()
            return

        top = counts.head(20).sort_values(ascending=True)  # naik → barh dari bawah ke atas

        # Plot
        bars = self.canvas.ax.barh(top.index.tolist(), top.values.tolist())

        # Label angka di ujung bar
        xmax = top.values.max()
        
        for bar, val in zip(bars, top.values):
            self.canvas.ax.text(bar.get_width() + max(1, xmax) * 0.01,
                                bar.get_y() + bar.get_height()/2,
                                str(int(val)),
                                va='center', ha='left', fontsize=9)

        self.canvas.ax.set_xlabel("Documents")
        self.canvas.ax.set_title("Documents by Country (Top 20)", pad=14)
        self.canvas.ax.grid(axis='x', linestyle=':', alpha=0.35)

        # ruang kiri untuk label negara
        self.canvas.figure.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.12)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # 12. World Map Choropleth (matplotlib-only fallback)

    def _pick_world_shapefile(self) -> str | None:
        """Urutan: ./110m/ne_110m_admin_0_countries.shp -> ~/.slrdesk/cache/... -> dialog"""
        # 1) relatif ke file main.py
        default_shp = (Path(__file__).parent / "110m" / "ne_110m_admin_0_countries.shp").resolve()
        # 2) cache
        cache_shp = (CACHE_DIR / "ne_110m_admin_0_countries.shp").resolve()
        if default_shp.exists():
            return str(default_shp)
        if cache_shp.exists():
            return str(cache_shp)
        # 3) dialog
        shp_path, _ = QFileDialog.getOpenFileName(
            self, "Select world shapefile (.shp)", str(Path.home()), "Shapefile (*.shp)"
        )
        return shp_path or None

    def _map_choropleth_docs_country(self):
        if gpd is None:
            self.canvas.ax.text(0.5,0.5,"GeoPandas not installed.\nInstall: pip install geopandas shapely fiona pyproj", ha='center', va='center')
            return

        ser = self._get_country_series()
        if ser is None or ser.eq("").all():
            self.canvas.ax.text(0.5,0.5,"No 'Country' column or values.", ha='center', va='center')
            return

        shp_path = self._pick_world_shapefile()
        if not shp_path:
            self.canvas.ax.text(0.5,0.5,"No shapefile selected/found.", ha='center', va='center')
            return

        # Baca shapefile
        try:
            world = gpd.read_file(shp_path)
        except Exception as e:
            self.canvas.ax.text(0.5,0.5,f"Failed to read shapefile:\n{e}", ha='center', va='center'); return

        # siapkan distribusi negara
        country_dist = ser.value_counts().reset_index()
        country_dist.columns = ['Country', 'Document Count']
        country_dist['Country'] = country_dist['Country'].str.strip()

        # kolom nama negara di shapefile
        name_col = None
        for cand in ("ADMIN", "admin", "NAME", "name"):
            if cand in world.columns:
                name_col = cand
                break
        if not name_col:
            self.canvas.ax.text(0.5,0.5,"Shapefile needs 'ADMIN'/'NAME' column.", ha='center', va='center'); return

        world = world.copy()
        world['name_norm'] = world[name_col].astype(str).str.strip()

        merged = world.merge(country_dist, how='left', left_on='name_norm', right_on='Country')

        # plot
        merged.plot(
            column='Document Count',
            cmap='viridis',
            linewidth=0.8,
            ax=self.canvas.ax,
            edgecolor='0.8',
            legend=True,
            missing_kwds={"color": "lightgrey", "label": "No data"}
        )
        self.canvas.ax.set_title('Distribusi Dokumen per Negara (World Map Choropleth)')
        self.canvas.ax.axis('off')

        # 13. World Map (GeoPandas) (Docs per Country)
        def _map_geopandas_docs_country(self):
            # Sama dengan (12), biar konsisten (bisa kita pake gaya lain)
            self._map_choropleth_docs_country()

    # 14. Collaboration Scope by Country (Top 20) — Horizontal Stacked
    def _plot_collab_scope_by_country_stacked(self):
        data = self._build_collab_scope_table()
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        if data is None or data.empty:
            self.canvas.ax.text(0.5, 0.5, "Collaboration scope data not available", ha='center', va='center')
            self.canvas.draw()
            return

        y_labels = data.index.tolist()
        y_pos = range(len(data))

        # tumpuk Domestic + International
        domestic = data["Domestic"].values
        international = data["International"].values
        
        self.canvas.ax.barh(y_pos, domestic, label="Domestic", edgecolor='white')
        self.canvas.ax.barh(y_pos, international, left=domestic, label="International", edgecolor='white')

        self.canvas.ax.set_yticks(list(y_pos), labels=y_labels)
        self.canvas.ax.set_title("Collaboration Scope by Country (Top 20) — Horizontal Stacked")
        self.canvas.ax.set_xlabel("Number of Papers")
        self.canvas.ax.set_ylabel("Country")
        self.canvas.ax.legend(title="Collaboration Scope", loc='lower right')
        # agar negara terbanyak di atas (seperti contohmu)
        self.canvas.ax.invert_yaxis()

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # 15. Collaboration Scope by Country — % by Category
    def _plot_collab_scope_by_country_percent(self):
        data = self._build_collab_scope_table()
        self.canvas.ax.clear()
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        if data is None or data.empty:
            self.canvas.ax.text(0.5, 0.5, "Collaboration scope data not available", ha='center', va='center')
            self.canvas.draw()
            return

        # Hitung total & persentase
        totals = data.sum(axis=1)
        # Hindari div by zero
        denom = totals.replace(0, 1)
        domestic_p = (data["Domestic"] / denom) * 100
        international_p = (data["International"] / denom) * 100

        y_labels = data.index.tolist()
        y_pos = range(len(data))

        # tumpukan persentase
        bars_dom = self.canvas.ax.barh(y_pos, domestic_p.values, label="Domestic", edgecolor='white')
        bars_int = self.canvas.ax.barh(y_pos, international_p.values, left=domestic_p.values, label="International", edgecolor='white')

        # Tambahkan label % di dalam bar (seperti notebook)
        # untuk dua seri, kita iterate dua set bar
        # domestic first
        for idx, bar in enumerate(bars_dom):
            width = bar.get_width()
            if width > 1:  # tampilkan jika >1%
                self.canvas.ax.text(
                    bar.get_x() + width/2,
                    bar.get_y() + bar.get_height()/2,
                    f"{width:.0f}%",
                    ha='center', va='center', fontsize=9, color='white'
                )
        # international
        for idx, bar in enumerate(bars_int):
            width = bar.get_width()
            # posisi X bar international = left(domestic_p[idx]) + width/2 — tapi matplotlib sudah menempatkan bar.x sebagai left
            if width > 1:
                self.canvas.ax.text(
                    bar.get_x() + width/2,
                    bar.get_y() + bar.get_height()/2,
                    f"{width:.0f}%",
                    ha='center', va='center', fontsize=9, color='white'
                )

        # Format
        self.canvas.ax.set_yticks(list(y_pos), labels=y_labels)
        self.canvas.ax.set_title("Collaboration Scope by Country (Top 20) — % by Category")
        self.canvas.ax.set_xlabel("Percent of Papers")
        self.canvas.ax.set_ylabel("Country")
        self.canvas.ax.legend(title="Collaboration Scope", loc='lower right')
        self.canvas.ax.invert_yaxis()

        # batas x-axis 0–100
        self.canvas.ax.set_xlim(0, 100)
        
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # 16. Distribution of Institutional Collaboration Types
    def _plot_institution_collab_types(self):
        # Heuristik: dari Affiliations, deteksi jenis instansi per paper → gabungkan tipe kolaborasi:
        # Academic-Academic, Academic-NonAcademic, NonAcademic-NonAcademic
        c_aff = self._get_col("Affiliations")
        if not c_aff:
            self.canvas.ax.text(0.5,0.5,"Affiliations column not found", ha='center'); return
        def paper_type(s):
            orgs = self._split_semicolon(str(s)) or [str(s)]
            types = set(self._institution_type(org) for org in orgs)
            if types == {"Academic"}:
                return "Academic–Academic"
            if types == {"Non-Academic"}:
                return "NonAcademic–NonAcademic"
            return "Academic–NonAcademic"
        ser = self._series_clean(self.df[c_aff]).apply(paper_type).value_counts()
        # Bersihkan seluruh figure (termasuk colorbar-colorbar lama)
        self.canvas.figure.clf()
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        self.canvas.ax.bar(ser.index, ser.values)
        self.canvas.ax.set_ylabel("Documents"); self.canvas.ax.set_title("Institutional Collaboration Types")
        self.canvas.ax.tick_params(axis='x', rotation=20)
    

# --------------------------------- main ------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
