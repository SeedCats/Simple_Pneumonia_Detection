"""
GUI Module
Handles all user interface components and event handlers
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np

# Import configuration and helper modules
from config import (
    MODEL_PATH, NUM_CLASSES, IMAGE_SIZE, CLASS_NAMES,
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, MEDICAL_URLS,
    TEMP_GRAD_CAM_PATH, TEMP_IMAGE_PATH,
)
from models import get_transform, load_trained_model, predict_pneumonia_advanced
from rag import WebBasedMedicalKnowledgeRAG
from ai_agent import (
    call_grok_api, generate_initial_analysis_prompt, format_conversation_for_display
)
from pdf_report import create_pdf_report


# ============================================================================
# MODERN COLOR SCHEME & STYLING
# ============================================================================
class Colors:
    """Modern color palette for the application"""
    # Primary colors
    PRIMARY = "#2563EB"          # Blue
    PRIMARY_DARK = "#1D4ED8"     # Darker blue
    PRIMARY_LIGHT = "#3B82F6"    # Lighter blue

    # Secondary colors
    SECONDARY = "#10B981"        # Green/Teal
    SECONDARY_DARK = "#059669"   # Darker green

    # Accent colors
    ACCENT = "#8B5CF6"           # Purple
    WARNING = "#F59E0B"          # Amber/Orange
    DANGER = "#EF4444"           # Red
    SUCCESS = "#10B981"          # Green

    # Neutral colors
    BG_MAIN = "#F8FAFC"          # Very light gray background
    BG_CARD = "#FFFFFF"          # White card background
    BG_DARK = "#1E293B"          # Dark background
    BG_SIDEBAR = "#F1F5F9"       # Light sidebar

    # Text colors
    TEXT_PRIMARY = "#1E293B"     # Dark text
    TEXT_SECONDARY = "#64748B"   # Gray text
    TEXT_LIGHT = "#FFFFFF"       # White text
    TEXT_MUTED = "#94A3B8"       # Muted text

    # Border colors
    BORDER = "#E2E8F0"           # Light border
    BORDER_FOCUS = "#3B82F6"     # Focus border

    # Status colors
    NORMAL_BG = "#DCFCE7"        # Light green for normal
    NORMAL_TEXT = "#166534"      # Dark green text
    PNEUMONIA_BG = "#FEE2E2"     # Light red for pneumonia
    PNEUMONIA_TEXT = "#991B1B"   # Dark red text


class Fonts:
    """Font configurations"""
    TITLE = ("Segoe UI", 20, "bold")
    HEADING = ("Segoe UI", 14, "bold")
    SUBHEADING = ("Segoe UI", 12, "bold")
    BODY = ("Segoe UI", 11)
    BODY_SMALL = ("Segoe UI", 10)
    BUTTON = ("Segoe UI", 11, "bold")
    MONO = ("Consolas", 10)


class ModernStyle:
    """Configure modern ttk styles"""

    @staticmethod
    def configure(root):
        style = ttk.Style(root)
        style.theme_use('clam')

        # Configure main styles
        style.configure(".",
            background=Colors.BG_MAIN,
            foreground=Colors.TEXT_PRIMARY,
            font=Fonts.BODY
        )

        # Primary Button
        style.configure("Primary.TButton",
            background=Colors.PRIMARY,
            foreground=Colors.TEXT_LIGHT,
            padding=(20, 12),
            font=Fonts.BUTTON,
            borderwidth=0
        )
        style.map("Primary.TButton",
            background=[("active", Colors.PRIMARY_DARK), ("disabled", Colors.TEXT_MUTED)],
            foreground=[("disabled", Colors.BG_CARD)]
        )

        # Secondary Button
        style.configure("Secondary.TButton",
            background=Colors.SECONDARY,
            foreground=Colors.TEXT_LIGHT,
            padding=(20, 12),
            font=Fonts.BUTTON,
            borderwidth=0
        )
        style.map("Secondary.TButton",
            background=[("active", Colors.SECONDARY_DARK), ("disabled", Colors.TEXT_MUTED)]
        )

        # Accent Button
        style.configure("Accent.TButton",
            background=Colors.ACCENT,
            foreground=Colors.TEXT_LIGHT,
            padding=(20, 12),
            font=Fonts.BUTTON,
            borderwidth=0
        )
        style.map("Accent.TButton",
            background=[("active", "#7C3AED"), ("disabled", Colors.TEXT_MUTED)]
        )

        # Warning Button (PDF Report)
        style.configure("Warning.TButton",
            background=Colors.WARNING,
            foreground=Colors.TEXT_LIGHT,
            padding=(20, 12),
            font=Fonts.BUTTON,
            borderwidth=0
        )
        style.map("Warning.TButton",
            background=[("active", "#D97706"), ("disabled", Colors.TEXT_MUTED)]
        )

        # Outline Button
        style.configure("Outline.TButton",
            background=Colors.BG_CARD,
            foreground=Colors.PRIMARY,
            padding=(15, 10),
            font=Fonts.BODY,
            borderwidth=2
        )
        style.map("Outline.TButton",
            background=[("active", Colors.BG_SIDEBAR)]
        )

        # Card Frame
        style.configure("Card.TFrame",
            background=Colors.BG_CARD,
            relief="flat"
        )

        # Sidebar Frame
        style.configure("Sidebar.TFrame",
            background=Colors.BG_SIDEBAR
        )

        # Labels
        style.configure("Title.TLabel",
            background=Colors.BG_SIDEBAR,
            foreground=Colors.TEXT_PRIMARY,
            font=Fonts.TITLE
        )

        style.configure("Heading.TLabel",
            background=Colors.BG_CARD,
            foreground=Colors.TEXT_PRIMARY,
            font=Fonts.HEADING
        )

        style.configure("Body.TLabel",
            background=Colors.BG_SIDEBAR,
            foreground=Colors.TEXT_SECONDARY,
            font=Fonts.BODY
        )

        # Entry
        style.configure("Modern.TEntry",
            fieldbackground=Colors.BG_CARD,
            foreground=Colors.TEXT_PRIMARY,
            padding=(10, 8),
            font=Fonts.BODY
        )

        # Progress bar
        style.configure("Primary.Horizontal.TProgressbar",
            background=Colors.PRIMARY,
            troughcolor=Colors.BORDER,
            borderwidth=0,
            thickness=6
        )

        return style


class PneumoniaApp:
    """Main GUI application for pneumonia detection - Modern Design"""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=Colors.BG_MAIN)
        self.root.minsize(1200, 700)

        # Configure modern styles
        self.style = ModernStyle.configure(root)

        # Initialize RAG with website-based knowledge base
        self.rag_retriever = WebBasedMedicalKnowledgeRAG(seed_urls=MEDICAL_URLS)
        print(f"RAG loaded {len(self.rag_retriever.documents) if self.rag_retriever.documents else 0} documents")

        # Initialize model and state variables
        self._init_model_state()

        # Setup UI
        self.setup_ui()
        self.load_model_on_startup()

        # Register cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_model_state(self):
        """Initialize model and state variables"""
        self.model = None
        self.transform = get_transform()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.current_image_path = None
        self.current_pil_image = None
        self.current_prediction = None
        self.current_confidence = None
        self.current_analysis_data = None
        self.current_grad_cam_overlay_image_path = None

        self.grok_conversation_history = []
        self.analysis_history = []

        # Store paths for cleanup
        self.kb_files = [
            self.rag_retriever.knowledge_base_path,
            self.rag_retriever.index_path
        ]

    def on_closing(self):
        """Cleanup local crawled data and close application"""
        print("\n" + "="*60)
        print("Closing application and cleaning up...")
        print("="*60)
        self._cleanup_knowledge_base_files()
        self.root.destroy()

    def _cleanup_knowledge_base_files(self):
        """Delete local knowledge base files"""
        self.rag_retriever.cleanup()

        # Also cleanup temporary image files
        temp_files = [TEMP_GRAD_CAM_PATH, TEMP_IMAGE_PATH]
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✓ Cleanup: Deleted {file_path}")
            except Exception as e:
                print(f"Cleanup warning: Could not delete {file_path}: {e}")

    def load_model_on_startup(self):
        """Load AI model on application startup"""
        self._update_status("Loading AI model...", "loading")
        self.root.update_idletasks()
        self.model = load_trained_model(MODEL_PATH, NUM_CLASSES, self.device)
        if self.model:
            device_name = "GPU (CUDA)" if "cuda" in str(self.device) else "CPU"
            self._update_status(f"✓ AI Agent ready • Using {device_name}", "success")
            self.ai_interpretation_btn.config(state=tk.DISABLED)
        else:
            self._update_status("✗ AI Agent loading failed. Check console for errors.", "error")
            self.browse_btn.config(state=tk.DISABLED)
            self.ai_interpretation_btn.config(state=tk.DISABLED)

    def setup_ui(self):
        """Setup all UI components with modern design"""
        self._setup_main_container()
        self._setup_sidebar()
        self._setup_main_content()
        self._setup_status_bar()

    def _setup_main_container(self):
        """Set up the main container with proper grid layout"""
        self.root.grid_columnconfigure(0, weight=0, minsize=320)  # Sidebar
        self.root.grid_columnconfigure(1, weight=1)  # Main content
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)  # Status bar

    def _setup_sidebar(self):
        """Setup left sidebar with controls"""
        # Sidebar container
        self.sidebar = tk.Frame(self.root, bg=Colors.BG_SIDEBAR, width=320)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)

        # Inner padding frame
        sidebar_inner = tk.Frame(self.sidebar, bg=Colors.BG_SIDEBAR)
        sidebar_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # ===== HEADER SECTION =====
        header_frame = tk.Frame(sidebar_inner, bg=Colors.BG_SIDEBAR)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_container = tk.Frame(header_frame, bg=Colors.BG_SIDEBAR)
        title_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(title_container, text="Pneumonia Prediction", font=Fonts.TITLE, bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_PRIMARY).pack(anchor="w")

        # Separator
        self._create_separator(sidebar_inner)

        # ===== FILE SELECTION SECTION =====
        file_section = tk.Frame(sidebar_inner, bg=Colors.BG_SIDEBAR)
        file_section.pack(fill=tk.X, pady=(0, 20))

        tk.Label(file_section, text="📁 Image Selection", font=Fonts.SUBHEADING,
                bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        # Selected file display
        self.file_display = tk.Frame(file_section, bg=Colors.BG_CARD, highlightbackground=Colors.BORDER,
                                     highlightthickness=1, height=60)
        self.file_display.pack(fill=tk.X, pady=(0, 12))
        self.file_display.pack_propagate(False)

        file_inner = tk.Frame(self.file_display, bg=Colors.BG_CARD)
        file_inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self.file_icon_label = tk.Label(file_inner, text="📷", font=("Segoe UI", 16),
                                        bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED)
        self.file_icon_label.pack(side=tk.LEFT, padx=(0, 8))

        file_text_frame = tk.Frame(file_inner, bg=Colors.BG_CARD)
        file_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.path_label = tk.Label(file_text_frame, text="No image selected", font=Fonts.BODY,
                                  bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY, anchor="w",
                                  wraplength=200)
        self.path_label.pack(anchor="w")

        self.file_size_label = tk.Label(file_text_frame, text="Select an X-ray to analyze",
                                        font=Fonts.BODY_SMALL, bg=Colors.BG_CARD,
                                        fg=Colors.TEXT_MUTED, anchor="w", wraplength=180)
        self.file_size_label.pack(anchor="w")

        # Browse button
        self.browse_btn = tk.Button(file_section, text="📂  Browse X-ray Image",
                                    font=Fonts.BUTTON, bg=Colors.PRIMARY, fg=Colors.TEXT_LIGHT,
                                    activebackground=Colors.PRIMARY_DARK, activeforeground=Colors.TEXT_LIGHT,
                                    cursor="hand2", relief="flat", pady=12,
                                    command=self.browse_image)
        self.browse_btn.pack(fill=tk.X, ipady=2)
        self._add_button_hover(self.browse_btn, Colors.PRIMARY, Colors.PRIMARY_DARK)

        # ===== ANALYSIS SECTION =====
        analysis_section = tk.Frame(sidebar_inner, bg=Colors.BG_SIDEBAR)
        analysis_section.pack(fill=tk.X, pady=(0, 20))

        tk.Label(analysis_section, text="🔬 Analysis", font=Fonts.SUBHEADING,
                bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        # Run Analysis button
        self.analyze_btn = tk.Button(analysis_section, text="▶  Run AI Analysis",
                                     font=Fonts.BUTTON, bg="#059669", fg="#FFFFFF",
                                     activebackground="#047857", activeforeground="#FFFFFF",
                                     cursor="hand2", relief="flat", pady=12, state=tk.DISABLED,
                                     disabledforeground="#9CA3AF",
                                     command=self._run_advanced_analysis)
        self.analyze_btn.pack(fill=tk.X, ipady=2, pady=(0, 8))
        self._add_button_hover(self.analyze_btn, "#059669", "#047857")

        # ===== RESULTS CARD =====
        self.results_card = tk.Frame(sidebar_inner, bg=Colors.BG_CARD, highlightbackground=Colors.BORDER,
                                     highlightthickness=1)
        self.results_card.pack(fill=tk.X, pady=(0, 15))

        results_inner = tk.Frame(self.results_card, bg=Colors.BG_CARD)
        results_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        tk.Label(results_inner, text="Diagnosis Results", font=Fonts.SUBHEADING,
                bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(anchor="w", pady=(0, 12))

        # Prediction display
        pred_frame = tk.Frame(results_inner, bg=Colors.BG_CARD)
        pred_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(pred_frame, text="Prediction:", font=Fonts.BODY,
                bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(side=tk.LEFT)

        self.prediction_badge = tk.Label(pred_frame, text="N/A", font=Fonts.BODY,
                                         bg=Colors.BORDER, fg=Colors.TEXT_SECONDARY,
                                         padx=12, pady=4)
        self.prediction_badge.pack(side=tk.RIGHT)

        # Confidence display
        conf_frame = tk.Frame(results_inner, bg=Colors.BG_CARD)
        conf_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(conf_frame, text="Confidence:", font=Fonts.BODY,
                bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(side=tk.LEFT)

        self.confidence_label = tk.Label(conf_frame, text="—", font=("Segoe UI", 14, "bold"),
                                         bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY)
        self.confidence_label.pack(side=tk.RIGHT)

        # Confidence progress bar
        self.confidence_progress = ttk.Progressbar(results_inner, style="Primary.Horizontal.TProgressbar",
                                                   length=200, mode='determinate', maximum=100)
        self.confidence_progress.pack(fill=tk.X, pady=(0, 10))

        # Feature analysis (collapsible)
        self.feature_label = tk.Label(results_inner, text="", font=Fonts.BODY_SMALL,
                                      bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY,
                                      justify=tk.LEFT, anchor="w", wraplength=250)
        self.feature_label.pack(anchor="w")

        # ===== ACTION BUTTONS =====
        actions_section = tk.Frame(sidebar_inner, bg=Colors.BG_SIDEBAR)
        actions_section.pack(fill=tk.X, pady=(0, 10))

        # AI Interpretation button
        self.ai_interpretation_btn = tk.Button(actions_section, text="🤖  Get AI Interpretation",
                                               font=Fonts.BUTTON, bg="#7C3AED", fg="#FFFFFF",
                                               activebackground="#6D28D9", activeforeground="#FFFFFF",
                                               cursor="hand2", relief="flat", pady=12, state=tk.DISABLED,
                                               disabledforeground="#9CA3AF",
                                               command=self.perform_initial_grok_analysis)
        self.ai_interpretation_btn.pack(fill=tk.X, ipady=2, pady=(0, 8))
        self._add_button_hover(self.ai_interpretation_btn, "#7C3AED", "#6D28D9")

        # PDF Report button
        self.pdf_report_btn = tk.Button(actions_section, text="📄  Generate PDF Report",
                                        font=Fonts.BUTTON, bg="#D97706", fg="#FFFFFF",
                                        activebackground="#B45309", activeforeground="#FFFFFF",
                                        cursor="hand2", relief="flat", pady=12, state=tk.DISABLED,
                                        disabledforeground="#9CA3AF",
                                        command=self.generate_pdf_report)
        self.pdf_report_btn.pack(fill=tk.X, ipady=2)
        self._add_button_hover(self.pdf_report_btn, "#D97706", "#B45309")

    def _setup_main_content(self):
        """Setup main content area with image display and analysis panel"""
        # Main content container
        main_content = tk.Frame(self.root, bg=Colors.BG_MAIN)
        main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main_content.grid_columnconfigure(0, weight=1)
        main_content.grid_columnconfigure(1, weight=1)
        main_content.grid_rowconfigure(0, weight=1)

        # ===== LEFT: IMAGE DISPLAY PANEL =====
        image_panel = tk.Frame(main_content, bg=Colors.BG_CARD, highlightbackground=Colors.BORDER,
                               highlightthickness=1)
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        image_header = tk.Frame(image_panel, bg=Colors.BG_CARD)
        image_header.pack(fill=tk.X, padx=20, pady=(15, 10))

        tk.Label(image_header, text="🖼️ X-ray Visualization", font=Fonts.HEADING,
                bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(side=tk.LEFT)

        # Image canvas container
        self.image_container = tk.Frame(image_panel, bg=Colors.BG_MAIN)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Setup matplotlib figure
        self.fig, (self.ax_original, self.ax_grad_cam) = plt.subplots(2, 1, figsize=(5, 7))
        self.fig.patch.set_facecolor(Colors.BG_MAIN)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas_widget.configure(bg=Colors.BG_MAIN)

        # Configure initial axes
        for ax in [self.ax_original, self.ax_grad_cam]:
            ax.set_facecolor(Colors.BG_MAIN)
            ax.axis('off')

        self.ax_original.set_title("Original X-ray Image", fontsize=11, fontweight='bold',
                                   color=Colors.TEXT_PRIMARY, pad=10)
        self.ax_grad_cam.set_title("Grad-CAM Heatmap (Available after analysis)", fontsize=11,
                                   fontweight='bold', color=Colors.TEXT_SECONDARY, pad=10)
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

        # ===== RIGHT: ANALYSIS PANEL =====
        analysis_panel = tk.Frame(main_content, bg=Colors.BG_CARD, highlightbackground=Colors.BORDER,
                                  highlightthickness=1)
        analysis_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

        analysis_header = tk.Frame(analysis_panel, bg=Colors.BG_CARD)
        analysis_header.pack(fill=tk.X, padx=20, pady=(15, 10))

        tk.Label(analysis_header, text="💬 AI Agent Consultation", font=Fonts.HEADING,
                bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(side=tk.LEFT)

        # Clear button
        self.clear_btn = tk.Button(analysis_header, text="🗑️ Clear", font=Fonts.BODY_SMALL,
                                   bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_SECONDARY,
                                   activebackground=Colors.BORDER, relief="flat",
                                   cursor="hand2", padx=10, pady=4,
                                   command=self.clear_analysis)
        self.clear_btn.pack(side=tk.RIGHT)

        # Analysis text area with custom styling
        text_frame = tk.Frame(analysis_panel, bg=Colors.BG_CARD)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        self.analysis_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=Fonts.BODY,
            bg=Colors.BG_MAIN,
            fg=Colors.TEXT_PRIMARY,
            relief="flat",
            padx=15,
            pady=15,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=Colors.BORDER
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for styling
        self.analysis_text.tag_configure("welcome", foreground=Colors.TEXT_SECONDARY, font=Fonts.BODY)
        self.analysis_text.tag_configure("assistant", foreground=Colors.TEXT_PRIMARY, font=Fonts.BODY)
        self.analysis_text.tag_configure("user", foreground=Colors.PRIMARY, font=("Segoe UI", 11, "bold"))
        self.analysis_text.tag_configure("system", foreground=Colors.TEXT_MUTED, font=Fonts.BODY_SMALL)

        # Insert welcome message
        welcome_msg = """Welcome to the AI Pneumonia Detection Agent!

This intelligent system will help you analyze chest X-ray images for signs of pneumonia.

Here's how to get started:
  1. Click "Browse X-ray Image" to select an image
  2. Click "Run AI Analysis" to process the image
  3. Click "Get AI Interpretation" for detailed medical insights
  4. Ask follow-up questions in the chat below

The AI will provide:
  • Diagnostic predictions with confidence scores
  • Grad-CAM visualization showing areas of focus
  • Detailed clinical interpretations
  • Recommended next steps

⚠️ Note: This tool is for educational purposes only and should not replace professional medical diagnosis.
"""
        self.analysis_text.insert(tk.END, welcome_msg, "welcome")
        self.analysis_text.config(state=tk.DISABLED)

        # Question input area
        input_frame = tk.Frame(analysis_panel, bg=Colors.BG_CARD)
        input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        tk.Label(input_frame, text="Ask the AI Agent:", font=Fonts.BODY,
                bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor="w", pady=(0, 5))

        entry_container = tk.Frame(input_frame, bg=Colors.BG_CARD)
        entry_container.pack(fill=tk.X)

        self.question_entry = tk.Entry(entry_container, font=Fonts.BODY, bg=Colors.BG_MAIN,
                                       fg=Colors.TEXT_PRIMARY, relief="flat",
                                       highlightthickness=2, highlightbackground=Colors.BORDER,
                                       highlightcolor=Colors.PRIMARY, state=tk.DISABLED)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.question_entry.bind("<Return>", lambda e: self.send_question_to_grok())

        self.send_btn = tk.Button(entry_container, text="Send ➤", font=Fonts.BUTTON,
                                  bg=Colors.PRIMARY, fg=Colors.TEXT_LIGHT,
                                  activebackground=Colors.PRIMARY_DARK, relief="flat",
                                  cursor="hand2", padx=20, pady=8, state=tk.DISABLED,
                                  command=self.send_question_to_grok)
        self.send_btn.pack(side=tk.RIGHT)
        self._add_button_hover(self.send_btn, Colors.PRIMARY, Colors.PRIMARY_DARK)

    def _setup_status_bar(self):
        """Setup modern status bar"""
        self.status_bar = tk.Frame(self.root, bg=Colors.BG_DARK, height=40)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_bar.grid_propagate(False)

        status_inner = tk.Frame(self.status_bar, bg=Colors.BG_DARK)
        status_inner.pack(fill=tk.BOTH, expand=True, padx=20)

        # Status indicator dot
        self.status_dot = tk.Label(status_inner, text="●", font=("Segoe UI", 10),
                                   bg=Colors.BG_DARK, fg=Colors.SUCCESS)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8), pady=10)

        # Status message
        self.status_label = tk.Label(status_inner, text="AI Agent Ready", font=Fonts.BODY_SMALL,
                                     bg=Colors.BG_DARK, fg=Colors.TEXT_LIGHT, anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=10)

        # Device info
        device_text = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        self.device_label = tk.Label(status_inner, text=f"Device: {device_text}", font=Fonts.BODY_SMALL,
                                     bg=Colors.BG_DARK, fg=Colors.TEXT_MUTED)
        self.device_label.pack(side=tk.RIGHT, pady=10)

    def _create_separator(self, parent):
        """Create a subtle separator line"""
        sep = tk.Frame(parent, bg=Colors.BORDER, height=1)
        sep.pack(fill=tk.X, pady=15)

    def _add_button_hover(self, button, normal_color, hover_color):
        """Add hover effect to buttons"""
        def on_enter(e):
            if button['state'] != 'disabled':
                button['background'] = hover_color

        def on_leave(e):
            if button['state'] != 'disabled':
                button['background'] = normal_color

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def _update_status(self, message, status_type="info"):
        """Update status bar with appropriate styling"""
        colors = {
            "info": Colors.PRIMARY,
            "success": Colors.SUCCESS,
            "warning": Colors.WARNING,
            "error": Colors.DANGER,
            "loading": Colors.ACCENT
        }
        self.status_dot.config(fg=colors.get(status_type, Colors.TEXT_MUTED))
        self.status_label.config(text=message)

    def _update_analysis_text(self, text, role):
        """Helper to append text to the scrolledtext widget with role indication"""
        self.analysis_text.config(state=tk.NORMAL)
        formatted_text = format_conversation_for_display(role, text)

        tag = "assistant" if role == "assistant" else ("user" if role == "user" else "system")
        self.analysis_text.insert(tk.END, formatted_text, tag)
        self.analysis_text.see(tk.END)
        self.analysis_text.config(state=tk.DISABLED)

    def clear_analysis(self):
        """Clear analysis session"""
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Analysis session cleared. Select an image and run analysis to begin.\n", "system")
        self.analysis_text.config(state=tk.DISABLED)
        self.grok_conversation_history = []
        self.analysis_history = []
        self.question_entry.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.pdf_report_btn.config(state=tk.DISABLED)

    def browse_image(self):
        """Browse and load X-ray image"""
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            filename = os.path.basename(file_path)
            self.path_label.config(text=filename, fg=Colors.TEXT_PRIMARY)

            # Get file size
            file_size = os.path.getsize(file_path) / 1024  # KB
            self.file_size_label.config(text=f"{file_size:.1f} KB")
            self.file_icon_label.config(fg=Colors.PRIMARY)

            self._update_status(f"Image loaded: {filename}", "success")
            self.analyze_btn.config(state=tk.NORMAL)
            self.ai_interpretation_btn.config(state=tk.DISABLED)
            self.pdf_report_btn.config(state=tk.DISABLED)

            # Reset results
            self.prediction_badge.config(text="N/A", bg=Colors.BORDER, fg=Colors.TEXT_SECONDARY)
            self.confidence_label.config(text="—")
            self.confidence_progress['value'] = 0
            self.feature_label.config(text="")
            self.clear_analysis()

            try:
                self.current_pil_image = Image.open(file_path)
                self._display_original_image_only()
            except Exception as e:
                messagebox.showerror("Image Display Error", f"Could not display image: {e}")
                self.current_pil_image = None
                self.analyze_btn.config(state=tk.DISABLED)

    def _display_original_image_only(self):
        """Display only the original image before analysis"""
        self.ax_original.clear()
        self.ax_grad_cam.clear()

        self.ax_original.imshow(self.current_pil_image, cmap='gray')
        self.ax_original.set_title(f"X-ray: {os.path.basename(self.current_image_path)}",
                                   fontsize=11, fontweight='bold', color=Colors.TEXT_PRIMARY, pad=10)
        self.ax_original.axis('off')

        self.ax_grad_cam.text(0.5, 0.5, "Run analysis to generate\nGrad-CAM heatmap",
                             ha='center', va='center', fontsize=12, color=Colors.TEXT_MUTED,
                             transform=self.ax_grad_cam.transAxes)
        self.ax_grad_cam.set_title("Grad-CAM Heatmap", fontsize=11, fontweight='bold',
                                   color=Colors.TEXT_SECONDARY, pad=10)
        self.ax_grad_cam.axis('off')

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def _run_advanced_analysis(self):
        """Run enhanced analysis with feature extraction and Grad-CAM"""
        if not self.model:
            messagebox.showerror("Error", "AI Agent not loaded. Cannot analyze.")
            return
        if not self.current_image_path:
            messagebox.showerror("Error", "No image selected.")
            return

        self._update_status("AI Agent analyzing image features and generating Grad-CAM...", "loading")
        self.analyze_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        predicted_class, confidence, original_image, analysis_data, error = predict_pneumonia_advanced(
            self.current_image_path, self.model, self.transform, self.device, CLASS_NAMES
        )

        if error:
            messagebox.showerror("Analysis Error", error)
            self.prediction_badge.config(text="Error", bg=Colors.DANGER, fg=Colors.TEXT_LIGHT)
            self.confidence_label.config(text="—")
            self._update_status(f"Analysis failed: {error}", "error")
            self.analyze_btn.config(state=tk.NORMAL)
            self.ai_interpretation_btn.config(state=tk.DISABLED)
            self.pdf_report_btn.config(state=tk.DISABLED)
        else:
            self.current_prediction = predicted_class
            self.current_confidence = confidence
            self.current_analysis_data = analysis_data

            # Update prediction badge with color coding
            if predicted_class == "Normal":
                self.prediction_badge.config(text="✓ Normal", bg=Colors.NORMAL_BG, fg=Colors.NORMAL_TEXT)
            else:
                self.prediction_badge.config(text="⚠ Pneumonia", bg=Colors.PNEUMONIA_BG, fg=Colors.PNEUMONIA_TEXT)

            # Update confidence display
            confidence_percent = confidence * 100
            self.confidence_label.config(text=f"{confidence_percent:.1f}%")
            self.confidence_progress['value'] = confidence_percent

            # Update feature analysis display
            if analysis_data and analysis_data['image_analysis']:
                features = analysis_data['image_analysis']
                feature_text = f"📊 Image Features:\n"
                feature_text += f"  • Intensity: {features.get('mean_intensity', 'N/A'):.1f}±{features.get('std_intensity', 'N/A'):.1f}\n"
                feature_text += f"  • Edge Density: {features.get('edge_density', 'N/A'):.3f}\n"
                feature_text += f"  • Texture: {features.get('texture_complexity', 'N/A'):.1f}\n"
                feature_text += f"  • Entropy: {features.get('histogram_entropy', 'N/A'):.2f}"
                self.feature_label.config(text=feature_text)

            self._update_status(f"Analysis complete: {predicted_class} ({confidence_percent:.1f}% confidence)", "success")
            self.analyze_btn.config(state=tk.NORMAL)
            self.ai_interpretation_btn.config(state=tk.NORMAL)
            self.pdf_report_btn.config(state=tk.NORMAL)

            # Update image display with Grad-CAM
            self._display_analysis_results(original_image, analysis_data)

            # Store analysis in history
            self.analysis_history.append({
                'prediction': predicted_class,
                'confidence': confidence,
                'features': analysis_data['image_analysis'] if analysis_data else None,
            })

    def _display_analysis_results(self, original_image, analysis_data):
        """Display original image and Grad-CAM heatmap overlay"""
        self.ax_original.clear()
        self.ax_grad_cam.clear()

        # Original image in first subplot
        self.ax_original.imshow(original_image, cmap='gray')

        # Set title with color based on prediction
        title_color = Colors.NORMAL_TEXT if self.current_prediction == "Normal" else Colors.PNEUMONIA_TEXT
        self.ax_original.set_title(
            f"Diagnosis: {self.current_prediction} ({self.current_confidence * 100:.1f}%)",
            fontsize=11, fontweight='bold', color=title_color, pad=10
        )
        self.ax_original.axis('off')

        # Grad-CAM heatmap overlay in second subplot
        if analysis_data and 'grad_cam_heatmap' in analysis_data and analysis_data['grad_cam_heatmap'] is not None:
            heatmap = analysis_data['grad_cam_heatmap']
            original_np = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')) / 255.0

            # Create an image combining original and heatmap for display and saving
            fig_cam, ax_cam = plt.subplots(figsize=(IMAGE_SIZE / 100, IMAGE_SIZE / 100), dpi=100)
            ax_cam.imshow(original_np)
            ax_cam.imshow(heatmap, cmap='jet', alpha=0.5)
            ax_cam.axis('off')
            ax_cam.set_title("Grad-CAM Heatmap")

            # Save the combined image to a temporary file
            temp_cam_file = "temp_grad_cam_overlay.png"
            fig_cam.savefig(temp_cam_file, bbox_inches='tight', pad_inches=0)
            plt.close(fig_cam)

            self.current_grad_cam_overlay_image_path = temp_cam_file

            # Display in GUI
            self.ax_grad_cam.imshow(original_np)
            self.ax_grad_cam.imshow(heatmap, cmap='jet', alpha=0.5)
            self.ax_grad_cam.set_title("Grad-CAM: AI Focus Areas", fontsize=11,
                                       fontweight='bold', color=Colors.TEXT_PRIMARY, pad=10)
            self.ax_grad_cam.axis('off')
        else:
            self.ax_grad_cam.text(0.5, 0.5, "Grad-CAM Heatmap\nNot Available",
                                  ha='center', va='center', fontsize=12, color=Colors.TEXT_MUTED,
                                  transform=self.ax_grad_cam.transAxes)
            self.ax_grad_cam.axis('off')
            self.current_grad_cam_overlay_image_path = None

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def perform_initial_grok_analysis(self):
        """Enhanced initial analysis with feature data and Grad-CAM insights"""
        if not self.current_prediction or not self.current_analysis_data:
            messagebox.showwarning("Warning", "Please run AI analysis first before getting detailed interpretation.")
            return

        self._update_status("AI Agent generating comprehensive analysis...", "loading")
        self.ai_interpretation_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        self.clear_analysis()
        self._update_analysis_text("🔄 AI Agent analyzing findings... Please wait...\n", "system")
        self.root.update_idletasks()

        feature_analysis = self.current_analysis_data['image_analysis'] if self.current_analysis_data else {}

        # Generate initial prompt
        initial_prompt = generate_initial_analysis_prompt(
            self.current_prediction,
            self.current_confidence,
            feature_analysis
        )

        self.grok_conversation_history.append({"role": "user", "content": initial_prompt})

        # Call Grok API
        analysis_result, error = call_grok_api(self.grok_conversation_history, self.rag_retriever)

        if error:
            messagebox.showerror("AI Analysis Error", error)
            self._update_status("AI analysis failed", "error")
            self._update_analysis_text(f"❌ Error: {error}", "system")
            self.ai_interpretation_btn.config(state=tk.NORMAL)
            self.pdf_report_btn.config(state=tk.DISABLED)
        else:
            self._update_status("AI analysis complete. You can now ask follow-up questions.", "success")
            self.grok_conversation_history.append({"role": "assistant", "content": analysis_result})

            # Clear and add the analysis result
            self.analysis_text.config(state=tk.NORMAL)
            self.analysis_text.delete(1.0, tk.END)
            self._update_analysis_text(analysis_result, "assistant")

            self.question_entry.config(state=tk.NORMAL)
            self.send_btn.config(state=tk.NORMAL)
            self.question_entry.focus_set()
            self.pdf_report_btn.config(state=tk.NORMAL)

    def send_question_to_grok(self):
        """Send follow-up question to Grok API"""
        user_question = self.question_entry.get().strip()
        if not user_question:
            messagebox.showwarning("Warning", "Please type a question for the AI Agent.")
            return

        self.question_entry.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self._update_status("Consulting AI Agent...", "loading")
        self.root.update_idletasks()

        self._update_analysis_text(user_question, "user")
        self.question_entry.delete(0, tk.END)

        self.grok_conversation_history.append({"role": "user", "content": user_question})

        grok_response, error = call_grok_api(self.grok_conversation_history, self.rag_retriever)

        if error:
            messagebox.showerror("AI Consultation Error", error)
            self._update_status("AI consultation failed.", "error")
            self._update_analysis_text(f"❌ Error: {error}", "system")
        else:
            self._update_status("AI Agent responded.", "success")
            self.grok_conversation_history.append({"role": "assistant", "content": grok_response})
            self._update_analysis_text(grok_response, "assistant")

        self.question_entry.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.question_entry.focus_set()

    def generate_pdf_report(self):
        """Generate PDF report"""
        if not self.current_image_path or not self.current_prediction or not self.current_analysis_data:
            messagebox.showwarning("Warning", "Please perform a full AI analysis first to generate a report.")
            return

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Pneumonia_Report_{os.path.basename(self.current_image_path).split('.')[0]}.pdf"
        )
        if not file_path:
            return

        self._update_status("Generating PDF report...", "loading")
        self.root.update_idletasks()

        # Generate PDF report
        success, message = create_pdf_report(
            file_path=file_path,
            image_path=self.current_image_path,
            pil_image=self.current_pil_image,
            prediction=self.current_prediction,
            confidence=self.current_confidence,
            features=self.current_analysis_data['image_analysis'] if self.current_analysis_data else {},
            grad_cam_path=self.current_grad_cam_overlay_image_path,
            conversation_history=self.grok_conversation_history
        )

        if success:
            messagebox.showinfo("PDF Report", message)
            self._update_status("PDF report generated successfully.", "success")
        else:
            messagebox.showerror("PDF Error", message)
            self._update_status("PDF generation failed.", "error")
