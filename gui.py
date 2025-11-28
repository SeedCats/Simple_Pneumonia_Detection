"""
GUI Module
Handles all user interface components and event handlers
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
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


class PneumoniaApp:
    """Main GUI application for pneumonia detection"""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

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
        self.status_label.config(text="Loading AI model...")
        self.root.update_idletasks()
        self.model = load_trained_model(MODEL_PATH, NUM_CLASSES, self.device)
        if self.model:
            self.status_label.config(text=f"AI Agent ready. Using {self.device}.")
            self.initial_grok_button.config(state=tk.DISABLED)
        else:
            self.status_label.config(text="AI Agent loading failed. Check console for errors.")
            self.browse_button.config(state=tk.DISABLED)
            self.initial_grok_button.config(state=tk.DISABLED)

    def setup_ui(self):
        """Setup all UI components"""
        self._setup_frames()
        self._setup_control_panel()
        self._setup_image_display()
        self._setup_analysis_panel()

    def _setup_frames(self):
        """Setup main layout frames"""
        self.control_frame = tk.Frame(self.root, padx=10, pady=10, width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False)

        self.image_frame = tk.Frame(self.root, bg="lightgray", padx=10, pady=10)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.analysis_frame = tk.Frame(self.root, padx=10, pady=10, width=500)
        self.analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.analysis_frame.pack_propagate(False)

    def _setup_control_panel(self):
        """Setup left control panel"""
        tk.Label(self.control_frame, text="AI Pneumonia Agent", font=("Arial", 16, "bold")).pack(pady=10)

        self.path_label = tk.Label(self.control_frame, text="No image selected", wraplength=280)
        self.path_label.pack(pady=5)

        self.browse_button = tk.Button(self.control_frame, text="Browse X-ray Image", command=self.browse_image,
                                       width=20)
        self.browse_button.pack(pady=10)

        self.analyze_button = tk.Button(self.control_frame, text="Run AI Analysis", command=self._run_advanced_analysis,
                                        state=tk.DISABLED, width=20, bg="lightgreen")
        self.analyze_button.pack(pady=10)

        # Result display
        result_frame = tk.Frame(self.control_frame)
        result_frame.pack(pady=10, fill=tk.X)

        tk.Label(result_frame, text="AI Diagnosis:", font=("Arial", 12, "bold")).pack()
        self.prediction_label = tk.Label(result_frame, text="N/A", font=("Arial", 14), fg="blue")
        self.prediction_label.pack()

        tk.Label(result_frame, text="Confidence Score:", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.confidence_label = tk.Label(result_frame, text="N/A", font=("Arial", 14), fg="green")
        self.confidence_label.pack()

        # Feature analysis display
        self.feature_label = tk.Label(self.control_frame, text="", wraplength=280, justify=tk.LEFT, font=("Arial", 9))
        self.feature_label.pack(pady=10)

        # Grok Analysis Button
        self.initial_grok_button = tk.Button(self.control_frame, text="Get Detailed AI Interpretation",
                                             command=self.perform_initial_grok_analysis,
                                             state=tk.DISABLED, width=25, bg="lightblue")
        self.initial_grok_button.pack(pady=20)

        # Generate PDF Report Button
        self.pdf_report_button = tk.Button(self.control_frame, text="Generate PDF Report",
                                           command=self.generate_pdf_report,
                                           state=tk.DISABLED, width=25, bg="lightcoral")
        self.pdf_report_button.pack(pady=10)

        # Status bar
        self.status_label = tk.Label(self.root, text="AI Agent Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_image_display(self):
        """Setup image display using Matplotlib"""
        self.fig, (self.ax_original, self.ax_grad_cam) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax_original.axis('off')
        self.ax_grad_cam.axis('off')
        self.ax_original.set_title("Original X-ray Image")
        self.ax_grad_cam.set_title("Grad-CAM Heatmap (Available after analysis)")
        self.fig.tight_layout()
        self.canvas.draw()

    def _setup_analysis_panel(self):
        """Setup right analysis panel"""
        tk.Label(self.analysis_frame, text="AI Agent Analysis & Consultation", font=("Arial", 16, "bold")).pack(pady=10)

        self.analysis_text = scrolledtext.ScrolledText(
            self.analysis_frame,
            wrap=tk.WORD,
            width=60,
            height=20,
            font=("Arial", 10)
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.analysis_text.insert(tk.END, "AI Agent analysis will appear here. The agent will provide:\n"
                                          "- Direct model predictions\n"
                                          "- Feature analysis\n"
                                          "- Clinical interpretation\n"
                                          "- Follow-up recommendations\n")
        self.analysis_text.config(state=tk.DISABLED)

        # Follow-up question input
        question_frame = tk.Frame(self.analysis_frame)
        question_frame.pack(fill=tk.X, pady=10)

        tk.Label(question_frame, text="Ask AI Agent:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.grok_question_entry = tk.Entry(question_frame, width=40)
        self.grok_question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.grok_question_entry.config(state=tk.DISABLED)

        self.send_grok_question_button = tk.Button(
            question_frame,
            text="Consult",
            command=self.send_question_to_grok,
            state=tk.DISABLED
        )
        self.send_grok_question_button.pack(side=tk.LEFT, padx=(5, 0))

        # Clear Analysis Button
        self.clear_analysis_button = tk.Button(
            self.analysis_frame,
            text="Clear Analysis Session",
            command=self.clear_analysis,
            width=25
        )
        self.clear_analysis_button.pack(pady=10)

    def _update_analysis_text(self, text, role):
        """Helper to append text to the scrolledtext widget with role indication"""
        self.analysis_text.config(state=tk.NORMAL)
        formatted_text = format_conversation_for_display(role, text)
        self.analysis_text.insert(tk.END, formatted_text)
        self.analysis_text.see(tk.END)
        self.analysis_text.config(state=tk.DISABLED)

    def clear_analysis(self):
        """Clear analysis session"""
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Analysis session cleared. Run new analysis to begin.")
        self.analysis_text.config(state=tk.DISABLED)
        self.grok_conversation_history = []
        self.analysis_history = []
        self.grok_question_entry.config(state=tk.DISABLED)
        self.send_grok_question_button.config(state=tk.DISABLED)
        self.pdf_report_button.config(state=tk.DISABLED)

    def browse_image(self):
        """Browse and load X-ray image"""
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.status_label.config(text=f"Image loaded. Click 'Run AI Analysis' for detailed assessment.")
            self.analyze_button.config(state=tk.NORMAL)
            self.initial_grok_button.config(state=tk.DISABLED)
            self.pdf_report_button.config(state=tk.DISABLED)
            self.prediction_label.config(text="N/A")
            self.confidence_label.config(text="N/A")
            self.feature_label.config(text="")
            self.clear_analysis()

            try:
                self.current_pil_image = Image.open(file_path)
                self._display_original_image_only()
            except Exception as e:
                messagebox.showerror("Image Display Error", f"Could not display image: {e}")
                self.current_pil_image = None
                self.analyze_button.config(state=tk.DISABLED)

    def _display_original_image_only(self):
        """Display only the original image before analysis"""
        self.ax_original.clear()
        self.ax_grad_cam.clear()
        self.ax_original.imshow(self.current_pil_image, cmap='gray')
        self.ax_original.set_title(f"X-ray: {os.path.basename(self.current_image_path)}")
        self.ax_original.axis('off')
        self.ax_grad_cam.axis('off')
        self.ax_grad_cam.set_title("Grad-CAM Heatmap (Available after analysis)")
        self.fig.tight_layout()
        self.canvas.draw()

    def _run_advanced_analysis(self):
        """Run enhanced analysis with feature extraction and Grad-CAM"""
        if not self.model:
            messagebox.showerror("Error", "AI Agent not loaded. Cannot analyze.")
            return
        if not self.current_image_path:
            messagebox.showerror("Error", "No image selected.")
            return

        self.status_label.config(text="AI Agent analyzing image features and generating Grad-CAM...")
        self.root.update_idletasks()

        predicted_class, confidence, original_image, analysis_data, error = predict_pneumonia_advanced(
            self.current_image_path, self.model, self.transform, self.device, CLASS_NAMES
        )

        if error:
            messagebox.showerror("Analysis Error", error)
            self.prediction_label.config(text="Error")
            self.confidence_label.config(text="Error")
            self.status_label.config(text=f"Analysis failed: {error}")
            self.initial_grok_button.config(state=tk.DISABLED)
            self.pdf_report_button.config(state=tk.DISABLED)
        else:
            self.current_prediction = predicted_class
            self.current_confidence = confidence
            self.current_analysis_data = analysis_data

            # Update basic prediction display
            self.prediction_label.config(text=predicted_class)
            confidence_percent = confidence * 100
            self.confidence_label.config(text=f"{confidence_percent:.1f}%")

            # Update feature analysis display
            if analysis_data and analysis_data['image_analysis']:
                features = analysis_data['image_analysis']
                feature_text = f"Image Analysis:\n"
                feature_text += f"• Mean Intensity (0-255): {features.get('mean_intensity', 'N/A'):.1f}±{features.get('std_intensity', 'N/A'):.1f}\n"
                feature_text += f"• Edge Density (0-1): {features.get('edge_density', 'N/A'):.3f}\n"
                feature_text += f"• Texture Complexity (Sobel std): {features.get('texture_complexity', 'N/A'):.1f}\n"
                feature_text += f"• Hist. Entropy: {features.get('histogram_entropy', 'N/A'):.2f}"
                self.feature_label.config(text=feature_text)

            self.status_label.config(
                text=f"AI Analysis complete. {predicted_class} with {confidence_percent:.1f}% confidence. Ready for AI Interpretation.")
            self.initial_grok_button.config(state=tk.NORMAL)
            self.pdf_report_button.config(state=tk.NORMAL)

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
        self.ax_original.set_title(
            f"Diagnosis: {self.current_prediction}\nConfidence: {self.current_confidence * 100:.1f}%")
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
            self.ax_grad_cam.set_title("Grad-CAM Heatmap (Areas of AI Focus)")
            self.ax_grad_cam.axis('off')
        else:
            self.ax_grad_cam.text(0.5, 0.5, "Grad-CAM Heatmap\nNot Available",
                                  ha='center', va='center', transform=self.ax_grad_cam.transAxes, fontsize=12)
            self.ax_grad_cam.axis('off')
            self.current_grad_cam_overlay_image_path = None

        self.fig.tight_layout()
        self.canvas.draw()

    def perform_initial_grok_analysis(self):
        """Enhanced initial analysis with feature data and Grad-CAM insights"""
        if not self.current_prediction or not self.current_analysis_data:
            messagebox.showwarning("Warning", "Please run AI analysis first before getting detailed interpretation.")
            return

        self.status_label.config(text="AI Agent generating comprehensive analysis...")
        self.initial_grok_button.config(state=tk.DISABLED)
        self.root.update_idletasks()

        self.clear_analysis()
        self._update_analysis_text("AI Agent analyzing findings... Please wait...", "system")
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
            self.status_label.config(text="AI analysis failed")
            self._update_analysis_text(f"Error: {error}", "system")
            self.initial_grok_button.config(state=tk.NORMAL)
            self.pdf_report_button.config(state=tk.DISABLED)
        else:
            self.status_label.config(text="AI analysis complete. You can now ask follow-up questions.")
            self.grok_conversation_history.append({"role": "assistant", "content": analysis_result})
            self._update_analysis_text(analysis_result, "assistant")

            self.grok_question_entry.config(state=tk.NORMAL)
            self.send_grok_question_button.config(state=tk.NORMAL)
            self.grok_question_entry.focus_set()
            self.pdf_report_button.config(state=tk.NORMAL)

    def send_question_to_grok(self):
        """Send follow-up question to Grok API"""
        user_question = self.grok_question_entry.get().strip()
        if not user_question:
            messagebox.showwarning("Warning", "Please type a question for the AI Agent.")
            return

        self.grok_question_entry.config(state=tk.DISABLED)
        self.send_grok_question_button.config(state=tk.DISABLED)
        self.status_label.config(text="Consulting AI Agent...")
        self.root.update_idletasks()

        self._update_analysis_text(user_question, "user")
        self.grok_question_entry.delete(0, tk.END)

        self.grok_conversation_history.append({"role": "user", "content": user_question})

        grok_response, error = call_grok_api(self.grok_conversation_history, self.rag_retriever)

        if error:
            messagebox.showerror("AI Consultation Error", error)
            self.status_label.config(text="AI consultation failed.")
            self._update_analysis_text(f"Error: {error}", "system")
        else:
            self.status_label.config(text="AI Agent responded.")
            self.grok_conversation_history.append({"role": "assistant", "content": grok_response})
            self._update_analysis_text(grok_response, "assistant")

        self.grok_question_entry.config(state=tk.NORMAL)
        self.send_grok_question_button.config(state=tk.NORMAL)
        self.grok_question_entry.focus_set()

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

        self.status_label.config(text="Generating PDF report...")
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
            self.status_label.config(text="PDF report generated.")
        else:
            messagebox.showerror("PDF Generation Error", message)
            self.status_label.config(text="Failed to generate PDF report.")

