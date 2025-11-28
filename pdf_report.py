"""
PDF Report Generation Module
Handles comprehensive PDF report creation with images and analysis
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image as PILImage
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from config import (
    IMAGE_SIZE, PDF_DISCLAIMER, TEXT_REPLACEMENTS,
    TEMP_IMAGE_PATH
)


class PneumoniaReportGenerator:
    """Generate comprehensive PDF reports for pneumonia analysis"""

    def __init__(self):
        self.pdf = None
        self.font_family = 'helvetica'

    def generate_report(self,
                       file_path: str,
                       image_path: str,
                       pil_image: PILImage.Image,
                       prediction: str,
                       confidence: float,
                       features: Dict,
                       grad_cam_path: Optional[str],
                       conversation_history: List[Dict]) -> tuple:
        """
        Generate comprehensive PDF report

        Args:
            file_path: Output PDF file path
            image_path: Original X-ray image path
            pil_image: PIL Image object
            prediction: Model prediction (Normal/Pneumonia)
            confidence: Confidence score (0-1)
            features: Dictionary of image features
            grad_cam_path: Path to Grad-CAM image (if available)
            conversation_history: List of conversation messages

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            self.pdf = FPDF()
            self.pdf.add_page()
            self.pdf.set_auto_page_break(auto=True, margin=15)

            # Determine font
            self._setup_fonts()

            # Clean conversation history
            cleaned_conversation = self._clean_conversation(conversation_history)

            # Add sections
            self._add_title()
            self._add_image_information(image_path)
            self._add_xray_images(pil_image, grad_cam_path)
            self._add_diagnosis_summary(prediction, confidence)
            self._add_image_features(features)
            self._add_consultation_log(cleaned_conversation)
            self._add_disclaimer()

            # Save PDF
            self.pdf.output(file_path)
            return True, f"PDF report successfully saved to:\n{file_path}"

        except Exception as e:
            import traceback
            print(f"PDF generation error details: {traceback.format_exc()}")
            return False, f"Failed to generate PDF report: {e}"

        finally:
            self._cleanup_temp_files()

    def _setup_fonts(self):
        """Setup fonts - try Unicode fonts first, fall back to basic"""
        try:
            font_dir = "fonts"
            dejavu_sans_path = os.path.join(font_dir, "DejaVuSans.ttf")
            dejavu_sans_bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")

            if os.path.exists(dejavu_sans_path) and os.path.exists(dejavu_sans_bold_path):
                self.pdf.add_font('DejaVu', '', dejavu_sans_path, uni=True)
                self.pdf.add_font('DejaVu', 'B', dejavu_sans_bold_path, uni=True)
                self.font_family = 'DejaVu'
                print("Using DejaVu Unicode fonts")
                return

            raise FileNotFoundError("DejaVu fonts not found")

        except:
            self.font_family = 'helvetica'
            print("Using basic Helvetica font with character filtering")

    def _add_title(self):
        """Add report title and date"""
        self.pdf.set_font(self.font_family, "B", 16)
        self.pdf.cell(0, 10, "AI-Assisted Pneumonia X-ray Analysis Report",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        self.pdf.set_font(self.font_family, "", 10)
        now = datetime.now()
        # Format: Day, Month Date, Year at HH:MM:SS AM/PM (e.g., Thursday, November 28, 2025 at 2:30:45 PM)
        report_date = now.strftime('%A, %B %d, %Y at %I:%M:%S %p')
        self.pdf.cell(0, 5, f"Date: {report_date}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.pdf.ln(10)

    def _add_image_information(self, image_path: str):
        """Add image file information"""
        self.pdf.set_font(self.font_family, "B", 12)
        self.pdf.cell(0, 8, "1. Image Information",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.pdf.set_font(self.font_family, "", 10)
        self.pdf.multi_cell(0, 5, f"Original Image File: {os.path.basename(image_path)}")
        self.pdf.ln(5)

    def _add_xray_images(self, pil_image: PILImage.Image, grad_cam_path: Optional[str]):
        """Add X-ray and Grad-CAM images"""
        self.pdf.set_font(self.font_family, "B", 12)
        self.pdf.cell(0, 8, "2. X-ray Images",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Save original image temporarily
        temp_original_file = TEMP_IMAGE_PATH
        pil_image.save(temp_original_file)

        # Image dimensions
        page_width = self.pdf.w - 2 * self.pdf.l_margin
        img_width = min(80, page_width - 20)
        img_height = 60

        # Original Image
        self.pdf.set_font(self.font_family, "", 10)
        self.pdf.cell(0, 5, "Original X-ray Image:",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        if os.path.exists(temp_original_file):
            x_original = (self.pdf.w - img_width) / 2
            self.pdf.image(temp_original_file, x=x_original, y=self.pdf.get_y(),
                          w=img_width, h=img_height)
            self.pdf.ln(img_height + 5)

        # Grad-CAM Image (if available)
        if grad_cam_path and os.path.exists(grad_cam_path):
            self.pdf.cell(0, 5, "Grad-CAM Heatmap Overlay:",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            x_gradcam = (self.pdf.w - img_width) / 2
            self.pdf.image(grad_cam_path, x=x_gradcam, y=self.pdf.get_y(),
                          w=img_width, h=img_height)
            self.pdf.ln(img_height + 5)

        self.pdf.ln(10)

    def _add_diagnosis_summary(self, prediction: str, confidence: float):
        """Add AI diagnosis summary"""
        self.pdf.set_font(self.font_family, "B", 12)
        self.pdf.cell(0, 8, "3. AI Diagnosis Summary",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.pdf.set_font(self.font_family, "", 10)
        self.pdf.cell(0, 5, f"Predicted Class: {prediction}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.cell(0, 5, f"Confidence Score: {confidence * 100:.2f}%",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.ln(5)

    def _add_image_features(self, features: Dict):
        """Add quantitative image features"""
        self.pdf.set_font(self.font_family, "B", 12)
        self.pdf.cell(0, 8, "4. Quantitative Image Features",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.pdf.set_font(self.font_family, "", 10)

        if features:
            feature_text = f"""Mean Intensity (0-255): {features.get('mean_intensity', 'N/A'):.1f}
Intensity Standard Deviation: {features.get('std_intensity', 'N/A'):.1f}
Edge Density (0-1): {features.get('edge_density', 'N/A'):.4f}
Texture Complexity (Sobel std): {features.get('texture_complexity', 'N/A'):.1f}
Histogram Entropy: {features.get('histogram_entropy', 'N/A'):.2f}"""
            self.pdf.multi_cell(0, 5, feature_text)
        else:
            self.pdf.multi_cell(0, 5, "No quantitative image analysis data available.")

        self.pdf.ln(5)

    def _add_consultation_log(self, conversation: List[Dict]):
        """Add AI agent consultation log"""
        if not conversation:
            return

        self.pdf.set_font(self.font_family, "B", 12)
        self.pdf.cell(0, 8, "5. AI Agent Consultation Log",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font(self.font_family, "", 10)

        for i, message in enumerate(conversation):
            role = message['role'].capitalize()
            content = message['content']

            if i > 0:
                self.pdf.ln(3)

            self.pdf.set_font(self.font_family, "B", 10)
            self.pdf.cell(0, 6, f"[{role}]:",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.pdf.set_font(self.font_family, "", 10)

            paragraphs = content.split('\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    self.pdf.multi_cell(0, 5, paragraph.strip())
                    self.pdf.ln(1)

    def _add_disclaimer(self):
        """Add disclaimer footer"""
        self.pdf.ln(10)
        self.pdf.set_font(self.font_family, "I", 8)
        self.pdf.multi_cell(0, 4, PDF_DISCLAIMER)

    def _clean_conversation(self, conversation: List[Dict]) -> List[Dict]:
        """Clean conversation text for PDF compatibility"""
        cleaned = []
        for message in conversation:
            cleaned_content = self._clean_text_for_pdf(message['content'])
            cleaned.append({
                'role': message['role'],
                'content': cleaned_content
            })
        return cleaned

    def _clean_text_for_pdf(self, text: str, aggressive: bool = False) -> str:
        """Clean text to remove characters not supported by basic PDF fonts"""
        if not text:
            return ""

        cleaned_text = text
        for old, new in TEXT_REPLACEMENTS.items():
            cleaned_text = cleaned_text.replace(old, new)

        if aggressive or self.font_family == 'helvetica':
            # For basic fonts, remove any non-ASCII characters
            cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')

        return cleaned_text

    def _cleanup_temp_files(self):
        """Clean up temporary image files"""
        temp_files = [TEMP_IMAGE_PATH]

        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✓ Cleanup: Deleted {file_path}")
            except Exception as e:
                print(f"Cleanup warning: Could not delete {file_path}: {e}")


def create_pdf_report(file_path: str,
                     image_path: str,
                     pil_image: PILImage.Image,
                     prediction: str,
                     confidence: float,
                     features: Dict,
                     grad_cam_path: Optional[str],
                     conversation_history: List[Dict]) -> tuple:
    """
    Convenience function to create PDF report

    Args:
        file_path: Output PDF file path
        image_path: Original X-ray image path
        pil_image: PIL Image object
        prediction: Model prediction (Normal/Pneumonia)
        confidence: Confidence score (0-1)
        features: Dictionary of image features
        grad_cam_path: Path to Grad-CAM image (if available)
        conversation_history: List of conversation messages

    Returns:
        Tuple of (success: bool, message: str)
    """
    generator = PneumoniaReportGenerator()
    return generator.generate_report(
        file_path, image_path, pil_image, prediction,
        confidence, features, grad_cam_path, conversation_history
    )

