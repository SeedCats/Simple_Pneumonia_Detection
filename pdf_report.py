"""
PDF Report Generation Module - Professional medical report format
"""

import os
import re
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image as PILImage
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from config import PDF_DISCLAIMER, TEXT_REPLACEMENTS, TEMP_IMAGE_PATH


class PneumoniaReportGenerator:
    """Generate comprehensive PDF reports for pneumonia analysis"""

    COLORS = {
        'primary': (37, 99, 235), 'success': (22, 163, 74), 'danger': (220, 38, 38),
        'warning': (217, 119, 6), 'text_dark': (30, 41, 59), 'text_light': (100, 116, 139),
        'bg_light': (248, 250, 252), 'border': (226, 232, 240),
    }

    def __init__(self):
        self.pdf = None
        self.font_family = 'helvetica'

    def generate_report(self, file_path: str, image_path: str, pil_image: PILImage.Image,
                       prediction: str, confidence: float, features: Dict,
                       grad_cam_path: Optional[str], conversation_history: List[Dict]) -> tuple:
        """Generate comprehensive PDF report"""
        try:
            self.pdf = FPDF()
            self.pdf.set_auto_page_break(auto=True, margin=15)
            self._setup_fonts()

            insights = self._extract_insights(conversation_history)

            self._add_cover_page(image_path, prediction, confidence)
            self.pdf.add_page()
            self._add_images_section(pil_image, grad_cam_path)
            self._add_features_section(features)
            self.pdf.add_page()
            self._add_clinical_interpretation(insights, prediction, confidence)
            self._add_recommendations_section(insights, prediction)
            self._add_disclaimer_section()

            self.pdf.output(file_path)
            return True, f"PDF report saved to:\n{file_path}"
        except Exception as e:
            return False, f"Failed to generate PDF: {e}"
        finally:
            self._cleanup_temp_files()

    def _setup_fonts(self):
        """Setup fonts"""
        try:
            font_dir = "fonts"
            if os.path.exists(f"{font_dir}/DejaVuSans.ttf"):
                self.pdf.add_font('DejaVu', '', f"{font_dir}/DejaVuSans.ttf", uni=True)
                self.pdf.add_font('DejaVu', 'B', f"{font_dir}/DejaVuSans-Bold.ttf", uni=True)
                self.font_family = 'DejaVu'
        except:
            self.font_family = 'helvetica'

    def _set_color(self, name: str):
        self.pdf.set_text_color(*self.COLORS.get(name, self.COLORS['text_dark']))

    def _set_fill(self, name: str):
        self.pdf.set_fill_color(*self.COLORS.get(name, self.COLORS['bg_light']))

    def _set_draw(self, name: str):
        self.pdf.set_draw_color(*self.COLORS.get(name, self.COLORS['border']))

    def _add_cover_page(self, image_path: str, prediction: str, confidence: float):
        """Add cover page"""
        self.pdf.add_page()
        is_pneumonia = prediction.lower() == "pneumonia"

        # Header
        self._set_fill('primary')
        self.pdf.rect(0, 0, 210, 45, 'F')
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font(self.font_family, "B", 24)
        self.pdf.set_xy(0, 12)
        self.pdf.cell(0, 10, "Pneumonia Detection Report", align="C")
        self.pdf.set_font(self.font_family, "", 11)
        self.pdf.set_xy(0, 26)
        self.pdf.cell(0, 6, "AI-Assisted Chest X-ray Analysis", align="C")

        # Metadata box
        self._set_fill('bg_light')
        self._set_draw('border')
        self.pdf.rect(10, 55, 190, 35, 'DF')
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 11)
        self.pdf.set_xy(15, 60)
        self.pdf.cell(0, 6, "Report Information")
        self.pdf.set_font(self.font_family, "", 10)
        self._set_color('text_light')
        self.pdf.set_xy(15, 68)
        self.pdf.cell(60, 5, "Generated:")
        self._set_color('text_dark')
        self.pdf.cell(0, 5, datetime.now().strftime('%B %d, %Y at %I:%M %p'))
        self._set_color('text_light')
        self.pdf.set_xy(15, 75)
        self.pdf.cell(60, 5, "Image File:")
        self._set_color('text_dark')
        self.pdf.cell(0, 5, os.path.basename(image_path))

        # Diagnosis result
        self.pdf.set_xy(10, 100)
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 14)
        self.pdf.cell(0, 10, "DIAGNOSIS RESULT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self._set_fill('danger' if is_pneumonia else 'success')
        self.pdf.rect(10, 115, 190, 50, 'F')
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font(self.font_family, "B", 28)
        self.pdf.set_xy(10, 125)
        self.pdf.cell(190, 15, "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL", align="C")
        self.pdf.set_font(self.font_family, "", 14)
        self.pdf.set_xy(10, 143)
        self.pdf.cell(190, 10, f"Confidence: {confidence * 100:.1f}%", align="C")

        # Confidence interpretation
        self.pdf.set_xy(10, 175)
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 12)
        self.pdf.cell(0, 8, "Confidence Level Interpretation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font(self.font_family, "", 10)
        self._set_color('text_light')
        conf_texts = {0.9: "Very High", 0.75: "High", 0.6: "Moderate"}
        level = next((t for thresh, t in conf_texts.items() if confidence >= thresh), "Low")
        self.pdf.multi_cell(190, 5, f"{level} Confidence - {'Clinical verification advised.' if level == 'Low' else 'AI model shows certainty.'}")

        # Summary box
        self.pdf.set_xy(10, 200)
        self._set_fill('bg_light')
        self._set_draw('border')
        self.pdf.rect(10, 200, 190, 60, 'DF')
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 11)
        self.pdf.set_xy(15, 205)
        self.pdf.cell(0, 6, "Quick Summary")
        self.pdf.set_font(self.font_family, "", 10)
        self._set_color('text_light')
        self.pdf.set_xy(15, 215)
        summary = ("Patterns consistent with pneumonia detected. Review by healthcare professional recommended."
                  if is_pneumonia else "Normal lung patterns. No significant abnormalities detected.")
        self.pdf.multi_cell(180, 5, summary)

    def _add_images_section(self, pil_image: PILImage.Image, grad_cam_path: Optional[str]):
        """Add X-ray images section"""
        self._add_section_header("1", "X-ray Image Analysis")
        pil_image.save(TEMP_IMAGE_PATH)

        y = self.pdf.get_y() + 5
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 10)

        self.pdf.set_xy(10, y)
        self.pdf.cell(85, 6, "Original X-ray", align="C")
        if os.path.exists(TEMP_IMAGE_PATH):
            self.pdf.image(TEMP_IMAGE_PATH, x=10, y=y + 8, w=85, h=70)

        self.pdf.set_xy(105, y)
        self.pdf.cell(85, 6, "AI Focus Areas (Grad-CAM)" if grad_cam_path else "Grad-CAM Not Available", align="C")
        if grad_cam_path and os.path.exists(grad_cam_path):
            self.pdf.image(grad_cam_path, x=105, y=y + 8, w=85, h=70)

        self.pdf.set_xy(10, y + 90)
        self._set_fill('bg_light')
        self.pdf.rect(10, self.pdf.get_y(), 190, 25, 'F')
        self.pdf.set_xy(15, self.pdf.get_y() + 3)
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 9)
        self.pdf.cell(0, 5, "About Grad-CAM", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font(self.font_family, "", 9)
        self._set_color('text_light')
        self.pdf.set_x(15)
        self.pdf.multi_cell(180, 4, "Grad-CAM highlights regions influencing AI's decision. Red/yellow = high importance.")
        self.pdf.ln(10)

    def _add_features_section(self, features: Dict):
        """Add quantitative features section"""
        self._add_section_header("2", "Quantitative Image Analysis")
        if not features:
            self._set_color('text_light')
            self.pdf.multi_cell(0, 5, "No analysis data available.")
            return

        data = [
            ("Mean Intensity", f"{features.get('mean_intensity', 0):.1f}", "Average brightness (0-255)"),
            ("Intensity Std Dev", f"{features.get('std_intensity', 0):.1f}", "Brightness variation"),
            ("Edge Density", f"{features.get('edge_density', 0):.4f}", "Detected edges (0-1)"),
            ("Texture Complexity", f"{features.get('texture_complexity', 0):.1f}", "Pattern variation"),
            ("Histogram Entropy", f"{features.get('histogram_entropy', 0):.2f}", "Information content"),
        ]

        # Header
        self._set_fill('primary')
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font(self.font_family, "B", 9)
        self.pdf.cell(65, 12, "  Metric", border=1, fill=True)
        self.pdf.cell(30, 12, "Value", border=1, fill=True, align="C")
        self.pdf.cell(95, 12, "Description", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Rows
        for i, (metric, value, desc) in enumerate(data):
            self.pdf.set_fill_color(248, 250, 252) if i % 2 == 0 else self.pdf.set_fill_color(255, 255, 255)
            self._set_color('text_dark')
            self.pdf.set_font(self.font_family, "", 9)
            self.pdf.cell(65, 12, f"  {metric}", border=1, fill=True)
            self.pdf.set_font(self.font_family, "B", 9)
            self.pdf.cell(30, 12, value, border=1, fill=True, align="C")
            self.pdf.set_font(self.font_family, "", 9)
            self._set_color('text_light')
            self.pdf.cell(95, 12, desc, border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.ln(5)

    def _add_clinical_interpretation(self, insights: Dict, prediction: str, confidence: float):
        """Add clinical interpretation section"""
        self._add_section_header("3", "Clinical Interpretation")
        is_pneumonia = prediction.lower() == "pneumonia"

        # Findings
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 11)
        self.pdf.cell(0, 8, "Primary Findings", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font(self.font_family, "", 10)
        self._set_color('text_light')

        findings = insights.get('findings') or self._get_defaults('findings', is_pneumonia, confidence)
        for f in findings[:5]:
            self.pdf.set_x(15)
            self.pdf.multi_cell(180, 5, f"- {f}")
        self.pdf.ln(5)

        # Differentials
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 11)
        self.pdf.cell(0, 8, "Differential Considerations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font(self.font_family, "", 10)
        self._set_color('text_light')

        diffs = insights.get('differentials') or self._get_defaults('differentials', is_pneumonia)
        for d in diffs[:4]:
            self.pdf.set_x(15)
            self.pdf.multi_cell(180, 5, f"- {d}")
        self.pdf.ln(5)

        # Limitations box
        box_y = self.pdf.get_y()
        self.pdf.set_fill_color(254, 243, 199)
        self._set_draw('warning')
        self.pdf.rect(10, box_y, 190, 35, 'DF')
        self.pdf.set_text_color(217, 119, 6)
        self.pdf.set_font(self.font_family, "B", 10)
        self.pdf.set_xy(15, box_y + 5)
        self.pdf.cell(0, 6, "Important Limitations")
        self.pdf.set_font(self.font_family, "", 9)
        self.pdf.set_text_color(146, 64, 14)
        self.pdf.set_xy(15, box_y + 14)
        self.pdf.multi_cell(180, 4, "AI analysis should not replace clinical judgment. Patient history and other tests are essential.")
        self.pdf.set_y(box_y + 40)  # Move past the box

    def _add_recommendations_section(self, insights: Dict, prediction: str):
        """Add recommendations section"""
        self._add_section_header("4", "Recommendations")
        is_pneumonia = prediction.lower() == "pneumonia"

        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 11)
        self.pdf.cell(0, 8, "Suggested Next Steps", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self._set_color('text_light')
        self.pdf.set_font(self.font_family, "", 10)

        recs = insights.get('recommendations') or self._get_defaults('recommendations', is_pneumonia)
        for i, rec in enumerate(recs[:6], 1):
            self.pdf.set_x(15)
            self.pdf.multi_cell(180, 6, f"{i}. {rec}")
        self.pdf.ln(5)

        # Follow-up
        self._set_fill('bg_light')
        self.pdf.rect(10, self.pdf.get_y(), 190, 25, 'F')
        self._set_color('text_dark')
        self.pdf.set_font(self.font_family, "B", 10)
        self.pdf.set_xy(15, self.pdf.get_y() + 3)
        self.pdf.cell(0, 6, "Follow-up Guidance")
        self.pdf.set_font(self.font_family, "", 9)
        self._set_color('text_light')
        self.pdf.set_xy(15, self.pdf.get_y() + 8)
        self.pdf.multi_cell(180, 4, "Consult healthcare provider." if is_pneumonia else "Continue routine monitoring.")

    def _add_disclaimer_section(self):
        """Add disclaimer"""
        self.pdf.ln(15)
        self._set_draw('border')
        self.pdf.line(10, self.pdf.get_y(), 200, self.pdf.get_y())
        self.pdf.ln(5)
        self._set_color('text_light')
        self.pdf.set_font(self.font_family, "I", 8)
        self.pdf.multi_cell(0, 4, PDF_DISCLAIMER)

    def _add_section_header(self, number: str, title: str):
        """Add section header"""
        self._set_color('primary')
        self.pdf.set_font(self.font_family, "B", 14)
        self.pdf.cell(0, 10, f"{number}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self._set_draw('primary')
        self.pdf.line(10, self.pdf.get_y(), 60, self.pdf.get_y())
        self.pdf.ln(5)

    def _extract_insights(self, conversation: List[Dict]) -> Dict:
        """Extract key insights from AI conversation"""
        insights = {'findings': [], 'differentials': [], 'recommendations': []}
        if not conversation:
            return insights

        text = ' '.join(m.get('content', '') for m in conversation if m.get('role') == 'assistant')
        if not text:
            return insights

        text = self._clean_text(text)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        keywords = {
            'findings': ['finding', 'shows', 'indicates', 'detected', 'opacity', 'consolidation', 'pattern'],
            'differentials': ['differential', 'bacterial', 'viral', 'consider', 'rule out'],
            'recommendations': ['recommend', 'suggest', 'advise', 'should', 'consult', 'monitor']
        }

        for key, kws in keywords.items():
            for s in sentences:
                if 20 < len(s) < 200 and any(kw in s.lower() for kw in kws):
                    clean = re.sub(r'\[Ref \d+\]|\*+', '', s).strip()
                    if clean and clean not in insights[key]:
                        insights[key].append(clean)
                        if len(insights[key]) >= (5 if key == 'recommendations' else 4):
                            break
        return insights

    def _get_defaults(self, category: str, is_pneumonia: bool, confidence: float = 0) -> List[str]:
        """Get default content"""
        defaults = {
            'findings': {
                True: [f"Patterns consistent with pneumonia ({confidence*100:.1f}% confidence).",
                       "Potential opacity in lung fields.", "Grad-CAM highlights concern regions."],
                False: [f"Normal lung patterns ({confidence*100:.1f}% confidence).",
                        "No significant opacities detected.", "Lung fields appear clear."]
            },
            'differentials': {
                True: ["Bacterial pneumonia", "Viral pneumonia", "Atypical pneumonia", "Pulmonary edema"],
                False: ["Early-stage infection", "Subclinical conditions", "Image quality factors"]
            },
            'recommendations': {
                True: ["Seek healthcare evaluation", "Correlate with symptoms", "Consider blood work",
                       "Review patient history", "Monitor oxygen levels"],
                False: ["Continue health monitoring", "Report new symptoms", "Maintain preventive measures"]
            }
        }
        return defaults.get(category, {}).get(is_pneumonia, [])

    def _clean_text(self, text: str) -> str:
        """Clean text for PDF"""
        for old, new in TEXT_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text.encode('ascii', 'ignore').decode('ascii') if self.font_family == 'helvetica' else text

    def _cleanup_temp_files(self):
        """Clean up temp files"""
        try:
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
        except:
            pass


def create_pdf_report(file_path: str, image_path: str, pil_image: PILImage.Image,
                     prediction: str, confidence: float, features: Dict,
                     grad_cam_path: Optional[str], conversation_history: List[Dict]) -> tuple:
    """Create PDF report"""
    return PneumoniaReportGenerator().generate_report(
        file_path, image_path, pil_image, prediction, confidence, features, grad_cam_path, conversation_history)
