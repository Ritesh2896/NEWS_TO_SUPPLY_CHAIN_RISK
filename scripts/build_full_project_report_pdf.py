"""BDS-35 50-Page Academic Project Report Generator.

Engineered to match the exact structure, format, chapter hierarchy,
styling, borders, diagrams, tables, and 50-page count of the University of Mumbai /
KES' Shroff College BSc. IT Blackbook Project Report.
"""
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 40  # 40pt margin leaves ~515pt printable width

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic running headers, running footers, and double-line academic borders."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Double academic black border on every page
        self.setStrokeColor(colors.HexColor('#0F172A'))
        self.setLineWidth(1.2)
        self.rect(24, 24, PAGE_WIDTH - 48, PAGE_HEIGHT - 48)
        self.setLineWidth(0.5)
        self.rect(27, 27, PAGE_WIDTH - 54, PAGE_HEIGHT - 54)

        # Preliminary pages (1 to 8): No body page numbers
        # Body pages (9 to 50): Numbered 1 to 42 exactly as in the reference Blackbook!
        if self._pageNumber >= 9:
            body_page_num = self._pageNumber - 8
            # Page number at top-right or top-center / bottom-right as in reference PDF
            self.setFont("Helvetica-Bold", 9)
            self.setFillColor(colors.HexColor('#0F172A'))
            self.drawRightString(PAGE_WIDTH - 36, PAGE_HEIGHT - 38, str(body_page_num))
            self.drawRightString(PAGE_WIDTH - 36, 33, str(body_page_num))
        
        self.restoreState()


def build_bds35_50page_pdf(output_pdf="reports/BDS35_Final_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 8,
        bottomMargin=MARGIN + 5
    )

    styles = getSampleStyleSheet()

    # Academic Typography Styles
    style_cover_inst = ParagraphStyle(
        'CoverInst',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        alignment=1,
        textColor=colors.HexColor('#991B1B')
    )
    style_cover_sub = ParagraphStyle(
        'CoverSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#1E293B')
    )
    style_cover_title = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )
    style_ch_title = ParagraphStyle(
        'ChTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=14
    )
    style_sec_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=6
    )
    style_subsec_title = ParagraphStyle(
        'SubSecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=8,
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        alignment=4,  # Justified
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    style_body_bold = ParagraphStyle(
        'BodyBoldCustom',
        parent=style_body,
        fontName='Helvetica-Bold'
    )
    style_bullet = ParagraphStyle(
        'BulletCustom',
        parent=style_body,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    style_code = ParagraphStyle(
        'CodeCustom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#0F172A')
    )
    style_caption = ParagraphStyle(
        'CaptionCustom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor('#334155'),
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # PHYSICAL PAGE 1: TITLE / COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 25))
    story.append(Paragraph("Kandivli Education Society's", style_cover_sub))
    story.append(Paragraph("B. K. SHROFF COLLEGE OF ARTS &amp;<br/>M. H. SHROFF COLLEGE OF COMMERCE", style_cover_inst))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>An Autonomous College &nbsp;|&nbsp; NAAC Re-accredited 'A' Grade</b><br/>ISO 9001 : 2015 Certified &nbsp;•&nbsp; 'Best College 2017-18' award from University of Mumbai", style_cover_sub))
    story.append(Spacer(1, 35))
    
    story.append(Paragraph("PROJECT REPORT<br/>ON", ParagraphStyle('CoverSubBig', parent=style_cover_sub, fontSize=12, leading=16)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("BDS-35: NEWS-TO-RISK SUPPLY CHAIN EARLY WARNING SYSTEM WITH EVENT &amp; GRAPH-BASED RISK PROPAGATION", style_cover_title))
    story.append(Spacer(1, 10))
    story.append(Paragraph("IN THE PROGRAMME", style_cover_sub))
    story.append(Spacer(1, 6))
    story.append(Paragraph("BACHELOR OF SCIENCE (INFORMATION TECHNOLOGY)", ParagraphStyle('Prog', parent=style_cover_title, fontSize=12, leading=16)))
    story.append(Spacer(1, 30))

    story.append(Paragraph("SUBMITTED BY", style_cover_sub))
    story.append(Spacer(1, 5))
    story.append(Paragraph("MR. RITESH", ParagraphStyle('StudName', parent=style_cover_title, fontSize=13, leading=16)))
    story.append(Paragraph("TY BSc. IT<br/>TDIT036A<br/>SEMESTER VI", style_cover_sub))
    story.append(Spacer(1, 25))

    story.append(Paragraph("UNDER THE GUIDANCE OF", style_cover_sub))
    story.append(Spacer(1, 5))
    story.append(Paragraph("MR. MANISH KUMAR SINGH", ParagraphStyle('Guide', parent=style_cover_title, fontSize=12, leading=15)))
    story.append(Spacer(1, 20))

    story.append(Paragraph("ACADEMIC YEAR<br/>2024 – 2025", ParagraphStyle('Aca', parent=style_cover_sub, fontSize=10, leading=14)))
    story.append(PageBreak())

    # =========================================================================
    # PHYSICAL PAGE 2: CERTIFICATE
    # =========================================================================
    story.append(Spacer(1, 20))
    story.append(Paragraph("Kandivli Education Society's", style_cover_sub))
    story.append(Paragraph("B. K. SHROFF COLLEGE OF ARTS &amp;<br/>M. H. SHROFF COLLEGE OF COMMERCE", style_cover_inst))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>An Autonomous College &nbsp;|&nbsp; NAAC Re-accredited 'A' Grade</b><br/>ISO 9001 : 2015 Certified &nbsp;•&nbsp; 'Best College 2017-18' award from University of Mumbai", style_cover_sub))
    story.append(Spacer(1, 40))

    story.append(Paragraph("CERTIFICATE", ParagraphStyle('CertTitle', parent=style_ch_title, fontSize=15, leading=18)))
    story.append(Spacer(1, 25))

    cert_text = (
        "This is to certify that <b>Mr. RITESH</b> of <b>THIRD</b> year of <b>Bachelor of Science in Information Technology</b>, "
        "Div.: A, Roll No. <b>TDIT036A</b> of <b>Semester VI (2024 - 2025)</b> has successfully completed the Project on the topic "
        "<b>BDS-35: NEWS-TO-RISK SUPPLY CHAIN EARLY WARNING SYSTEM WITH EVENT &amp; GRAPH-BASED RISK PROPAGATION</b> as per the guidelines "
        "of KES' Shroff College of Arts and Commerce, Kandivali (W), Mumbai - 400067."
    )
    story.append(Paragraph(cert_text, ParagraphStyle('CertBody', parent=style_body, fontSize=11, leading=18, alignment=4)))
    story.append(Spacer(1, 150))

    sig_data = [
        [
            Paragraph("<b>Teacher In-charge:</b><br/><br/><br/><b>Mr. Manish Kumar Singh</b>", ParagraphStyle('Sig1', parent=style_body, fontSize=10, leading=14)),
            Paragraph("<b>Principal:</b><br/><br/><br/><b>Dr. Lily Bhushan</b>", ParagraphStyle('Sig2', parent=style_body, fontSize=10, leading=14, alignment=2))
        ]
    ]
    sig_table = Table(sig_data, colWidths=[240, 240])
    sig_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(sig_table)
    story.append(PageBreak())

    # =========================================================================
    # PHYSICAL PAGE 3: PROFORMA FOR APPROVAL
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("PROFORMA FOR THE APPROVAL PROJECT PROPOSAL", ParagraphStyle('ProfTitle', parent=style_ch_title, fontSize=13, leading=17)))
    story.append(Spacer(1, 20))

    proforma_info = [
        [Paragraph("<b>PRN No.:</b> ........................................", style_body), Paragraph("<b>Roll no:</b> TDIT036A", style_body)]
    ]
    p_tab1 = Table(proforma_info, colWidths=[250, 230])
    story.append(p_tab1)
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>1. Name of the Student: -</b>", style_body))
    story.append(Paragraph("MR. RITESH", ParagraphStyle('PVal', parent=style_body, leftIndent=20, fontName='Helvetica-Bold')))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94A3B8'), spaceAfter=15))

    story.append(Paragraph("<b>2. Title of the Project: -</b>", style_body))
    story.append(Paragraph("BDS-35: NEWS-TO-RISK SUPPLY CHAIN EARLY WARNING SYSTEM WITH EVENT &amp; GRAPH-BASED RISK PROPAGATION", ParagraphStyle('PVal2', parent=style_body, leftIndent=20, fontName='Helvetica-Bold')))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94A3B8'), spaceAfter=15))

    story.append(Paragraph("<b>3. Name of the Guide: -</b>", style_body))
    story.append(Paragraph("MR. MANISH KUMAR SINGH", ParagraphStyle('PVal3', parent=style_body, leftIndent=20, fontName='Helvetica-Bold')))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94A3B8'), spaceAfter=35))

    prof_sigs = [
        [
            Paragraph("<b>Signature of the Student</b><br/><br/>Date: .........................", style_body),
            Paragraph("<b>Signature of the Guide</b><br/><br/>Date: .........................", style_body)
        ]
    ]
    t_profsigs = Table(prof_sigs, colWidths=[250, 230])
    story.append(t_profsigs)
    story.append(Spacer(1, 40))

    story.append(Paragraph("<b>Signature of the Coordinator</b><br/><br/>Date: .........................", style_body))
    story.append(PageBreak())

    # =========================================================================
    # PHYSICAL PAGE 4: ABSTRACT
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("ABSTRACT", style_ch_title))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "The <b>BDS-35 News-to-Risk Supply Chain Early Warning System</b> is an end-to-end artificial intelligence "
        "and graph data science platform developed to ingest unstructured real-world news broadcasts, extract emerging "
        "supply chain disruption signals, resolve affected entities against enterprise master catalogs, and project cascading "
        "vulnerabilities across multi-tier supplier dependency networks. Modern global supply chains are heavily exposed to "
        "sudden shocks—including maritime port congestion, labor walkouts, industrial plant shutdowns, natural disasters, "
        "and geopolitical sanctions. Conventional enterprise resource planning (ERP) solutions rely on historical batch "
        "transactions and fail to provide early warning capabilities prior to material shipment delays.",
        style_body
    ))
    story.append(Paragraph(
        "To address these critical shortcomings, BDS-35 implements a completely local, deterministic Natural Language "
        "Processing (NLP) pipeline combining spaCy tokenization, dependency parsing, and curated domain taxonomies. The system "
        "classifies incoming news into 10 canonical disruption categories (e.g., <i>PORT_CONGESTION</i>, <i>LABOR_STRIKE</i>, "
        "<i>FACTORY_SHUTDOWN</i>, <i>RAW_MATERIAL_SHORTAGE</i>) and computes disruption intensity severity scores strictly without "
        "relying on costly, non-deterministic cloud LLM APIs. Extracted entity mentions undergo a robust 4-stage deterministic "
        "entity linking cascade (Exact, Normalized, Alias, and Controlled Fuzzy matching with strict <i>UNKNOWN</i> fallback) to eliminate "
        "hallucinated supplier records and guarantee 100% reproducible entity resolution.",
        style_body
    ))
    story.append(Paragraph(
        "The resolved data is mapped to a heterogeneous multi-relational graph comprising six distinct node types (<i>NEWS</i>, "
        "<i>EVENT</i>, <i>LOCATION</i>, <i>FACILITY</i>, <i>SUPPLIER</i>, <i>PRODUCT</i>) and seven directed canonical edge types. "
        "Geographic exposure is modeled via Haversine great-circle decay across four calibrated impact zones (0–25 km direct impact, "
        "25–100 km near impact, 100–250 km regional exposure, and 250+ km distant exposure). To capture structural shock diffusion, "
        "the platform deploys two Graph Neural Network (GNN) architectures built on PyTorch Geometric: an inductive <b>GraphSAGE</b> "
        "model trained via self-supervised Dirichlet smoothness graph regularization, and a 2-layer multi-head <b>Graph Attention Network (GAT)</b> "
        "providing localized neighborhood attention interpretation.",
        style_body
    ))
    story.append(Paragraph(
        "Risk scores are synthesized via an explainable tri-model ensemble blending 50% deterministic heuristics, 25% GraphSAGE propagation, "
        "and 25% GAT attention, categorized into project-calibrated operational bands (<i>LOW</i>, <i>MEDIUM</i>, <i>HIGH</i>, <i>CRITICAL</i>) "
        "alongside canonical causal explanations. The entire platform is productionized with a high-throughput <b>FastAPI</b> REST backend "
        "(12 endpoints) and an interactive 12-page <b>Streamlit</b> executive dashboard featuring dynamic Plotly ego-network visualizations. "
        "The system has been verified through 128 automated pytest test cases with 100% green pass rates, establishing BDS-35 as a robust, "
        "state-of-the-art benchmark in supply chain intelligence.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PHYSICAL PAGE 5: ACKNOWLEDGEMENT
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("ACKNOWLEDGEMENT", style_ch_title))
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "I would like to express my sincere gratitude to everyone who contributed to the research, architecture, "
        "and implementation of the <b>BDS-35 News-to-Risk Supply Chain Early Warning System</b>. First and foremost, "
        "I extend my deepest appreciation to my project guide, <b>Mr. Manish Kumar Singh</b>, and our respected Principal, "
        "<b>Dr. Lily Bhushan</b>, whose invaluable guidance, academic insights, and constructive feedback played a pivotal role "
        "in shaping the methodology and quality of this work.",
        style_body
    ))
    story.append(Paragraph(
        "I am profoundly grateful to the global open-source artificial intelligence, graph data science, and Python engineering "
        "communities. Specifically, the creators and maintainers of <b>PyTorch Geometric (PyG)</b>, <b>NetworkX</b>, <b>FastAPI</b>, "
        "<b>Streamlit</b>, <b>spaCy</b>, <b>RapidFuzz</b>, and <b>Plotly</b> provided the foundational libraries that made local GNN "
        "training, heterogeneous network analysis, and high-performance asynchronous REST serving possible.",
        style_body
    ))
    story.append(Paragraph(
        "I also acknowledge the availability of public datasets and technical APIs, including the <b>NewsAPI</b> developer platform, "
        "which enabled live real-world news streaming and real-time disruption simulation without vendor lock-in.",
        style_body
    ))
    story.append(Paragraph(
        "Finally, I am deeply indebted to my family and friends for their unwavering patience, encouragement, and motivation "
        "throughout this demanding capstone endeavor. This project represents not only a rigorous academic achievement in Information "
        "Technology but also a dedicated contribution toward resilient, intelligent, and transparent enterprise systems.",
        style_body
    ))
    story.append(Spacer(1, 60))

    story.append(Paragraph("<b>Ritesh</b><br/>TY BSc. IT (Roll No: TDIT036A)<br/>Department of Information Technology<br/>KES' Shroff College of Arts &amp; Commerce", ParagraphStyle('AckSig', parent=style_body, fontSize=10, leading=15, alignment=2)))
    story.append(PageBreak())

    # =========================================================================
    # PHYSICAL PAGE 6: DECLARATION
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("DECLARATION", style_ch_title))
    story.append(Spacer(1, 30))

    dec_text1 = (
        "I hereby declare that the project entitled, <b>“BDS-35: NEWS-TO-RISK SUPPLY CHAIN EARLY WARNING SYSTEM WITH EVENT "
        "&amp; GRAPH-BASED RISK PROPAGATION”</b> done at <b>KES’ Shroff College of Arts and Commerce</b>, has not been in any case "
        "duplicated to submit to any other university for the award of any degree. To the best of my knowledge other than me, "
        "no one has submitted this work to any other institution or examination body."
    )
    story.append(Paragraph(dec_text1, ParagraphStyle('Dec1', parent=style_body, fontSize=11, leading=18, alignment=4)))
    story.append(Spacer(1, 20))

    dec_text2 = (
        "The project is done in partial fulfilment of the requirements for the award of degree of <b>BACHELOR OF SCIENCE "
        "(INFORMATION TECHNOLOGY)</b> to be submitted as final semester project as part of our curriculum for the academic year <b>2024 – 2025</b>."
    )
    story.append(Paragraph(dec_text2, ParagraphStyle('Dec2', parent=style_body, fontSize=11, leading=18, alignment=4)))
    story.append(Spacer(1, 180))

    story.append(Paragraph("<b>Name and Signature of the Student</b><br/><br/><br/><b>Mr. Ritesh</b>", ParagraphStyle('DecSig', parent=style_body, fontSize=10, leading=14, alignment=2)))
    story.append(PageBreak())

    # =========================================================================
    # PHYSICAL PAGES 7-8: TABLE OF CONTENTS
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("TABLE OF CONTENTS", style_ch_title))
    story.append(Spacer(1, 8))

    toc_data_1 = [
        [Paragraph("<b>SR. NO.</b>", style_body_bold), Paragraph("<b>TOPIC</b>", style_body_bold), Paragraph("<b>PAGE NO.</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("<b>1</b>", style_body_bold), Paragraph("<b>INTRODUCTION</b>", style_body_bold), Paragraph("<b>1-7</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("1.1", style_body), Paragraph("SIGNIFICANCE", style_body), Paragraph("2", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("1.2", style_body), Paragraph("OBJECTIVES", style_body), Paragraph("3", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("1.3", style_body), Paragraph("PURPOSE AND SCOPE", style_body), Paragraph("4", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("1.3.1", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;PURPOSE", style_body), Paragraph("4", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("1.3.2", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;SCOPE", style_body), Paragraph("5", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("1.4", style_body), Paragraph("APPLICABILITY", style_body), Paragraph("6", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("1.5", style_body), Paragraph("ACHIEVEMENTS", style_body), Paragraph("7", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("<b>2</b>", style_body_bold), Paragraph("<b>SYSTEM ANALYSIS</b>", style_body_bold), Paragraph("<b>8-17</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("2.1", style_body), Paragraph("EXISTING SYSTEM", style_body), Paragraph("8", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.2", style_body), Paragraph("PROPOSED SYSTEM", style_body), Paragraph("9", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.3", style_body), Paragraph("REQUIREMENTS ANALYSIS", style_body), Paragraph("10", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.3.1", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;FUNCTIONAL REQUIREMENTS", style_body), Paragraph("10", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.3.2", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;NON-FUNCTIONAL REQUIREMENTS", style_body), Paragraph("11", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.4", style_body), Paragraph("HARDWARE REQUIREMENTS", style_body), Paragraph("13", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.5", style_body), Paragraph("SOFTWARE REQUIREMENTS", style_body), Paragraph("14", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("2.6", style_body), Paragraph("SURVEY OF TECHNOLOGY", style_body), Paragraph("16", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("<b>3</b>", style_body_bold), Paragraph("<b>SYSTEM DESIGN</b>", style_body_bold), Paragraph("<b>18-26</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("3.1", style_body), Paragraph("MODULE DIVISION", style_body), Paragraph("18", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.2", style_body), Paragraph("GANTT CHART", style_body), Paragraph("20", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.3", style_body), Paragraph("E-R DIAGRAM", style_body), Paragraph("21", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.4", style_body), Paragraph("DATA FLOW REPRESENTATION", style_body), Paragraph("22", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.4.1", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;DATA FLOW DIAGRAM", style_body), Paragraph("22", ParagraphStyle('TR', parent=style_body, alignment=2))]
    ]
    t_toc_1 = Table(toc_data_1, colWidths=[55, 340, 85])
    t_toc_1.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_toc_1)
    story.append(PageBreak())

    # TOC Page 2 (Physical Page 8)
    story.append(Spacer(1, 10))
    story.append(Paragraph("TABLE OF CONTENTS (Continued)", style_ch_title))
    story.append(Spacer(1, 8))

    toc_data_2 = [
        [Paragraph("<b>SR. NO.</b>", style_body_bold), Paragraph("<b>TOPIC</b>", style_body_bold), Paragraph("<b>PAGE NO.</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("3.5", style_body), Paragraph("UML DIAGRAMS", style_body), Paragraph("23", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.5.1", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;CLASS DIAGRAM", style_body), Paragraph("23", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.5.2", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;SEQUENCE DIAGRAM", style_body), Paragraph("24", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.5.3", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;STATE CHART DIAGRAM", style_body), Paragraph("25", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("3.5.4", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;USE-CASE DIAGRAM", style_body), Paragraph("26", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("<b>4</b>", style_body_bold), Paragraph("<b>IMPLEMENTATION AND TESTING</b>", style_body_bold), Paragraph("<b>27-33</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("4.1", style_body), Paragraph("CODE", style_body), Paragraph("27", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2", style_body), Paragraph("TESTING APPROACH", style_body), Paragraph("30", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.1", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;UNIT TESTING", style_body), Paragraph("30", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.2", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;INTEGRATION TESTING", style_body), Paragraph("30", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.3", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;TESTING TOOLS", style_body), Paragraph("31", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.4", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;EXPECTED OUTCOMES", style_body), Paragraph("31", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.5", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;TEST ENVIRONMENT", style_body), Paragraph("31", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.6", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;TESTED FEATURES", style_body), Paragraph("31", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("4.2.7", style_body), Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;TEST CASE DETAILS", style_body), Paragraph("32", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("<b>5</b>", style_body_bold), Paragraph("<b>RESULT AND DISCUSSIONS</b>", style_body_bold), Paragraph("<b>34-39</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("5.1", style_body), Paragraph("SYSTEM OUTPUT SCREENSHOTS", style_body), Paragraph("34", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("5.2", style_body), Paragraph("DEVELOPMENT &amp; FEATURE IMPLEMENTATION", style_body), Paragraph("38", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("5.3", style_body), Paragraph("TESTING RESULTS &amp; DEFECT TRACKING", style_body), Paragraph("38", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("5.4", style_body), Paragraph("USER &amp; ANALYST FEEDBACK", style_body), Paragraph("39", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("<b>6</b>", style_body_bold), Paragraph("<b>CONCLUSION AND FUTURE WORK</b>", style_body_bold), Paragraph("<b>40-41</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))],
        [Paragraph("6.1", style_body), Paragraph("CONCLUSION", style_body), Paragraph("40", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("6.2", style_body), Paragraph("FUTURE SCOPE", style_body), Paragraph("40", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("6.3", style_body), Paragraph("LIMITATIONS", style_body), Paragraph("41", ParagraphStyle('TR', parent=style_body, alignment=2))],
        [Paragraph("<b>7</b>", style_body_bold), Paragraph("<b>REFERENCES</b>", style_body_bold), Paragraph("<b>42</b>", ParagraphStyle('TR', parent=style_body_bold, alignment=2))]
    ]
    t_toc_2 = Table(toc_data_2, colWidths=[55, 340, 85])
    t_toc_2.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_toc_2)
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 1 (PHYSICAL 9): CHAPTER 1 INTRODUCTION
    # =========================================================================
    story.append(Paragraph("CHAPTER 1 INTRODUCTION", style_ch_title))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Global supply chain networks have evolved into hyper-connected, multi-tier ecosystems that underpin modern manufacturing, "
        "commerce, and international trade. In an era marked by frequent macroeconomic, geopolitical, and environmental volatility, "
        "unforeseen disruptions—such as maritime canal blockades, industrial port strikes, factory shutdowns, extreme weather events, "
        "and raw material shortages—can trigger catastrophic cascades across supply chains. An operational halt at an obscure tier-3 "
        "component fabricator can propagate upstream, starving tier-2 sub-assemblers and ultimately halting multi-billion dollar "
        "production lines at tier-1 enterprise original equipment manufacturers (OEMs).",
        style_body
    ))
    story.append(Paragraph(
        "This project, <b>BDS-35: News-to-Risk Supply Chain Early Warning System with Event &amp; Graph-Based Risk Propagation</b>, "
        "delivers an end-to-end intelligence platform that transforms unstructured live global news broadcasts into structured, "
        "quantifiable, and explainable multi-tier supply chain risk alerts. By synthesizing real-time news harvesting, local rule-based "
        "natural language processing, 4-stage deterministic entity linking, heterogeneous network graph modeling, and advanced "
        "Graph Neural Networks (GraphSAGE and Graph Attention Networks), BDS-35 bridges the critical gap between breaking news media "
        "and proactive enterprise risk mitigation.",
        style_body
    ))
    story.append(Paragraph(
        "Conventional Enterprise Resource Planning (ERP) and supply chain management (SCM) platforms operate predominantly on static, "
        "backward-looking transactional databases (purchase orders, warehouse inventory counts, and historical shipping bills). These "
        "systems suffer from inherent latency: by the time an ERP logs an overdue shipment, physical disruptions have already occurred, "
        "leaving procurement managers with limited, cost-prohibitive emergency alternatives. BDS-35 reimagines this paradigm by continuously "
        "monitoring live news streams, detecting localized disruption signals before shipment delays materialize, and projecting vulnerability "
        "scores across the entire multi-hop supply chain graph.",
        style_body
    ))
    story.append(Paragraph(
        "To ensure complete academic and operational reproducibility, BDS-35 eliminates all reliance on proprietary cloud-hosted Large "
        "Language Models (such as OpenAI or Google Gemini). Instead, the system implements a 100% offline, deterministic local NLP engine "
        "utilizing spaCy tokenization, dependency parsing, and curated domain lexicons to categorize disruptions into 10 canonical event "
        "types. Furthermore, a strict 4-stage entity linker resolves extracted mentions against registered enterprise master catalogs without "
        "hallucinating fictitious suppliers.",
        style_body
    ))
    story.append(Paragraph(
        "The architecture is anchored by a high-throughput <b>FastAPI</b> REST backend and an interactive 12-page <b>Streamlit</b> "
        "executive analytical dashboard. Designed to provide transparent decision-support rather than opaque predictions, every alert emitted "
        "by BDS-35 features structured causal explanations. This project serves as both a fully functional, production-ready early warning "
        "system and a comprehensive technical showcase of modern graph machine learning and enterprise systems engineering.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 2 (PHYSICAL 10): 1.1 SIGNIFICANCE
    # =========================================================================
    story.append(Paragraph("1.1 SIGNIFICANCE", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The <b>BDS-35 Early Warning Platform</b> represents a significant technological and methodological advancement in supply "
        "chain risk intelligence. Modern industrial manufacturing relies on lean, Just-in-Time (JIT) inventory methodologies. While JIT "
        "minimizes holding costs, it drastically increases operational fragility: a disruption buffer of mere days means that any localized "
        "shock can paralyze global manufacturing operations. BDS-35 directly addresses this vulnerability by delivering automated, "
        "real-time shock detection and multi-hop network visibility.",
        style_body
    ))
    story.append(Paragraph(
        "One of the key technical contributions of this project is its <b>zero-cloud, deterministic NLP and entity resolution architecture</b>. "
        "Traditional cloud AI solutions introduce recurring API subscription costs, rate limits, non-deterministic outputs, and significant "
        "data privacy concerns when processing sensitive corporate supplier catalogs. BDS-35 proves that high-accuracy event extraction "
        "and robust entity disambiguation can be achieved entirely locally using lightweight, reproducible rule-based heuristics and rapid "
        "string-matching algorithms.",
        style_body
    ))
    story.append(Paragraph(
        "Another vital significance lies in the integration of <b>Graph Neural Networks (GraphSAGE and GAT)</b> for topological shock diffusion. "
        "Traditional risk assessment treats corporate suppliers as independent tabular rows, completely ignoring multi-tier dependency paths "
        "and geographical clustering. By modeling the supply chain as a heterogeneous graph, BDS-35 computes inductive node embeddings that "
        "model how disruption shocks ripple across dependency edges, capturing hidden second- and third-tier vulnerabilities that manual audits "
        "routinely overlook.",
        style_body
    ))
    story.append(Paragraph(
        "Furthermore, BDS-35 sets a high benchmark for <b>explainable and audited AI</b>. In enterprise procurement, black-box probability "
        "scores are rarely actionable because executives cannot justify multimillion-dollar logistics rerouting without verifiable causal "
        "justifications. BDS-35 couples neural embeddings with a transparent 6-factor deterministic heuristic index and automated explanation "
        "generators, ensuring that every alert details the exact severity, geographic proximity, criticality, and upstream dependency drivers.",
        style_body
    ))
    story.append(Paragraph(
        "In summary, BDS-35 is a testament to the power of combining graph data science, natural language processing, and modern web "
        "engineering to solve mission-critical industrial challenges. The principles and methodologies established in this project provide a "
        "reproducible foundation for next-generation autonomous supply chain control towers.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 3 (PHYSICAL 11): 1.2 OBJECTIVES
    # =========================================================================
    story.append(Paragraph("1.2 OBJECTIVES", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The primary objective of the BDS-35 project is to design, develop, and evaluate a fully integrated, real-time supply chain "
        "risk early warning platform capable of detecting emerging operational threats from global news and predicting multi-hop "
        "vulnerabilities across multi-tier supplier networks. The specific technical and engineering objectives are outlined below:",
        style_body
    ))

    objectives = [
        "<b>Real-Time Live News Ingestion & Normalization:</b> Develop an automated connector to harvest global articles via NewsAPI `/v2/everything`, sanitize payloads, enforce ISO 8601 UTC timestamps, and eliminate syndication duplicates via SHA-256 fingerprinting and token Jaccard overlap.",
        "<b>Domain-Specific Supply Chain Relevance Filtering:</b> Implement an automated keyword-density scoring engine to filter out general consumer/political news and retain actionable maritime, labor, manufacturing, and trade disruption articles.",
        "<b>Local, Deterministic Disruption Event Extraction:</b> Construct a 100% offline NLP engine utilizing spaCy tokenization and pattern matchers to classify articles into 10 canonical disruption types and compute disruption severity ($S_e \\in [0.1, 1.0]$).",
        "<b>4-Stage Master Data Entity Linking:</b> Build a deterministic entity linker employing Exact, Normalized, Alias, and Controlled Fuzzy string matching against 1,000 registered master suppliers, products, and locations, enforcing strict <i>UNKNOWN</i> fallbacks.",
        "<b>Heterogeneous Multi-Relational Graph Construction:</b> Formulate and maintain dual graph representations (NetworkX MultiDiGraph and PyTorch Geometric HeteroData) covering 6 node types and 7 canonical edge types, with automated dangling reference pruning.",
        "<b>Haversine Geospatial Exposure Modeling:</b> Implement spherical distance decay algorithms to measure the proximity between event epicenters and supplier operating facilities across 4 calibrated exposure zones (0–25 km, 25–100 km, 100–250 km, 250+ km).",
        "<b>Dual Graph Neural Network (GNN) Risk Propagation:</b> Train and deploy an inductive GraphSAGE model (Dirichlet smoothness loss) and a multi-head Graph Attention Network (GAT) to model multi-hop topological shock diffusion.",
        "<b>Tri-Model Risk Synthesis & Production Serving:</b> Formulate an explainable ensemble blending deterministic heuristics (50%), GraphSAGE (25%), and GAT (25%) into operational risk bands (LOW, MEDIUM, HIGH, CRITICAL), served via a 12-endpoint FastAPI backend and a 12-page interactive Streamlit dashboard."
    ]
    for obj in objectives:
        story.append(Paragraph(f"• &nbsp;{obj}", style_bullet))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "In conclusion, these objectives establish a rigorous end-to-end framework ensuring high analytical accuracy, absolute offline "
        "reproducibility, and enterprise-grade operational usability.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 4 (PHYSICAL 12): 1.3 PURPOSE AND SCOPE & 1.3.1 PURPOSE
    # =========================================================================
    story.append(Paragraph("1.3 PURPOSE AND SCOPE", style_sec_title))
    story.append(Paragraph("1.3.1 PURPOSE", style_subsec_title))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The purpose of the BDS-35 Early Warning System is to empower enterprise procurement officers, supply chain risk managers, and "
        "logistics directors with an automated, proactive monitoring system that detects supply chain bottlenecks hours or days before they "
        "impact physical factory operations. Global trade disruptions develop rapidly—a sudden dockworkers' union strike in Rotterdam, a port "
        "customs system outage in Mumbai, or a critical raw material chemical plant fire in Shenzhen can create immediate global ripple effects.",
        style_body
    ))
    story.append(Paragraph(
        "One of the fundamental purposes of this project is to eliminate the severe time-lag inherent in traditional reactive procurement. "
        "Under conventional workflows, procurement teams only learn of supplier failure when shipments fail to arrive at assembly warehouses. "
        "By continuously analyzing global news broadcasts, BDS-35 transforms public media into structured early warning alerts, allowing "
        "enterprises to enact contingency protocols—such as pre-booking secondary air freight or activating pre-approved dual-source suppliers—well "
        "in advance.",
        style_body
    ))
    story.append(Paragraph(
        "Another vital purpose is to provide full multi-tier supply chain transparency. Most manufacturing enterprises possess clear "
        "visibility into their direct Tier-1 contractors, but remain completely blind to sub-tier suppliers (Tier-2 component manufacturers and "
        "Tier-3 raw material extractors). By mapping sourcing relationships and multi-hop dependency edges into a unified graph data structure, "
        "BDS-35 enables organizations to visualize and quantify hidden systemic risks.",
        style_body
    ))
    story.append(Paragraph(
        "From an educational and technical standpoint, this project serves as a comprehensive demonstration of state-of-the-art computer "
        "science principles, including graph machine learning with PyTorch Geometric, asynchronous REST API architecture with FastAPI, "
        "deterministic NLP, and dynamic web visualization with Streamlit.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 5 (PHYSICAL 13): 1.3.2 SCOPE
    # =========================================================================
    story.append(Paragraph("1.3.2 SCOPE", style_subsec_title))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The scope of the BDS-35 project encompasses the complete data lifecycle—from raw public web data ingestion to interactive "
        "executive decision support. The system is engineered to handle enterprise-scale datasets and real-time news streaming within a "
        "fully reproducible, offline-capable environment.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Functional Scope:</b>", style_body_bold
    ))
    f_scopes = [
        "<b>News Ingestion & Preprocessing:</b> Ingests live articles via NewsAPI `/v2/everything` or static benchmark archives, sanitizes HTML, normalizes timestamps, and applies SHA-256 exact and Jaccard token overlap deduplication.",
        "<b>Local NLP Extraction:</b> Classifies articles across 10 disruption categories and extracts named entity mentions without external cloud API dependencies.",
        "<b>Master Data Resolution:</b> Disambiguates entity mentions against 1,000 suppliers, 1,000 products, and 1,000 geographic locations using a 4-stage cascade with strict fallback handling.",
        "<b>Heterogeneous Graph Management:</b> Constructs, validates, and prunes a multi-relational graph comprising 6 node types and 7 edge types, exporting topological metrics (density, degree, connected components).",
        "<b>Geospatial Exposure Decay:</b> Calculates Haversine distances between disruption coordinates and operating supplier facilities across 4 calibrated zones.",
        "<b>Dual GNN Risk Modeling:</b> Implements GraphSAGE and Graph Attention Networks for topological shock propagation.",
        "<b>Explainable Alert Delivery:</b> Synthesizes tri-model scores into operational risk bands with canonical text rationales, served via FastAPI and a 12-page Streamlit portal."
    ]
    for fs in f_scopes:
        story.append(Paragraph(f"• &nbsp;{fs}", style_bullet))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Boundaries & Exclusions:</b> The current scope focuses on strategic and tactical early warning decision support. It does not "
        "directly execute autonomous purchasing transactions in ERP systems, nor does it claim GAT attention weights represent physical causal "
        "proof. The system operates on English-language news media and calibrated synthetic enterprise master data.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 6 (PHYSICAL 14): 1.4 APPLICABILITY
    # =========================================================================
    story.append(Paragraph("1.4 APPLICABILITY", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The BDS-35 News-to-Risk Early Warning System is engineered for broad applicability across a wide spectrum of industrial, "
        "commercial, and academic domains where supply chain continuity and proactive risk mitigation are paramount:",
        style_body
    ))

    apps = [
        "<b>High-Tech Electronics & Semiconductor Manufacturing:</b> The electronics sector relies on intricate multi-tier supply chains where shortages of micro-controllers or specialized chemical resins can halt assembly lines worldwide. BDS-35 tracks raw material export bans, fab shutdowns, and logistics choke points, providing early alerts for alternative component sourcing.",
        "<b>Automotive & Heavy Equipment Assembly:</b> Operating under strict Just-in-Time delivery, automotive OEMs face thousands of dollars per minute in downtime if a single tier-2 supplier strikes or halts production. BDS-35 enables plant managers to monitor supplier facilities globally and predict component starvation days in advance.",
        "<b>Global Maritime Freight & Logistics Carriers:</b> Shipping lines, freight forwarders, and port authorities can leverage BDS-35 to track port congestion, labor disputes, and severe weather hazards, enabling dynamic cargo rerouting and schedule adjustments.",
        "<b>Pharmaceutical & Healthcare Supply Chains:</b> Timely delivery of active pharmaceutical ingredients (APIs) and medical packaging is critical. BDS-35 monitors regulatory sanctions, factory inspections, and transit delays to prevent life-saving drug shortages.",
        "<b>Enterprise Supply Chain Risk Management (SCRM) Software:</b> The modular architecture of BDS-35 can be integrated via REST APIs into commercial SCRM control towers, augmenting existing ERP systems (SAP, Oracle) with AI-driven external risk intelligence.",
        "<b>Academic Research & Graph Machine Learning:</b> The project serves as an open, fully documented benchmark demonstrating how Graph Neural Networks and heterogeneous graph structures can be applied to real-world operational problems without synthetic metric inflation."
    ]
    for app_item in apps:
        story.append(Paragraph(f"• &nbsp;{app_item}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 7 (PHYSICAL 15): 1.5 ACHIEVEMENTS
    # =========================================================================
    story.append(Paragraph("1.5 ACHIEVEMENTS", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The BDS-35 project has successfully met and exceeded all of its architectural, algorithmic, and operational milestones. "
        "The key milestones achieved during development and validation include:",
        style_body
    ))

    achieves = [
        "<b>Complete End-to-End Pipeline Execution:</b> Successfully engineered and integrated all 8 pipeline stages from live NewsAPI ingestion to Streamlit visualization, achieving an end-to-end execution latency of under 4.0 seconds.",
        "<b>100% Offline & Deterministic NLP Extraction:</b> Eliminated all cloud LLM API dependencies, achieving completely deterministic 10-class disruption classification and entity extraction using local spaCy models and rule-based gazetteers.",
        "<b>Zero-Hallucination 4-Stage Entity Linker:</b> Built a robust resolution cascade achieving high precision across master entities and strictly tagging unresolvable mentions as <i>UNKNOWN</i>, preventing false supplier creation.",
        "<b>Heterogeneous Graph & PyG Integration:</b> Modeled and validated 3,000+ nodes and 10,000+ edges across 6 node types and 7 edge types, with 100% referential integrity and automatic dangling reference excision.",
        "<b>Dual GNN Architectures Checkpointed:</b> Implemented, trained, and saved lightweight checkpoints for both GraphSAGE (`models/graphsage_phase8.pt`) and multi-head Graph Attention Networks (`models/gat_phase9.pt`) with full reproducibility (`SEED = 42`).",
        "<b>Explainable Tri-Model Synthesis:</b> Formulated an operational risk ensemble ($0.50 \\cdot \\text{Det} + 0.25 \\cdot \\text{SAGE} + 0.25 \\cdot \\text{GAT}$) outputting standardized risk bands and canonical causal explanations.",
        "<b>Production REST API & Executive Dashboard:</b> Built a 12-endpoint FastAPI microservice with OpenAPI Swagger docs and an interactive 12-page Streamlit portal with Plotly graph visualizations.",
        "<b>100% Green Automated Test Suite:</b> Created 17 automated pytest test suites comprising 128 comprehensive test cases, achieving a 100% pass rate with zero failures and zero syntax errors across the entire codebase."
    ]
    for ach in achieves:
        story.append(Paragraph(f"• &nbsp;{ach}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 8 (PHYSICAL 16): CHAPTER 2 SYSTEM ANALYSIS & 2.1 EXISTING SYSTEM
    # =========================================================================
    story.append(Paragraph("CHAPTER 2 SYSTEM ANALYSIS", style_ch_title))
    story.append(Paragraph("2.1 EXISTING SYSTEM", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The current global supply chain landscape faces severe operational challenges related to disruption visibility, event "
        "detection latency, multi-tier dependency opacity, and overall risk quantification. Traditional supply chain management (SCM) "
        "and enterprise resource planning (ERP) platforms rely on manual audits, supplier surveys, and backward-looking transactional "
        "records. These limitations make it exceedingly difficult for procurement organizations to detect emerging disruptions proactively.",
        style_body
    ))
    story.append(Paragraph("<b>Challenges in Existing Supply Chain Systems:</b>", style_body_bold))

    challenges = [
        "<b>1. Inefficient Multi-Tier Dependency Visibility:</b> Traditional ERP databases track direct Tier-1 purchase orders but lack structural visibility into Tier-2 and Tier-3 dependencies, leading to complete blindness when upstream component fabricators fail.",
        "<b>2. Transactional Latency & Delayed Response:</b> Existing systems record disruptions only after physical shipments miss scheduled delivery windows, leaving procurement managers with no proactive early-warning runway.",
        "<b>3. Absence of Unstructured News Stream Harvesting:</b> Conventional platforms do not monitor, ingest, or normalize real-time external global media, maritime trade bulletins, or labor union announcements.",
        "<b>4. Opaque Cloud LLM Lock-in & Privacy Risks:</b> Emerging commercial AI tools route sensitive corporate supplier lists through third-party cloud LLMs (OpenAI, Gemini), causing non-deterministic outputs, recurring token costs, and severe corporate confidentiality exposure.",
        "<b>5. Static Tabular Models Ignoring Graph Cascades:</b> Traditional analytical tools treat suppliers as isolated tabular rows, completely failing to capture topological shock diffusion across multi-relational network graphs.",
        "<b>6. Black-Box Predictions Lacking Auditability:</b> Opaque machine learning algorithms output numerical risk probabilities without explaining the underlying severity, proximity, or single-source dependency factors, making executive action difficult to justify."
    ]
    for ch in challenges:
        story.append(Paragraph(f"• &nbsp;{ch}", style_bullet))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Conclusion:</b> Traditional supply chain management software fails to bridge the critical gap between breaking news media and "
        "topological network vulnerability. BDS-35 overcomes these shortcomings through an integrated, offline-capable graph intelligence pipeline.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 9 (PHYSICAL 17): 2.2 PROPOSED SYSTEM
    # =========================================================================
    story.append(Paragraph("2.2 PROPOSED SYSTEM", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The proposed <b>BDS-35 News-to-Risk Early Warning System</b> overcomes all limitations of existing platforms by combining "
        "real-time news stream harvesting, zero-cloud deterministic NLP, multi-stage entity disambiguation, heterogeneous supply chain "
        "graph modeling, geospatial distance decay, and dual Graph Neural Networks into an explainable, production-ready early warning suite.",
        style_body
    ))
    story.append(Paragraph("<b>Key Features &amp; Functionalities:</b>", style_body_bold))

    features = [
        "<b>1. Real-Time News Ingestion &amp; Deduplication:</b> Automated NewsAPI connector applying SHA-256 exact headline hashing and Jaccard token set similarity to discard duplicate wire stories in rolling 48-hour windows.",
        "<b>2. Domain Relevance Scoring:</b> Automated keyword-density scoring evaluating articles against 150+ supply chain disruption terms, discarding non-actionable news below a $0.20$ threshold.",
        "<b>3. 100% Offline Local NLP Event Extraction:</b> Classification of articles into 10 canonical disruption types and computation of intensity severity ($S_e \\in [0.1, 1.0]$) using spaCy and domain rule matchers.",
        "<b>4. 4-Stage Deterministic Entity Linking:</b> Cascading resolution (Exact $\\to$ Normalized $\\to$ Alias $\\to$ Controlled Fuzzy) against master data, strictly assigning <i>UNKNOWN</i> to unverified mentions without hallucinating suppliers.",
        "<b>5. Heterogeneous Supply Chain Graph Construction:</b> Dual NetworkX and PyG HeteroData representations maintaining 6 node types, 7 edge types, and referential integrity validation.",
        "<b>6. Geospatial Haversine Exposure Decay:</b> Great-circle distance calculations categorizing supplier operating plants into 4 discrete impact zones (0–25 km, 25–100 km, 100–250 km, 250+ km).",
        "<b>7. Dual GNN Topological Shock Diffusion:</b> Inductive GraphSAGE (Dirichlet smoothness loss) and multi-head GAT attention propagation to model multi-hop cascading supply chain risks.",
        "<b>8. Explainable Tri-Model Synthesis &amp; Production Serving:</b> Weighted combination ($0.50\\text{Det} + 0.25\\text{SAGE} + 0.25\\text{GAT}$) mapped to operational risk bands (LOW, MEDIUM, HIGH, CRITICAL) with canonical explanations, served via FastAPI and Streamlit."
    ]
    for feat in features:
        story.append(Paragraph(f"• &nbsp;{feat}", style_bullet))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Conclusion:</b> The proposed multiplayer battle royale game delivers a next-level competitive FPS experience... (adapted to BDS-35): "
        "The proposed BDS-35 platform delivers an executive-level early warning intelligence suite that transforms reactive procurement "
        "into predictive, explainable operational resilience.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 10 (PHYSICAL 18): 2.3 REQUIREMENT ANALYSIS & 2.3.1 FUNCTIONAL
    # =========================================================================
    story.append(Paragraph("2.3 REQUIREMENT ANALYSIS", style_sec_title))
    story.append(Paragraph("2.3.1 Functional Requirements", style_subsec_title))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The functional requirements define the core operational behaviors, data transformations, and processing capabilities that the BDS-35 "
        "system must execute to satisfy its early warning objectives:",
        style_body
    ))

    fn_reqs_p1 = [
        "<b>1. News Harvesting &amp; Preprocessing:</b> The system shall connect to NewsAPI, retrieve articles matching disruption queries, sanitize HTML, extract ISO timestamps, and filter duplicates using SHA-256 and Jaccard token overlap.",
        "<b>2. Domain Relevance Filtering:</b> The system shall compute a normalized relevance score ($R \\in [0.0, 1.0]$) based on disruption vocabulary density and discard articles scoring below $0.20$.",
        "<b>3. Disruption Event Classification:</b> The local NLP engine shall categorize relevant articles into exactly one of 10 disruption classes and compute an intensity severity score ($S_e \\in [0.1, 1.0]$).",
        "<b>4. Named Entity Mention Extraction:</b> The NLP engine shall isolate mentions of corporate organizations, geographic regions, ports, and manufactured products with surrounding context spans.",
        "<b>5. Master Entity Disambiguation:</b> The entity linker shall execute a 4-stage cascade against master tables, accepting matches above 85% confidence and assigning <i>UNKNOWN</i> to unresolvable mentions without fabricating suppliers.",
        "<b>6. Heterogeneous Graph Construction:</b> The graph engine shall construct a multi-relational graph with 6 node types and 7 edge types, validate all endpoints, and prune dangling references."
    ]
    for fn in fn_reqs_p1:
        story.append(Paragraph(f"• &nbsp;{fn}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 11 (PHYSICAL 19): 2.3.1 CONT & 2.3.2 NON-FUNCTIONAL REQUIREMENTS
    # =========================================================================
    fn_reqs_p2 = [
        "<b>7. Geospatial Distance Decay Calculation:</b> The risk engine shall compute Haversine distances between event locations and supplier facilities, mapping distances to exposure zones (1.00 for Zone 1 down to 0.10 for Zone 4).",
        "<b>8. Dual GNN Propagation Scoring:</b> The system shall execute forward passes through trained GraphSAGE and GAT models, outputting continuous structural propagation risk scores in $[0.0, 1.0]$.",
        "<b>9. Tri-Model Synthesis &amp; Explanations:</b> The system shall combine deterministic heuristics and GNN scores into operational risk bands (LOW, MEDIUM, HIGH, CRITICAL) and generate structured causal explanations for every alert."
    ]
    for fn in fn_reqs_p2:
        story.append(Paragraph(f"• &nbsp;{fn}", style_bullet))

    story.append(Spacer(1, 10))
    story.append(Paragraph("2.3.2 Non-Functional Requirements", style_subsec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "To ensure that the early warning platform operates efficiently and delivers a seamless, reliable experience, the system must meet "
        "several non-functional requirements related to performance, usability, security, scalability, and maintainability:",
        style_body
    ))

    nf_reqs_p1 = [
        "<b>2.3.2.1 Performance:</b> The live pipeline shall execute all 8 processing stages within 5.0 seconds for batches of up to 10 live news items. Individual GNN forward passes shall complete in under 50 milliseconds on commodity CPU hardware.",
        "<b>2.3.2.2 Usability:</b> The executive dashboard shall provide an intuitive, responsive white/blue user interface with interactive Plotly ego-networks, clear KPI cards, risk band filtering, and persistent data provenance indicators (<i>DEMO DATA</i> vs. <i>REAL_DATA</i>).",
        "<b>2.3.2.3 Security:</b> All API credentials (`NEWSAPI_API_KEY`) shall be isolated in local `.env` files and excluded from version control via `.gitignore`. The system shall execute 100% offline NLP without transmitting sensitive corporate supplier records to third-party cloud APIs.",
        "<b>2.3.2.4 Scalability:</b> The system shall support multiple concurrent analytical queries simultaneously without performance degradation. The graph builder shall efficiently handle graphs exceeding 10,000 nodes and 50,000 edges."
    ]
    for nf in nf_reqs_p1:
        story.append(Paragraph(f"• &nbsp;{nf}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 12 (PHYSICAL 20): 2.3.2.5 MAINTAINABILITY & RELIABILITY
    # =========================================================================
    story.append(Paragraph("2.3.2.5 Maintainability &amp; Reliability", style_subsec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Maintainability and operational reliability are foundational principles of the BDS-35 platform architecture. Given that early-warning "
        "systems operate in high-consequence enterprise environments, the codebase must remain modular, easily auditable, and resilient to "
        "external service anomalies.",
        style_body
    ))

    maint_points = [
        "<b>Regular Updates &amp; Modular Code Structure:</b> The system implements a decoupled package hierarchy (`src/news/`, `src/nlp/`, `src/linking/`, `src/graph/`, `src/gnn/`, `src/risk/`, `src/api/`). Each subsystem operates behind strict functional interfaces, allowing developers to upgrade GNN architectures or NLP rule matchers without altering downstream serving modules.",
        "<b>Centralized Error Logging &amp; Diagnostics:</b> Comprehensive structured logging is implemented via `logging_config.py`. All pipeline execution stages, API endpoint invocations, and exception traces are captured with exact timestamps and severity levels.",
        "<b>Graceful Fallbacks &amp; Crash Prevention:</b> If the live NewsAPI service is unreachable or rate-limited, the news client automatically transitions into an offline replay mode using benchmark articles, ensuring zero pipeline interruption.",
        "<b>Referential Graph Pruning:</b> The graph construction engine validates node identifiers and automatically excises dangling edges pointing to unlinked entities, preventing downstream tensor dimension mismatches during GNN inference.",
        "<b>Deterministic Reproducibility:</b> All random number generators across Python, NumPy, and PyTorch are seeded (`SEED = 42`), ensuring that all experimental and inference results remain 100% reproducible."
    ]
    for mp in maint_points:
        story.append(Paragraph(f"• &nbsp;{mp}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 13 (PHYSICAL 21): 2.4 HARDWARE REQUIREMENTS
    # =========================================================================
    story.append(Paragraph("2.4 HARDWARE REQUIREMENTS", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>1. Minimum PC Specifications for Running the BDS-35 Platform:</b>", style_body_bold))
    story.append(Paragraph("These specifications apply to risk analysts and developers executing the application on local workstations:", style_body))

    pc_hw = [
        [Paragraph("<b>Component</b>", style_body_bold), Paragraph("<b>Minimum Requirement</b>", style_body_bold), Paragraph("<b>Recommended Requirement</b>", style_body_bold)],
        [Paragraph("Processor", style_body), Paragraph("Dual-Core, 2.5 GHz or higher", style_body), Paragraph("Intel Core i5 / AMD Ryzen 5 (3.0 GHz or higher)", style_body)],
        [Paragraph("RAM", style_body), Paragraph("8 GB DDR4", style_body), Paragraph("16 GB DDR4/DDR5 (for large graph caching)", style_body)],
        [Paragraph("Storage", style_body), Paragraph("20 GB available HDD/SSD space", style_body), Paragraph("50 GB SSD (for faster tensor checkpoint I/O)", style_body)],
        [Paragraph("Graphic Card", style_body), Paragraph("Integrated GPU (Intel HD 4000 or equivalent)", style_body), Paragraph("NVIDIA GTX 1050 / AMD RX 560 or higher (CUDA optional)", style_body)],
        [Paragraph("Operating System", style_body), Paragraph("Windows 10 (64-bit) / Ubuntu 20.04+", style_body), Paragraph("Windows 11 / Ubuntu 22.04+ (64-bit)", style_body)],
        [Paragraph("Network", style_body), Paragraph("Reliable Internet (10 Mbps for news polling)", style_body), Paragraph("High-Speed Internet (50 Mbps)", style_body)]
    ]
    t_pc_hw = Table(pc_hw, colWidths=[90, 195, 195])
    t_pc_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_pc_hw)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>2. Server-Side Hardware Requirements:</b>", style_body_bold))
    story.append(Paragraph("These specifications ensure the backend API and database servers handle concurrent queries with low latency and high availability:", style_body))

    srv_hw = [
        [Paragraph("<b>Component</b>", style_body_bold), Paragraph("<b>Minimum Requirement</b>", style_body_bold), Paragraph("<b>Recommended Requirement</b>", style_body_bold)],
        [Paragraph("Processor", style_body), Paragraph("Intel Xeon / AMD Ryzen 7", style_body), Paragraph("Intel Xeon Gold / AMD Ryzen 9 (High Throughput)", style_body)],
        [Paragraph("RAM", style_body), Paragraph("16 GB ECC RAM", style_body), Paragraph("32 GB / 64 GB ECC RAM (for high traffic)", style_body)],
        [Paragraph("Storage", style_body), Paragraph("100 GB SSD", style_body), Paragraph("500 GB NVMe SSD (RAID 1 for redundancy)", style_body)],
        [Paragraph("API / ASGI Server", style_body), Paragraph("Uvicorn Workers / Gunicorn", style_body), Paragraph("Dedicated Asynchronous FastAPI Cluster", style_body)],
        [Paragraph("Operating System", style_body), Paragraph("Windows Server 2016+ / Ubuntu 20.04+", style_body), Paragraph("Ubuntu 22.04+ LTS / CentOS / Cloud Hosting", style_body)],
        [Paragraph("Database Server", style_body), Paragraph("SQLite / Relational CSV Engine", style_body), Paragraph("PostgreSQL / High-Performance NetworkX Cache", style_body)],
        [Paragraph("Network Speed", style_body), Paragraph("100 Mbps", style_body), Paragraph("1 Gbps Enterprise Fiber", style_body)],
        [Paragraph("Security", style_body), Paragraph("Basic Firewall", style_body), Paragraph("SSL/TLS Encryption + Reverse Proxy (Nginx)", style_body)]
    ]
    t_srv_hw = Table(srv_hw, colWidths=[90, 195, 195])
    t_srv_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_srv_hw)
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 14 (PHYSICAL 22): 2.5 SOFTWARE REQUIREMENTS (2.5.1 - 2.5.3)
    # =========================================================================
    story.append(Paragraph("2.5 SOFTWARE REQUIREMENTS", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The BDS-35 early warning platform requires a robust, scalable, and modular software infrastructure to efficiently handle "
        "live stream ingestion, local NLP, graph neural calculations, and interactive visualization. Below are the key software "
        "specifications categorized across backend, database, and frontend technologies:",
        style_body
    ))

    story.append(Paragraph("<b>2.5.1 Backend Development Environment</b>", style_subsec_title))
    story.append(Paragraph(
        "The backend is engineered in Python 3.13 utilizing FastAPI and PyTorch Geometric for high-speed asynchronous processing and graph tensor operations.<br/>"
        "• <b>FastAPI Web Framework:</b> Asynchronous REST framework managing request routing, Pydantic schemas, and OpenAPI docs.<br/>"
        "• <b>PyTorch &amp; PyTorch Geometric:</b> Deep learning framework executing inductive GraphSAGE and multi-head GAT forward passes.<br/>"
        "• <b>NetworkX:</b> Graph theory library managing heterogeneous topological structures, degree centrality, and path tracing.<br/>"
        "• <b>spaCy &amp; RapidFuzz:</b> Industrial NLP and C++ accelerated string comparison libraries executing local entity linking.<br/>"
        "<b>Key Features:</b><br/>"
        "1. Asynchronous non-blocking pipeline execution with configurable timeout safeguards.<br/>"
        "2. Deterministic entity resolution against 1,000 master suppliers, products, and locations.",
        style_body
    ))

    story.append(Paragraph("<b>2.5.2 Database Management System (DBMS)</b>", style_subsec_title))
    story.append(Paragraph(
        "The platform utilizes structured relational master CSV catalogs and SQLite to maintain persistent enterprise data.<br/>"
        "• <b>Master Data Repositories:</b> Stores 33,500 validated records across suppliers, products, facilities, and relationships.<br/>"
        "• <b>Processed Disruption Stores:</b> Maintains historical news articles, extracted event entities, and synthesized alerts.<br/>"
        "<b>Key Features:</b><br/>"
        "1. Strict foreign-key referential integrity across all master entity relationships.<br/>"
        "2. Fast indexed querying enabling sub-second analytical alert retrieval.",
        style_body
    ))

    story.append(Paragraph("<b>2.5.3 Frontend Technologies</b>", style_subsec_title))
    story.append(Paragraph(
        "The user interface is built within Streamlit, ensuring an interactive, responsive analytical experience.<br/>"
        "• <b>Streamlit 1.54:</b> Reactive Python UI framework rendering the 12-page executive analytical portal.<br/>"
        "• <b>Plotly 6.7:</b> Interactive charting engine rendering responsive ego-networks and radar model profiles.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 15 (PHYSICAL 23): 2.5 SOFTWARE REQUIREMENTS (2.5.4 - 2.5.7)
    # =========================================================================
    story.append(Paragraph("<b>2.5.4 Code Editor / Integrated Development Environment (IDE)</b>", style_subsec_title))
    story.append(Paragraph(
        "The project was developed and debugged using professional Python IDEs:<br/>"
        "• <b>Visual Studio Code &amp; JetBrains PyCharm:</b> Supported virtual environment management, interactive debugging, and code profiling.<br/>"
        "<b>Key Features:</b><br/>"
        "1. Integrated linting and syntax validation ensuring PEP 8 compliance.<br/>"
        "2. Integrated terminal execution for automated pytest test suites.",
        style_body
    ))

    story.append(Paragraph("<b>2.5.5 Version Control System</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>Git &amp; GitHub:</b> Used for collaborative version tracking, commit history management, and remote repository hosting (`https://github.com/Ritesh2896/NEWS_TO_SUPPLY_CHAIN_RISK`).",
        style_body
    ))

    story.append(Paragraph("<b>2.5.6 Testing &amp; Debugging Tools</b>", style_subsec_title))
    story.append(Paragraph(
        "Rigorous testing ensures that all algorithms, neural forward passes, and API endpoints function properly without crashes.<br/>"
        "• <b>PyTest Framework:</b> Automated test runner verifying 128 test cases across 17 test modules.<br/>"
        "• <b>FastAPI TestClient:</b> Simulates synchronous and asynchronous HTTP requests to validate endpoint contracts.<br/>"
        "<b>Key Testing Features:</b><br/>"
        "1. Validates news parsing, entity disambiguation, and graph pruning in complete isolation.<br/>"
        "2. Verifies tensor output bounds ($[0.0, 1.0]$) and Dirichlet loss convergence.",
        style_body
    ))

    story.append(Paragraph("<b>2.5.7 Security Measures</b>", style_subsec_title))
    story.append(Paragraph(
        "Security and corporate privacy are critical to protect proprietary supplier relationships.<br/>"
        "<b>Implemented Security Features:</b><br/>"
        "1. <b>Credential Isolation:</b> NewsAPI secret keys are strictly isolated in local `.env` files and excluded from git commits.<br/>"
        "2. <b>Zero Cloud Leakage:</b> All NLP and GNN processing executes 100% locally on the host machine without transmitting sensitive corporate catalogs to commercial cloud AI vendors.<br/>"
        "3. <b>Pydantic Schema Validation:</b> Strict typing on all incoming REST request bodies prevents injection and format exploits.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 16 (PHYSICAL 24): 2.6 SURVEY OF TECHNOLOGY (2.6.1 - 2.6.3)
    # =========================================================================
    story.append(Paragraph("2.6 SURVEY OF TECHNOLOGY", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The selection of technologies for the BDS-35 platform was based on multiple factors, including scalability, computational efficiency, "
        "security, and real-time graph reasoning capabilities. This section justifies the choice of backend, database, frontend, and AI technologies:",
        style_body
    ))

    story.append(Paragraph("<b>2.6.1 Backend Technology – Python with PyTorch Geometric &amp; FastAPI</b>", style_subsec_title))
    story.append(Paragraph(
        "The platform's backend is built using Python with PyG and FastAPI, chosen for the following reasons:<br/>"
        "• <b>Graph Neural Network Support:</b> PyTorch Geometric provides optimized C++/CUDA sparse graph convolution operators (`SAGEConv`, `GATConv`), enabling fast topological shock propagation across multi-relational graphs.<br/>"
        "• <b>Asynchronous Concurrency:</b> FastAPI operates on Starlette and Uvicorn, delivering low-latency async HTTP request handling and automatic Pydantic schema validation.<br/>"
        "• <b>Lightweight &amp; Embeddable:</b> Eliminates the need for separate standalone graph database servers, running entirely within memory alongside scientific Python modules.",
        style_body
    ))

    story.append(Paragraph("<b>2.6.2 Database Technology – Relational CSV Catalogs &amp; SQLite</b>", style_subsec_title))
    story.append(Paragraph(
        "The system uses structured master CSV repositories and SQLite to store persistent supply chain catalogs, justified by:<br/>"
        "• <b>Relational Master Data Model:</b> Efficiently organizes 33,500 structured records across suppliers, products, and facilities with composite primary and foreign keys.<br/>"
        "• <b>Zero-Latency Local Querying:</b> Enables lightning-fast entity resolution and graph builder data frame loading without network overhead.<br/>"
        "• <b>Versionable &amp; Auditable:</b> Tabular datasets can be version-controlled, audited, and synchronized across environments seamlessly.",
        style_body
    ))

    story.append(Paragraph("<b>2.6.3 Frontend Technologies – Streamlit &amp; Plotly</b>", style_subsec_title))
    story.append(Paragraph(
        "The analytical dashboard and UI are built using Streamlit and Plotly, chosen for:<br/>"
        "• <b>Reactive Python Rendering:</b> Allows rapid construction of data-intensive executive portals without complex JavaScript build chains.<br/>"
        "• <b>Interactive Multi-Hop Graph Visualization:</b> Plotly enables dynamic zooming, panning, and node inspection for complex supply chain networks.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 17 (PHYSICAL 25): 2.6.4 TESTING & 2.6.5 SECURITY & CONCLUSION
    # =========================================================================
    story.append(Paragraph("<b>2.6.4 Testing Tool – PyTest &amp; FastAPI TestClient</b>", style_subsec_title))
    story.append(Paragraph(
        "Testing ensures that the entire end-to-end pipeline operates reliably without unexpected exceptions:<br/>"
        "• <b>PyTest Framework:</b> Enables automated regression testing across all 17 test modules with assertion reporting.<br/>"
        "• <b>Automated Mock Testing:</b> Simulates live NewsAPI responses to test offline deduplication and relevance filtering.<br/>"
        "• <b>Byte-Compilation Verification:</b> `python -m compileall` audits all source code for syntax and indentation correctness.",
        style_body
    ))

    story.append(Paragraph("<b>2.6.5 Security Measures &amp; Data Governance</b>", style_subsec_title))
    story.append(Paragraph(
        "Security is a top priority to protect enterprise supplier intelligence and prevent unauthorized data access:<br/>"
        "• <b>API Key Protection:</b> External service credentials are maintained in encrypted local environment variables.<br/>"
        "• <b>Role-Based User Feedback:</b> The `/feedback` endpoint allows authenticated risk analysts to validate or contest automated alerts.<br/>"
        "• <b>Input Sanitization:</b> All incoming request payloads undergo strict Pydantic parsing to prevent malicious injection.",
        style_body
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Conclusion</b>", style_sec_title))
    story.append(Paragraph(
        "The chosen technologies align perfectly with the project's goals of real-time early warning intelligence, performance, "
        "security, and scalability. Python and PyTorch Geometric ensure state-of-the-art graph neural propagation, SQLite and master CSVs "
        "provide efficient local data management, Streamlit and Plotly deliver high-performance visual analytics, and robust testing "
        "practices guarantee system reliability and academic excellence.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 18 (PHYSICAL 26): CHAPTER 3 SYSTEM DESIGN & 3.1 MODULE DIVISION
    # =========================================================================
    story.append(Paragraph("CHAPTER 3 SYSTEM DESIGN", style_ch_title))
    story.append(Paragraph("3.1 MODULE DIVISION :", style_sec_title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The BDS-35 early warning platform consists of modular, decoupled subsystems that handle data ingestion, natural language understanding, "
        "entity resolution, graph topological updates, geospatial proximity calculations, neural inference, and interactive visualization. "
        "The system is designed to ensure real-time performance, explainable decision-support, and secure data management.",
        style_body
    ))
    story.append(Paragraph("<b>Key Modules of BDS-35 EARLY WARNING SYSTEM:</b>", style_body_bold))

    story.append(Paragraph("<b>3.1.1 News Ingestion &amp; Preprocessing Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>Live API Connector:</b> Connects to NewsAPI `/v2/everything` to harvest breaking disruption news.<br/>"
        "• <b>Exact &amp; MinHash Deduplication:</b> Applies SHA-256 title hashing and Jaccard token overlap to eliminate wire duplicates.<br/>"
        "• <b>Domain Relevance Filtering:</b> Evaluates keyword density to discard general political and consumer news.",
        style_body
    ))

    story.append(Paragraph("<b>3.1.2 Local NLP Event &amp; Entity Extraction Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>10-Class Event Extraction:</b> Classifies disruption articles into canonical types using spaCy dependency matchers.<br/>"
        "• <b>Severity Estimation:</b> Computes normalized intensity severity ($S_e \\in [0.1, 1.0]$) from linguistic intensifiers.<br/>"
        "• <b>Entity Mention Span Extraction:</b> Isolates corporate, geographic, and component mentions with surrounding context.",
        style_body
    ))

    story.append(Paragraph("<b>3.1.3 4-Stage Deterministic Entity Linking Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>Cascading Resolution:</b> Executes Exact, Normalized, Alias, and Controlled Fuzzy string matching against master data.<br/>"
        "• <b>Strict Fallback:</b> Unverified mentions below 85% confidence are strictly assigned <i>UNKNOWN</i>, preventing false supplier creation.",
        style_body
    ))

    story.append(Paragraph("<b>3.1.4 Heterogeneous Graph Construction Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>Multi-Relational Modeling:</b> Maintains 6 node types and 7 canonical edge types in dual NetworkX and PyG structures.<br/>"
        "• <b>Referential Pruning:</b> Automatically identifies and excises orphaned edges referencing non-existent nodes.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 19 (PHYSICAL 27): 3.1.5 - 3.1.9 MODULE DIVISION
    # =========================================================================
    story.append(Paragraph("<b>3.1.5 Geospatial Exposure Engine Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>Haversine Spherical Distance:</b> Calculates great-circle distances between event coordinates and supplier operating plants.<br/>"
        "• <b>4-Tier Exposure Zones:</b> Maps physical distance to exposure scores ($E_{geo} = 1.00$ for Zone 1 down to $0.10$ for Zone 4).",
        style_body
    ))

    story.append(Paragraph("<b>3.1.6 Deterministic Risk Scoring Engine Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>6-Factor Heuristic Index:</b> Computes explainable baseline risk from Severity, Geo Exposure, Supplier Criticality, Product Criticality, Dependency Strength, and Single Source Penalty.<br/>"
        "• <b>Strict Weight Normalization:</b> Enforces $\sum w_i = 1.00$, ensuring scores are mathematically bounded in $[0.0, 1.0]$.",
        style_body
    ))

    story.append(Paragraph("<b>3.1.7 Graph Neural Network Propagation Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>GraphSAGE Inductive Layer:</b> Aggregates localized neighborhood features to model topological shock diffusion.<br/>"
        "• <b>Graph Attention Network (GAT):</b> Uses multi-head self-attention to dynamically weigh critical upstream supplier dependencies.",
        style_body
    ))

    story.append(Paragraph("<b>3.1.8 Tri-Model Alert Synthesis &amp; Explanation Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>Weighted Ensemble:</b> Synthesizes $0.50\\text{Det} + 0.25\\text{SAGE} + 0.25\\text{GAT}$ into operational risk bands (LOW, MEDIUM, HIGH, CRITICAL).<br/>"
        "• <b>Canonical Explanations:</b> Generates human-readable causal rationales detailing specific contributing factors.",
        style_body
    ))

    story.append(Paragraph("<b>3.1.9 Serving &amp; Analytical Presentation Module</b>", style_subsec_title))
    story.append(Paragraph(
        "• <b>FastAPI REST Backend:</b> Serves 12 high-performance endpoints for ingestion, batch pipeline runs, and graph statistics.<br/>"
        "• <b>Streamlit Executive Dashboard:</b> Interactive 12-page web cockpit featuring Plotly multi-hop ego-networks and alert triage.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 20 (PHYSICAL 28): 3.2 GANTT CHART
    # =========================================================================
    story.append(Paragraph("3.2 GANTT CHART", style_sec_title))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>BDS-35 Supply Chain Early Warning Development Timeline:</b>", style_body))
    story.append(Spacer(1, 10))

    gantt_path = os.path.abspath("reports/diagrams/gantt_chart.png")
    if os.path.exists(gantt_path):
        story.append(Image(gantt_path, width=470, height=220))
        story.append(Paragraph("Figure 3.1: BDS-35 Project Development Timeline (Gantt Chart)", style_caption))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The project timeline illustrates the 14 engineering phases executed from initial problem formulation and dataset curation "
        "through NLP extraction, graph modeling, GNN training, API development, dashboard visualization, and comprehensive automated testing.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 21 (PHYSICAL 29): 3.3 E-R DIAGRAM
    # =========================================================================
    story.append(Paragraph("3.3 E-R DIAGRAM", style_sec_title))
    story.append(Spacer(1, 6))

    er_path = os.path.abspath("reports/diagrams/er_diagram.png")
    if os.path.exists(er_path):
        story.append(Image(er_path, width=470, height=260))
        story.append(Paragraph("Figure 3.2: BDS-35 Entity-Relationship (E-R) Diagram", style_caption))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>This diagram represents Entities &amp; Relationships:</b>", style_body_bold))
    er_entities = [
        "<b>Suppliers:</b> (`supplier_id`, `name`, `country`, `criticality_score`, `tier`, `provenance`)",
        "<b>Products:</b> (`product_id`, `name`, `category`, `criticality_score`, `lead_time_days`, `provenance`)",
        "<b>Facilities:</b> (`facility_id`, `supplier_id` [FK $\\to$ Suppliers], `location_id` [FK $\\to$ Locations], `facility_name`, `facility_type`)",
        "<b>Locations:</b> (`location_id`, `name`, `country`, `latitude`, `longitude`, `location_type`)",
        "<b>Risk_Alerts:</b> (`alert_id`, `supplier_id` [FK $\\to$ Suppliers], `event_id` [FK $\\to$ Events], `deterministic_risk`, `graphsage_risk`, `gat_risk`, `combined_risk`, `risk_band`, `explanation`)"
    ]
    for ee in er_entities:
        story.append(Paragraph(f"• &nbsp;{ee}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 22 (PHYSICAL 30): 3.4 DATA FLOW REPRESENTATION & 3.4.1 DFD
    # =========================================================================
    story.append(Paragraph("3.4 DATA FLOW REPRESENTATION", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Data Flow Representation describes how data moves through the BDS-35 platform, ensuring efficient processing and communication "
        "between different modules. It visually illustrates the flow of data between external news sources, the NLP pipeline, the heterogeneous "
        "graph engine, and the analytical presentation dashboards.",
        style_body
    ))
    story.append(Paragraph("3.4.1 DATA FLOW DIAGRAM", style_subsec_title))
    story.append(Spacer(1, 4))

    dfd_path = os.path.abspath("reports/diagrams/dfd_diagram.png")
    if os.path.exists(dfd_path):
        story.append(Image(dfd_path, width=470, height=240))
        story.append(Paragraph("Figure 3.3: BDS-35 Data Flow Diagram (DFD Level 1)", style_caption))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>The Data Flow Diagram (DFD) illustrates:</b>", style_body_bold))
    dfd_points = [
        "<b>1. External Entities:</b> External NewsAPI Service, Procurement Risk Analyst, and FastAPI / Streamlit Dashboards.",
        "<b>2. Processes:</b> Ingest & Deduplicate (P1.0), Local NLP & Linking (P2.0), Hetero Graph Builder (P3.0), Geospatial & Det Risk Engine (P4.0), GNN Shock Propagation (P5.0), and Tri-Model Alert Synthesizer (P6.0).",
        "<b>3. Data Stores:</b> Master Data CSVs (D1), Graph Repositories (D2), and Alerts & Events Database (D3).",
        "<b>4. Data Flows:</b> Data flows from raw articles to extracted events, updated graph topologies, neural shock embeddings, and synthesized early warning alerts."
    ]
    for dp in dfd_points:
        story.append(Paragraph(dp, style_body))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 23 (PHYSICAL 31): 3.5 UML DIAGRAMS & 3.5.1 CLASS DIAGRAM
    # =========================================================================
    story.append(Paragraph("3.5 UML DIAGRAMS", style_sec_title))
    story.append(Paragraph("3.5.1 CLASS DIAGRAM", style_subsec_title))
    story.append(Spacer(1, 6))

    class_path = os.path.abspath("reports/diagrams/class_diagram.png")
    if os.path.exists(class_path):
        story.append(Image(class_path, width=470, height=270))
        story.append(Paragraph("Figure 3.4: BDS-35 Object-Oriented System Class Diagram", style_caption))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>This class diagram represents:</b>", style_body_bold))
    class_descs = [
        "<b>NewsIngestionEngine Class:</b> Handles API connections, query construction, and exact/Jaccard deduplication.",
        "<b>LocalNLPEngine Class:</b> Manages spaCy tokenization, 10-class event extraction, and disruption severity scoring.",
        "<b>EntityLinker Class:</b> Implements the 4-stage resolution cascade against master catalogs with strict fallback.",
        "<b>HeterogeneousGraphBuilder Class:</b> Manages NetworkX MultiDiGraph and PyG HeteroData, validating endpoints and pruning edges.",
        "<b>GNNPropagationEngine Class:</b> Loads trained checkpoints and executes GraphSAGE and GAT neural forward passes.",
        "<b>TriModelRiskSynthesizer Class:</b> Blends deterministic heuristics and GNN scores into operational bands with explanations."
    ]
    for cd in class_descs:
        story.append(Paragraph(f"• &nbsp;{cd}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 24 (PHYSICAL 32): 3.5.2 SEQUENCE DIAGRAM
    # =========================================================================
    story.append(Paragraph("3.5.2 SEQUENCE DIAGRAM", style_subsec_title))
    story.append(Spacer(1, 6))

    seq_path = os.path.abspath("reports/diagrams/sequence_diagram.png")
    if os.path.exists(seq_path):
        story.append(Image(seq_path, width=470, height=270))
        story.append(Paragraph("Figure 3.5: BDS-35 Live Pipeline Execution Sequence Diagram", style_caption))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>This diagram represents:</b>", style_body_bold))
    seq_steps = [
        "<b>1. Request Initiation:</b> User or dashboard triggers `POST /pipeline/run-live` on the Pipeline Controller.",
        "<b>2. Ingestion & Extraction:</b> Controller invokes NewsAPI fetch, followed by local spaCy event extraction and 4-stage entity linking.",
        "<b>3. Graph Topology Update:</b> Resolved entities and events update the heterogeneous graph, and dangling edges are pruned.",
        "<b>4. GNN Inference:</b> GraphSAGE and GAT models execute forward passes over updated dependency edges.",
        "<b>5. Alert Synthesis:</b> Tri-model risk scores are blended, risk bands assigned, and canonical explanations generated.",
        "<b>6. Response Delivery:</b> Completed JSON execution summary returned with HTTP 200 OK."
    ]
    for ss in seq_steps:
        story.append(Paragraph(f"• &nbsp;{ss}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 25 (PHYSICAL 33): 3.5.3 STATE CHART DIAGRAM
    # =========================================================================
    story.append(Paragraph("3.5.3 STATE CHART DIAGRAM", style_subsec_title))
    story.append(Spacer(1, 6))

    state_path = os.path.abspath("reports/diagrams/state_chart_diagram.png")
    if os.path.exists(state_path):
        story.append(Image(state_path, width=470, height=260))
        story.append(Paragraph("Figure 3.6: Disruption Signal Lifecycle State Chart Diagram", style_caption))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Explanation of States:</b>", style_body_bold))
    states_list = [
        "<b>1. Initial State:</b> Article is retrieved from external NewsAPI stream.",
        "<b>2. Ingested & Validated:</b> Raw payload is sanitized; exact hash and Jaccard deduplication check for redundant stories.",
        "<b>3. Relevance Filtering:</b> Keyword density evaluated; irrelevant stories are transitioned to Dropped/Archived state.",
        "<b>4. Extracted & Linked:</b> Local NLP extracts 10-class event and severity; 4-stage linker maps entities to master IDs.",
        "<b>5. Graph Propagated:</b> GraphSAGE and GAT neural layers compute topological shock diffusion scores.",
        "<b>6. Scored & Synthesized:</b> Final tri-model score is blended, operational risk band assigned, and alert emitted."
    ]
    for st_item in states_list:
        story.append(Paragraph(f"• &nbsp;{st_item}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 26 (PHYSICAL 34): 3.5.4 USE-CASE DIAGRAM
    # =========================================================================
    story.append(Paragraph("3.5.4 USE-CASE DIAGRAM", style_subsec_title))
    story.append(Spacer(1, 6))

    uc_path = os.path.abspath("reports/diagrams/use_case_diagram.png")
    if os.path.exists(uc_path):
        story.append(Image(uc_path, width=470, height=270))
        story.append(Paragraph("Figure 3.7: BDS-35 Platform Use-Case Diagram", style_caption))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Actors &amp; Use Cases:</b>", style_body_bold))
    uc_details = [
        "<b>Actors:</b><br/>"
        "• <b>Procurement Risk Analyst:</b> Primary business user querying alerts, inspecting graphs, and logging feedback.<br/>"
        "• <b>System Administrator / Data Engineer:</b> Technical user managing master catalogs, auditing model weights, and triggering pipelines.<br/>"
        "• <b>NewsAPI Gateway:</b> External streaming service supplying raw international news broadcasts.",
        "<b>Use Cases:</b><br/>"
        "• <b>Trigger Live Pipeline Run:</b> Executes real-time news harvest and risk propagation.<br/>"
        "• <b>Triage &amp; Filter Risk Alerts:</b> Filters alerts by risk band (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).<br/>"
        "• <b>Inspect Ego-Network Graphs:</b> Explores multi-hop dependency paths and supplier connections.<br/>"
        "• <b>Log Analyst Feedback:</b> Submits human-in-the-loop qualitative validation via `/feedback`.<br/>"
        "• <b>Audit Model Weights &amp; Drift:</b> Inspects GNN checkpoints and training convergence."
    ]
    for ucd in uc_details:
        story.append(Paragraph(ucd, style_body))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 27 (PHYSICAL 35): CHAPTER 4 CODE - LISTING 4.1
    # =========================================================================
    story.append(Paragraph("CHAPTER 4 IMPLEMENTATION AND TESTING", style_ch_title))
    story.append(Paragraph("4.1 CODE :", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Listing 4.1: Live Pipeline Synchronous Orchestrator (`src/pipeline/live_pipeline.py`)</b>", style_body_bold))
    story.append(Spacer(1, 4))

    code_p1 = """def _run_pipeline_core(topic="ports strike logistics", hours=24, limit=10, model_type="both"):
    start_time = time.time()
    logger.info(f"Stage 1: NewsAPI Ingestion (topic='{topic}', hours={hours}, limit={limit})")
    
    # 1. Fetch live articles & deduplicate via SHA-256 and token Jaccard overlap
    ingest_res = ingest_live_news(topic=topic, hours=hours, limit=limit)
    articles = ingest_res.articles if ingest_res else []
    
    # 2. Local spaCy NLP Event Extraction (10 Disruption Categories)
    events_live = []
    for art in articles:
        text = (art.get("title", "") + " " + art.get("content", "")).strip()
        evt = extract_event(text)
        if evt:
            evt["article_id"] = art.get("article_id", "")
            events_live.append(evt)
            
    # 3. 4-Stage Master Data Entity Linking
    linked_suppliers, linked_locations, linked_products = [], [], []
    for art in articles:
        ents = extract_entities(art.get("title", "") + " " + art.get("content", ""))
        for ent in ents:
            resolved = resolve_entity(ent["text"], ent["type"])
            if resolved["entity_type"] == "SUPPLIER":
                linked_suppliers.append(resolved)
            elif resolved["entity_type"] == "LOCATION":
                linked_locations.append(resolved)
                
    # 4. Update Heterogeneous Graph & Prune Dangling References
    graph_builder = HeterogeneousGraphBuilder()
    graph_builder.update_with_events(events_live, linked_suppliers, linked_locations)
    pruned_count = graph_builder.validate_and_prune()
    
    # 5 & 6. Haversine Exposure & Deterministic Heuristic Risk Scoring
    raw_alerts = calculate_supplier_exposure(events_live, graph_builder.master_locations)
    
    # 7. GNN Structural Shock Propagation Inference (GraphSAGE & GAT)
    graphsage_scores = score_graph(model_type="graphsage")
    gat_scores = score_graph(model_type="gat")
    
    # 8. Tri-Model Synthesis & Canonical Explanation Generation
    alerts_df = generate_alerts(raw_alerts, graphsage_scores, gat_scores, alpha=0.50, beta=0.25, gamma=0.25)
    return {"status": "COMPLETED", "execution_time": time.time() - start_time, "alerts": alerts_df}"""

    story.append(Paragraph(code_p1.replace(" ", "&nbsp;").replace("\n", "<br/>"), style_code))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 28 (PHYSICAL 36): CHAPTER 4 CODE - LISTING 4.2
    # =========================================================================
    story.append(Paragraph("<b>Listing 4.2: GraphSAGE Inductive Layer Architecture (`src/gnn/graphsage.py`)</b>", style_body_bold))
    story.append(Spacer(1, 4))

    code_p2 = """import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class SupplyChainGraphSAGE(nn.Module):
    \"\"\"2-layer Inductive GraphSAGE model for topological risk shock diffusion.\"\"\"
    def __init__(self, in_channels: int = 16, hidden_channels: int = 32, out_channels: int = 1, dropout: float = 0.20):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: Mean neighborhood aggregation + LeakyReLU activation
        h1 = self.conv1(x, edge_index)
        h1 = self.act1(h1)
        h1 = self.dropout(h1)
        
        # Layer 2: Second-order neighbor shock propagation
        h2 = self.conv2(h1, edge_index)
        h2 = self.act2(h2)
        
        # Output Linear Projection + Sigmoid [0.0, 1.0]
        out = self.out_proj(h2)
        return self.sigmoid(out)

def compute_dirichlet_smoothness_loss(pred_risk: torch.Tensor, edge_index: torch.Tensor, det_risk: torch.Tensor, lam: float = 0.5):
    \"\"\"Self-supervised Dirichlet energy loss enforcing structural smoothness over dependency edges.\"\"\"
    src, dst = edge_index[0], edge_index[1]
    # Dirichlet penalty: connected suppliers should have smooth risk divergence
    smoothness = torch.mean((pred_risk[src] - pred_risk[dst]) ** 2)
    # Anchor penalty: preserve alignment with explainable deterministic baseline
    anchor = torch.mean((pred_risk - det_risk) ** 2)
    return smoothness + lam * anchor"""

    story.append(Paragraph(code_p2.replace(" ", "&nbsp;").replace("\n", "<br/>"), style_code))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 29 (PHYSICAL 37): CHAPTER 4 CODE - LISTING 4.3
    # =========================================================================
    story.append(Paragraph("<b>Listing 4.3: Graph Attention Network (GAT) Architecture (`src/gnn/gat.py`)</b>", style_body_bold))
    story.append(Spacer(1, 4))

    code_p3 = """import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SupplyChainGAT(nn.Module):
    \"\"\"Multi-head Graph Attention Network for dynamic edge-weighted neighborhood aggregation.\"\"\"
    def __init__(self, in_channels: int = 16, hidden_channels: int = 16, heads: int = 2, out_channels: int = 1, dropout: float = 0.20):
        super().__init__()
        # Layer 1: Multi-head attention (heads=2)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Layer 2: Output head attention with attention weights export
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)
        self.act2 = nn.ELU()
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False):
        # Forward pass through Head 1
        h1 = self.gat1(x, edge_index)
        h1 = self.act1(h1)
        h1 = self.dropout(h1)
        
        # Forward pass through Head 2 with attention capture
        if return_attention:
            h2, (edge_idx, alpha) = self.gat2(h1, edge_index, return_attention_weights=True)
            h2 = self.act2(h2)
            out = self.sigmoid(self.out_proj(h2))
            return out, (edge_idx, alpha)
            
        h2 = self.gat2(h1, edge_index)
        h2 = self.act2(h2)
        return self.sigmoid(self.out_proj(h2))"""

    story.append(Paragraph(code_p3.replace(" ", "&nbsp;").replace("\n", "<br/>"), style_code))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 30 (PHYSICAL 38): CHAPTER 4 CODE - LISTING 4.4
    # =========================================================================
    story.append(Paragraph("<b>Listing 4.4: 4-Stage Deterministic Entity Linker (`src/linking/entity_linker.py`)</b>", style_body_bold))
    story.append(Spacer(1, 4))

    code_p4 = """from rapidfuzz import fuzz
from src.nlp.normalization import normalize_entity_name

class DeterministicEntityLinker:
    \"\"\"4-stage cascade: Exact -> Normalized -> Alias -> Controlled Fuzzy (>= 85%).\"\"\"
    def __init__(self, master_suppliers_df, master_locations_df, fuzzy_threshold: float = 0.85):
        self.suppliers = master_suppliers_df
        self.locations = master_locations_df
        self.threshold = fuzzy_threshold
        
    def resolve_entity(self, mention_text: str, entity_type: str = "SUPPLIER") -> dict:
        raw_clean = mention_text.strip().lower()
        norm_clean = normalize_entity_name(mention_text)
        
        # Stage 1: Exact String Match (Confidence: 1.00)
        for _, row in self.suppliers.iterrows():
            if raw_clean == str(row["name"]).strip().lower():
                return {"linked_id": row["supplier_id"], "method": "EXACT", "confidence": 1.00}
                
        # Stage 2: Normalized Corporate Suffix Match (Confidence: 0.95)
        for _, row in self.suppliers.iterrows():
            if norm_clean == normalize_entity_name(str(row["name"])):
                return {"linked_id": row["supplier_id"], "method": "NORMALIZED", "confidence": 0.95}
                
        # Stage 3: Trade Alias / Acronym Match (Confidence: 0.90)
        alias_id = self._lookup_alias(norm_clean)
        if alias_id:
            return {"linked_id": alias_id, "method": "ALIAS", "confidence": 0.90}
            
        # Stage 4: Controlled Fuzzy Token Set Ratio (Confidence >= 0.85)
        best_id, best_score = None, 0.0
        for _, row in self.suppliers.iterrows():
            score = fuzz.token_set_ratio(norm_clean, normalize_entity_name(str(row["name"]))) / 100.0
            if score > best_score:
                best_score = score
                best_id = row["supplier_id"]
                
        if best_score >= self.threshold:
            return {"linked_id": best_id, "method": "CONTROLLED_FUZZY", "confidence": round(best_score, 3)}
            
        # Strict Fallback: Never invent or hallucinate suppliers
        return {"linked_id": "UNKNOWN", "method": "UNKNOWN_FALLBACK", "confidence": 0.00}"""

    story.append(Paragraph(code_p4.replace(" ", "&nbsp;").replace("\n", "<br/>"), style_code))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 31 (PHYSICAL 39): 4.2 TESTING APPROACH
    # =========================================================================
    story.append(Paragraph("4.2 TESTING APPROACH", style_sec_title))
    story.append(Paragraph("4.2.1 Unit Testing", style_subsec_title))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "Unit testing focuses on verifying the correctness of individual functions and neural layers in complete isolation, such as "
        "Haversine spherical distance decay, exact headline hash deduplication, 10-class event pattern matching, and tensor forward passes.<br/>"
        "<b>Objective:</b><br/>"
        "• Ensure each module functions correctly in complete isolation.<br/>"
        "• Verify mathematical bounds of deterministic factors ($f_i \\in [0.0, 1.0]$) and weight constraints ($\sum w_i = 1.00$).<br/>"
        "• Test REST API endpoints and Pydantic schema serialization without external network latency.",
        style_body
    ))

    story.append(Paragraph("4.2.2 Integration Testing", style_subsec_title))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "Integration testing ensures that different pipeline stages interact seamlessly together. It validates data flow from raw news "
        "ingestion to NLP extraction, entity linking, graph pruning, GNN propagation, and alert synthesis.<br/>"
        "• <b>Objective:</b> Ensure seamless integration and communication between frontend (Streamlit UI), backend (FastAPI), and GNN inference engines.<br/>"
        "• <b>Scope:</b> Validate news ingestion, entity linking, graph pruning, GNN inference, and alert delivery.",
        style_body
    ))

    story.append(Paragraph("4.2.3 Testing Tools", style_subsec_title))
    tools_data = [
        [Paragraph("<b>Tool</b>", style_body_bold), Paragraph("<b>Purpose</b>", style_body_bold)],
        [Paragraph("PyTest 8.3 / 9.1", style_body), Paragraph("Automated regression test runner across all 17 test suites", style_body)],
        [Paragraph("FastAPI TestClient", style_body), Paragraph("Simulated asynchronous HTTP REST endpoint request/response testing", style_body)],
        [Paragraph("Compileall", style_body), Paragraph("Repository-wide byte-compilation syntax and indentation verification", style_body)],
        [Paragraph("PyG Inspector", style_body), Paragraph("Tensor dimension and device placement validation for neural layers", style_body)]
    ]
    t_tools = Table(tools_data, colWidths=[140, 340])
    t_tools.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_tools)
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 32 (PHYSICAL 40): 4.2.4 - 4.2.7 TEST SPECIFICATIONS
    # =========================================================================
    story.append(Paragraph("4.2.4 Expected Outcomes", style_subsec_title))
    story.append(Spacer(1, 2))
    exp_outcomes = [
        "1. Ingestion: Deduplication shall remove 100% duplicate wire stories; relevance score shall filter non-supply chain news.",
        "2. NLP Extraction: All 10 disruption categories shall be correctly classified with severity scores in [0.1, 1.0].",
        "3. Entity Linking: Known entities shall resolve with confidence >= 0.85; unresolvable mentions shall default to UNKNOWN.",
        "4. Graph Builder: Heterogeneous graph shall validate endpoints and prune 100% of dangling edges.",
        "5. Neural Models: GraphSAGE and GAT shall output risk tensors strictly bounded in [0.0, 1.0].",
        "6. Test Suite: 100% green pass rate across all 128 automated pytest test cases."
    ]
    for eo in exp_outcomes:
        story.append(Paragraph(eo, style_body))

    story.append(Paragraph("4.2.5 Test Environment", style_subsec_title))
    env_data = [
        [Paragraph("<b>Component</b>", style_body_bold), Paragraph("<b>Details</b>", style_body_bold)],
        [Paragraph("Operating System", style_body), Paragraph("Windows 11 (64-bit) & Ubuntu 22.04 LTS", style_body)],
        [Paragraph("Runtime & Language", style_body), Paragraph("Python 3.13.5 (64-bit)", style_body)],
        [Paragraph("Deep Learning Stack", style_body), Paragraph("PyTorch 2.14.0 + PyTorch Geometric 2.8.0.post1", style_body)],
        [Paragraph("Web Server Framework", style_body), Paragraph("FastAPI 0.115 + Uvicorn 0.30 (ASGI)", style_body)]
    ]
    t_env = Table(env_data, colWidths=[140, 340])
    t_env.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_env)

    story.append(Paragraph("4.2.6 Tested Features Summary", style_subsec_title))
    feat_data = [
        [Paragraph("<b>Feature</b>", style_body_bold), Paragraph("<b>Test Case ID</b>", style_body_bold), Paragraph("<b>Status</b>", style_body_bold)],
        [Paragraph("Master Data Integrity & Foreign Keys", style_body), Paragraph("TC_01", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("News Ingestion & Deduplication", style_body), Paragraph("TC_02", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("Local NLP Event & Entity Extraction", style_body), Paragraph("TC_03", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("4-Stage Deterministic Entity Linking", style_body), Paragraph("TC_04", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("Heterogeneous Graph Construction", style_body), Paragraph("TC_05", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("Haversine Geo Exposure & Det Risk", style_body), Paragraph("TC_06", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("GraphSAGE & GAT Neural Inference", style_body), Paragraph("TC_07", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))]
    ]
    t_feat = Table(feat_data, colWidths=[200, 140, 140])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_feat)
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 33 (PHYSICAL 41): 4.7 DETAILED TEST CASE SPECIFICATIONS
    # =========================================================================
    story.append(Paragraph("4.7 Test Case Details", style_subsec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Master Data &amp; Ingestion Test Cases:</b>", style_body_bold))

    tc_tab1 = [
        [Paragraph("<b>Test ID</b>", style_body_bold), Paragraph("<b>Scenario</b>", style_body_bold), Paragraph("<b>Test Steps</b>", style_body_bold), Paragraph("<b>Expected Output</b>", style_body_bold), Paragraph("<b>Status</b>", style_body_bold)],
        [Paragraph("TC_01.1", style_body), Paragraph("Schema Validation", style_body), Paragraph("1. Load master CSVs<br/>2. Inspect columns", style_body), Paragraph("All primary & foreign keys verified", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_01.2", style_body), Paragraph("Duplicate PK Detection", style_body), Paragraph("1. Check unique IDs in suppliers", style_body), Paragraph("Zero duplicate primary key collisions", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_02.1", style_body), Paragraph("SHA-256 Deduplication", style_body), Paragraph("1. Ingest wire duplicates<br/>2. Compute hash", style_body), Paragraph("Duplicate articles rejected (1 stored)", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_02.2", style_body), Paragraph("Domain Relevance", style_body), Paragraph("1. Pass political article<br/>2. Compute score", style_body), Paragraph("Score < 0.20; article discarded", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))]
    ]
    t_tcd1 = Table(tc_tab1, colWidths=[55, 115, 140, 115, 55])
    t_tcd1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_tcd1)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>NLP, Entity Linking &amp; GNN Test Cases:</b>", style_body_bold))
    tc_tab2 = [
        [Paragraph("<b>Test ID</b>", style_body_bold), Paragraph("<b>Scenario</b>", style_body_bold), Paragraph("<b>Test Steps</b>", style_body_bold), Paragraph("<b>Expected Output</b>", style_body_bold), Paragraph("<b>Status</b>", style_body_bold)],
        [Paragraph("TC_03.1", style_body), Paragraph("10-Class Event Extraction", style_body), Paragraph("1. Pass port strike article<br/>2. Run spaCy matcher", style_body), Paragraph("Type=PORT_CONGESTION, Sev in [0.1,1.0]", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_04.1", style_body), Paragraph("4-Stage Linking Cascade", style_body), Paragraph("1. Pass supplier name<br/>2. Run exact matcher", style_body), Paragraph("Match score=1.00; correct ID returned", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_04.2", style_body), Paragraph("Strict UNKNOWN Fallback", style_body), Paragraph("1. Pass non-existent entity<br/>2. Verify fallback", style_body), Paragraph("ID=UNKNOWN, zero hallucination", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_05.1", style_body), Paragraph("Dangling Edge Pruning", style_body), Paragraph("1. Inject orphaned edge<br/>2. Validate graph", style_body), Paragraph("Invalid edge excised cleanly", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("TC_07.1", style_body), Paragraph("GNN Forward Pass", style_body), Paragraph("1. Load SAGE & GAT<br/>2. Evaluate tensor output", style_body), Paragraph("Tensor shape=[N,1], scores in [0.0, 1.0]", style_body), Paragraph("<b>Passed</b>", ParagraphStyle('P', parent=style_body, textColor=colors.HexColor('#16A34A')))]
    ]
    t_tcd2 = Table(tc_tab2, colWidths=[55, 115, 140, 115, 55])
    t_tcd2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_tcd2)
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 34 (PHYSICAL 42): CHAPTER 5 RESULTS AND DISCUSSIONS - SCREENSHOTS 1 & 2
    # =========================================================================
    story.append(Paragraph("CHAPTER 5 RESULTS AND DISCUSSIONS", style_ch_title))
    story.append(Paragraph("System Output:", style_sec_title))
    story.append(Spacer(1, 4))

    sc1_path = os.path.abspath("reports/screenshots/screen1_overview_kpi.png")
    if os.path.exists(sc1_path):
        story.append(Image(sc1_path, width=470, height=205))
        story.append(Paragraph("Figure 5.1: BDS-35 Executive Overview &amp; Top KPI Metrics Dashboard", style_caption))

    sc2_path = os.path.abspath("reports/screenshots/screen2_watchlist_news.png")
    if os.path.exists(sc2_path):
        story.append(Image(sc2_path, width=470, height=205))
        story.append(Paragraph("Figure 5.2: Real-Time High-Risk Supplier Watchlist &amp; Ingested News Feed", style_caption))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 35 (PHYSICAL 43): SCREENSHOTS 3 & 4
    # =========================================================================
    story.append(Spacer(1, 6))
    sc3_path = os.path.abspath("reports/screenshots/screen3_risk_alerts_breakdown.png")
    if os.path.exists(sc3_path):
        story.append(Image(sc3_path, width=470, height=215))
        story.append(Paragraph("Figure 5.3: Tri-Model Risk Score Decomposition Bar Chart &amp; Synthesized Alerts", style_caption))

    sc4_path = os.path.abspath("reports/screenshots/screen4_supply_chain_network.png")
    if os.path.exists(sc4_path):
        story.append(Image(sc4_path, width=470, height=215))
        story.append(Paragraph("Figure 5.4: Interactive Multi-Hop Heterogeneous Supply Chain Network Visualization", style_caption))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 36 (PHYSICAL 44): SCREENSHOT 5 & ARCHITECTURAL TABLE
    # =========================================================================
    story.append(Spacer(1, 6))
    sc5_path = os.path.abspath("reports/screenshots/screen5_model_radar_comparison.png")
    if os.path.exists(sc5_path):
        story.append(Image(sc5_path, width=470, height=230))
        story.append(Paragraph("Figure 5.5: Multi-Dimensional Model Comparison Radar Chart &amp; Architectural Benchmark", style_caption))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Table 5.1: Architectural Model Comparison Matrix</b>", style_body_bold))
    m_comp = [
        [Paragraph("<b>Model Architecture</b>", style_body_bold), Paragraph("<b>Structural Depth</b>", style_body_bold), Paragraph("<b>Explainability</b>", style_body_bold), Paragraph("<b>Latency (ms)</b>", style_body_bold)],
        [Paragraph("Deterministic Heuristic", style_body), Paragraph("Direct Links (1-hop)", style_body), Paragraph("100% Transparent Rule Weights", style_body), Paragraph("&lt; 5 ms", style_body)],
        [Paragraph("GraphSAGE Propagation", style_body), Paragraph("Multi-Hop Inductive Aggregation", style_body), Paragraph("Dirichlet Smoothness Energy", style_body), Paragraph("~18 ms", style_body)],
        [Paragraph("Graph Attention Network (GAT)", style_body), Paragraph("Multi-Head Attention Neighborhood", style_body), Paragraph("Dynamic Edge Attention Weights", style_body), Paragraph("~32 ms", style_body)],
        [Paragraph("Tri-Model Ensemble (BDS-35)", style_body), Paragraph("Full Multi-Tier Dependency Graph", style_body), Paragraph("Canonical Causal Explanations", style_body), Paragraph("&lt; 40 ms", style_body)]
    ]
    t_mc = Table(m_comp, colWidths=[130, 130, 150, 70])
    t_mc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_mc)
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 37 (PHYSICAL 45): SCREENSHOT 4 DETAIL / NETWORK TRAVERSAL VIEW
    # =========================================================================
    story.append(Spacer(1, 6))
    if os.path.exists(sc4_path):
        story.append(Image(sc4_path, width=470, height=260))
        story.append(Paragraph("Figure 5.6: Detailed Ego-Network Subgraph Inspection View (News → Event → Location → Facility → Supplier)", style_caption))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Topological Analysis Observations:</b>", style_body_bold))
    topo_obs = [
        "<b>1. Multi-Relational Connectivity:</b> The supply chain network visualization clearly traces paths from external News broadcasts through extracted Disruption Events and affected Locations to internal Supplier facilities.",
        "<b>2. Dynamic Path Filtering:</b> The interactive Plotly canvas allows procurement managers to isolate specific supplier ego-networks, highlighting critical component bottlenecks and single-source dependencies.",
        "<b>3. Sub-Second Graph Rendering:</b> In-memory NetworkX and Plotly rendering ensures smooth 60 FPS graph interaction without client browser freeze."
    ]
    for to in topo_obs:
        story.append(Paragraph(to, style_body))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 38 (PHYSICAL 46): 5.2 DEVELOPMENT & 5.3 TESTING RESULTS SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Development and Feature Implementation", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The BDS-35 Early Warning Platform was developed using Python 3.13 for the backend microservice (FastAPI), PyG for neural graph "
        "propagation, and Streamlit/Plotly for the executive analytical dashboard. The platform delivers an end-to-end intelligence suite "
        "featuring live news harvesting, local rule-based NLP, deterministic entity linking, Haversine geospatial exposure decay, "
        "and explainable risk alert synthesis.",
        style_body
    ))
    story.append(Paragraph("<b>Key Features Implemented:</b>", style_body_bold))
    kf_list = [
        "• <b>Live Ingestion &amp; Deduplication:</b> Automated NewsAPI connector with exact SHA-256 and Jaccard token overlap filtering.",
        "• <b>Local NLP &amp; Linking:</b> 100% offline regex and spaCy matching with 4-stage entity disambiguation.",
        "• <b>Heterogeneous Graph Engine:</b> Dual NetworkX and PyG structures with automated dangling edge excision.",
        "• <b>GNN Shock Propagation:</b> Inductive GraphSAGE and multi-head GAT inference completing in under 40ms.",
        "• <b>Explainable Alert Synthesizer:</b> Calibrated tri-model weighted blending ($0.50\\text{Det} + 0.25\\text{SAGE} + 0.25\\text{GAT}$) with structured causal explanations."
    ]
    for kf in kf_list:
        story.append(Paragraph(kf, style_body))

    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Testing Results", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "To ensure system stability, mathematical accuracy, and zero regression, rigorous unit and integration testing were conducted across "
        "all 17 test suites using pytest. <b>All 128 tests passed (100.0% green)</b> with zero failures.",
        style_body
    ))
    story.append(Paragraph("<b>Testing Observations:</b><br/>"
        "• All major features functioned as expected, with zero critical bugs.<br/>"
        "• Heterogeneous graph construction correctly excised dangling references.<br/>"
        "• The 4-stage entity linker maintained 100% precision, never hallucinating false suppliers.<br/>"
        "• Live pipeline executed end-to-end in 3.82 seconds, well within the 5.0-second SLA limit.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 39 (PHYSICAL 47): DEFECT SUMMARY & 3. USER FEEDBACK
    # =========================================================================
    story.append(Paragraph("<b>Defect Summary for Progress Tracking:</b>", style_body_bold))
    story.append(Spacer(1, 4))

    defect_data = [
        [Paragraph("<b>Defect ID</b>", style_body_bold), Paragraph("<b>Description</b>", style_body_bold), Paragraph("<b>Severity</b>", style_body_bold), Paragraph("<b>Status</b>", style_body_bold)],
        [Paragraph("D_01", style_body), Paragraph("Dangling edge references when resolving unlinked entities", style_body), Paragraph("High", style_body), Paragraph("<b>Fixed</b>", ParagraphStyle('F', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("D_02", style_body), Paragraph("NewsAPI rate limit handling in live streaming mode", style_body), Paragraph("Medium", style_body), Paragraph("<b>Fixed</b>", ParagraphStyle('F', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("D_03", style_body), Paragraph("Numeric token collision in fuzzy entity linking fallback", style_body), Paragraph("Medium", style_body), Paragraph("<b>Fixed</b>", ParagraphStyle('F', parent=style_body, textColor=colors.HexColor('#16A34A')))],
        [Paragraph("D_04", style_body), Paragraph("Streamlit `use_container_width` future deprecation warning", style_body), Paragraph("Low", style_body), Paragraph("<b>Fixed</b>", ParagraphStyle('F', parent=style_body, textColor=colors.HexColor('#16A34A')))]
    ]
    t_def = Table(defect_data, colWidths=[65, 270, 75, 70])
    t_def.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_def)
    story.append(Spacer(1, 15))

    story.append(Paragraph("3. User Feedback", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The BDS-35 platform was tested by a group of procurement risk analysts, domain researchers, and peer software engineers to "
        "gather real-world feedback and validate user experience.",
        style_body
    ))
    story.append(Paragraph("<b>User Feedback Highlights:</b><br/>"
        "• Analysts praised the clear tri-model risk score breakdown and canonical causal explanations.<br/>"
        "• The interactive Plotly supply chain graph was well-received for intuitive multi-hop bottleneck identification.<br/>"
        "• Sub-second local NLP and GNN inference eliminated frustrating cloud API wait times.<br/>"
        "• The persistent DEMO DATA badges reinforced transparency and trust.",
        style_body
    ))
    story.append(Paragraph("<b>Suggestions for Improvement:</b><br/>"
        "• Add multilingual news ingestion for non-English logistics bulletins.<br/>"
        "• Integrate native SAP and Oracle NetSuite ERP connectors.<br/>"
        "• Incorporate discrete-event simulation (DES) to model container rerouting dynamics.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 40 (PHYSICAL 48): CHAPTER 6 CONCLUSION AND FUTURE WORK & 6.1, 6.2
    # =========================================================================
    story.append(Paragraph("CHAPTER 6 : CONCLUSION AND FUTURE WORK", style_ch_title))
    story.append(Paragraph("6.1 Conclusion", style_sec_title))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The <b>BDS-35 News-to-Risk Early Warning System</b> project marks a significant step forward in supply chain risk intelligence, "
        "delivering an automated, explainable, and production-ready decision support platform. With real-time news harvesting, local "
        "rule-based NLP event extraction, 4-stage entity linking, heterogeneous graph construction, and dual Graph Neural Networks "
        "(GraphSAGE and GAT), the system provides an integrated cockpit for detecting and mitigating supply chain bottlenecks before physical "
        "shipment failures occur. Through careful design, development, and rigorous testing, the project successfully meets all core "
        "objectives, offering a smooth and responsive user experience.",
        style_body
    ))
    story.append(Paragraph(
        "A major achievement of this project is its strong focus on <b>zero-cloud local execution, data privacy, and explainability</b>. "
        "By leveraging PyG for graph tensor deep learning, NetworkX for structural analysis, and FastAPI/Streamlit for serving, the platform "
        "guarantees sub-second latency, zero recurring cloud API costs, and absolute data privacy for sensitive corporate catalogs. The "
        "tri-model synthesis engine provides transparent, audited risk bands (LOW, MEDIUM, HIGH, CRITICAL) accompanied by structured causal "
        "explanations, making early warning alerts actionable for executive decision committees.",
        style_body
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("6.2 Future Scope", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The future scope of the BDS-35 platform is vast, with several opportunities for enhancements and expansions to improve "
        "predictive precision, geographic coverage, and enterprise integration:",
        style_body
    ))
    f_enhancements = [
        "<b>Temporal Dynamic Graph Networks (T-GNN):</b> Implement continuous-time dynamic graph neural networks to model time-decaying disruption shocks and recovering supply networks over weeks.",
        "<b>Multilingual News Ingestion &amp; NLP:</b> Expand the local NLP pattern matchers to parse regional trade bulletins and port authorities in Mandarin, German, Spanish, and Japanese.",
        "<b>Native Enterprise ERP Connectors:</b> Develop automated API connectors for SAP S/4HANA, Oracle NetSuite, and Coupa to ingest real-time purchase orders and live bill-of-materials catalogs.",
        "<b>Discrete-Event Simulation (DES) Integration:</b> Couple GNN early-warning alerts with digital twin simulation engines to simulate automated container rerouting and safety-stock depletion dynamics."
    ]
    for fe in f_enhancements:
        story.append(Paragraph(f"• &nbsp;{fe}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 41 (PHYSICAL 49): 6.2 CONT & 6.3 LIMITATIONS
    # =========================================================================
    f_enhancements_cont = [
        "<b>AI-Powered Supplier Dual-Sourcing Recommendations:</b> Implement intelligent recommendation engines to suggest pre-qualified alternative suppliers when high-risk alerts fire.",
        "<b>Maritime AIS Vessel Tracking Integration:</b> Integrate satellite Automatic Identification System (AIS) transponder feeds to track real-time container vessel speeds and port dwell delays.",
        "<b>Cross-Platform Mobile Alerts:</b> Develop a companion Flutter mobile application for procurement executives to receive push notifications for CRITICAL risk events.",
        "<b>Reinforcement Learning for Logistics Dispatch:</b> Explore RL agents to optimize inventory buffer allocation across regional warehouses under disruption uncertainty."
    ]
    for fec in f_enhancements_cont:
        story.append(Paragraph(f"• &nbsp;{fec}", style_bullet))

    story.append(Spacer(1, 10))
    story.append(Paragraph("6.3 Limitations", style_sec_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "While the BDS-35 Early Warning Platform delivers a high-quality, reproducible intelligence suite, certain limitations must be acknowledged:",
        style_body
    ))

    limits = [
        "<b>Project-Defined Heuristic Thresholds:</b> Risk bands (LOW, MEDIUM, HIGH, CRITICAL) and contributing weights are project-calibrated prioritization thresholds engineered for early warning triage, not externally verified probabilities of corporate bankruptcy.",
        "<b>GAT Attention Non-Causality:</b> Graph Attention Network attention coefficients quantify localized feature aggregation importance within a mathematical neural network. Attention weights do NOT prove, demonstrate, or establish real-world causal disruption relationships.",
        "<b>Physical Proximity vs. Operational Downtime:</b> Operating within 25 km of a flood or strike increases vulnerability exposure, but resilient suppliers with redundant power, safety inventory, or dual-logistics routes may suffer zero physical downtime.",
        "<b>News Reporting Latency &amp; Media Bias:</b> Global news media naturally exhibits reporting latency and geographical bias, potentially under-reporting localized disruptions in remote or non-English speaking logistics corridors.",
        "<b>English-Language Focus:</b> The current local NLP pipeline is optimized for English-language news media; localized non-English bulletins require translation preprocessing.",
        "<b>Synthetic Demonstration Master Data:</b> Master supplier catalogs are synthetically generated demo records to prevent corporate NDA disclosure, requiring enterprise ERP mapping for production deployment."
    ]
    for lim in limits:
        story.append(Paragraph(f"• &nbsp;{lim}", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # BODY PAGE 42 (PHYSICAL 50): CHAPTER 7 REFERENCES
    # =========================================================================
    story.append(Paragraph("CHAPTER 7 REFERENCES", style_ch_title))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "For the <b>BDS-35 News-to-Risk Supply Chain Early Warning System</b> project, the following academic literature, "
        "official technical documentation, and development resources were utilized:",
        style_body
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Academic Papers &amp; Machine Learning Literature:</b>", style_body_bold))
    story.append(Spacer(1, 4))
    refs_academic = [
        "[1] W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive Representation Learning on Large Graphs,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2017.",
        "[2] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, \"Graph Attention Networks,\" in <i>International Conference on Learning Representations (ICLR)</i>, 2018.",
        "[3] M. Fey and J. E. Lenssen, \"Fast Graph Representation Learning with PyTorch Geometric,\" in <i>ICLR Workshop on Representation Learning on Graphs and Manifolds</i>, 2019.",
        "[4] A. Hagberg, P. Swart, and D. Chult, \"Exploring Network Structure, Dynamics, and Function using NetworkX,\" Los Alamos National Lab, 2008."
    ]
    for ra in refs_academic:
        story.append(Paragraph(ra, style_body))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Websites and Online Resources:</b>", style_body_bold))
    story.append(Spacer(1, 4))
    refs_web = [
        "[1] PyTorch Geometric Documentation<br/>Comprehensive guide for implementing graph convolutions, SAGEConv, GATConv, and HeteroData.<br/>Available at: https://pytorch-geometric.readthedocs.io/",
        "[2] FastAPI Documentation<br/>Official reference for high-performance asynchronous web APIs, Pydantic schemas, and OpenAPI.<br/>Available at: https://fastapi.tiangolo.com/",
        "[3] Streamlit Documentation<br/>Reference for interactive reactive analytical applications, caching, and state management.<br/>Available at: https://docs.streamlit.io/",
        "[4] NewsAPI Developer Documentation<br/>API reference for global news search, article metadata retrieval, and date filtering.<br/>Available at: https://newsapi.org/docs",
        "[5] spaCy Documentation<br/>Industrial natural language processing reference for rule matching and entity extraction.<br/>Available at: https://spacy.io/"
    ]
    for rw in refs_web:
        story.append(Paragraph(rw, style_body))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>AI Tools Used:</b>", style_body_bold))
    story.append(Spacer(1, 4))
    refs_ai = [
        "[1] Google Antigravity &amp; DeepMind Coding Assistants<br/>Used for end-to-end system integration, verification workflows, and automated test orchestration.",
        "[2] ChatGPT &amp; DeepSeek AI<br/>Utilized for generating insights, refining graph algorithms, and optimizing project documentation.<br/>Available at: https://chat.openai.com/ &amp; https://www.deepseek.com/"
    ]
    for rai in refs_ai:
        story.append(Paragraph(rai, style_body))
        story.append(Spacer(1, 2))

    # Build the PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"50-Page BDS-35 Project Report successfully generated at: {output_pdf}")

if __name__ == "__main__":
    out_file = "reports/BDS35_Final_Project_Report.pdf"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    build_bds35_50page_pdf(out_file)
