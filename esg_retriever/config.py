"""Module containing path and file configurations for the ESG report data."""

from pathlib import Path


# -------------------------- Paths  --------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Root directory of the repository."""

DATA_DIR = REPO_ROOT / "data"
"""Root directory of the data."""

PDF_REPORTS_DIR = DATA_DIR / "reports"
"""Directory containing the company annual report pdf files."""

MISC_DATA_DIR = DATA_DIR / "misc"
"""Directory containing miscellaneous data files."""

JSON_REPORTS_DIR = REPO_ROOT / "azure_ai_doc_intelligence" / "outputs"
"""Directory containing the json outputs from Azure AI Document Intelligence."""


# -------------------------- ESG Report File Configuration  --------------------------

COMPANY_YEAR_PDF_MAPPING = {
    "Absa": {
        2022: [
            "2022-Absa-Group-limited-Environmental-Social-and-Governance-Data-sheet.pdf"
        ],
        2021: [
            "2022-Absa-Group-limited-Environmental-Social-and-Governance-Data-sheet.pdf"
        ],
    },
    "Clicks": {
        2022: ["Clicks-Sustainability-Report-2022.pdf"],
    },
    "Distell": {
        2022: ["DISTELL ESG Appendix 2022.pdf"],
        2021: ["DISTELL ESG Appendix 2022.pdf"],
    },
    "Impala": {
        2023: ["ESG-spreads.pdf"],
        2022: ["ESG-spreads.pdf"],
    },
    "Oceana": {
        2022: [
            "Oceana_ESG_Databook_FY2022.pdf",
            "Oceana_Group_Sustainability_Report_2022.pdf",
        ],
    },
    "Picknpay": {
        2023: ["picknpay-esg-report-spreads-2023.pdf"],
        2022: ["picknpay-esg-report-spreads-2023.pdf"],
        2021: ["picknpay-esg-report-spreads-2023.pdf"],
    },
    "Sasol": {
        2023: ["SASOL Sustainability Report 2023 20-09_0.pdf"],
        2022: ["SASOL Sustainability Report 2023 20-09_0.pdf"],
        2021: ["SASOL Sustainability Report 2023 20-09_0.pdf"],
    },
    "Ssw": {
        2022: ["ssw-IR22.pdf"],
        2021: ["ssw-IR22.pdf"],
    },
    "Tongaat": {
        2021: ["2021ESG.pdf"],
        2020: ["2021ESG.pdf"],
    },
    "Uct": {
        2021: ["UCT_Carbon_Footprint_Report_2020-2021.pdf", "afs2021.pdf"],
    },
}
"""
Mapping from company and year to the corresponding annual report pdf file name(s).

The annual report corresponding to a company and year contains the environmental,
social, and governance (ESG) data for that year.
"""

COMPANIES = list(COMPANY_YEAR_PDF_MAPPING.keys())
"""List of companies with annual report pdf files."""
