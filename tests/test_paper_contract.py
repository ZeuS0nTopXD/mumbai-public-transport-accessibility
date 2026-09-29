from pathlib import Path
from zipfile import ZipFile
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT.parents[1] / "outputs"
DOCX_PATH = OUTPUTS / "mumbai_public_transport_accessibility_research_paper.docx"
PDF_PATH = OUTPUTS / "mumbai_public_transport_accessibility_research_paper.pdf"


class ResearchPaperContractTests(unittest.TestCase):
    def test_research_paper_outputs_contain_measured_metrics(self):
        self.assertTrue(DOCX_PATH.is_file())
        self.assertGreater(DOCX_PATH.stat().st_size, 0)
        self.assertTrue(PDF_PATH.is_file())
        self.assertGreater(PDF_PATH.stat().st_size, 0)

        with ZipFile(DOCX_PATH) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")

        for required_text in (
            "Uniform Cost Search",
            "A* Search",
            "Success Rate",
            "Route Optimality",
            "Search Efficiency",
            "Compute-Time Efficiency",
            "Compute Time",
            "MAE",
            "MSE",
            "RMSE",
            "MAPE",
            "R-squared",
            "300",
            "Evaluation Rank",
            "lowest route-cost RMSE",
            "Winner = arg min (route-cost RMSE, nodes checked, compute time)",
            "fewer nodes checked wins",
        ):
            self.assertIn(required_text, document_xml)

        self.assertNotIn("Final Score", document_xml)
        self.assertNotIn("50/50 efficiency score", document_xml)
        self.assertNotIn("Runtime Efficiency", document_xml)


if __name__ == "__main__":
    unittest.main()
