import os
import requests
from typing import List


CAC40_REPORTS = [
    {
        "company": "TotalEnergies",
        "url": "https://www.totalenergies.com/sites/g/files/nytnzq121/files/documents/2024-03/totalenergies_rapport-annuel_2023.pdf",
        "filename": "totalenergies_2023.pdf"
    },
    {
        "company": "BNP Paribas",
        "url": "https://invest.bnpparibas/sites/default/files/documents/bnp_paribas_rapport_annuel_2023.pdf",
        "filename": "bnpparibas_2023.pdf"
    },
    {
        "company": "Airbus",
        "url": "https://www.airbus.com/sites/g/files/jlcbta16/files/2024-02/Airbus-Annual-Report-2023.pdf",
        "filename": "airbus_2023.pdf"
    }
]

OUTPUT_FOLDER = "data/raw/pdf"


def download_pdf(company: str, url: str, filename: str) -> bool:
    """
    Download a single PDF report.
    Returns True if successful, False otherwise.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    if os.path.exists(output_path):
        print(f"  Already exists: {filename}")
        return True

    print(f"Downloading: {company}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  → Saved: {filename} ({size_kb:.1f} KB)")
        return True

    except Exception as e:
        print(f"  ERROR {company}: {e}")
        return False


def download_all_reports() -> List[str]:
    """
    Download all CAC40 reports.
    Returns list of successfully downloaded filenames.
    """
    print("Downloading CAC40 annual reports...")
    downloaded = []

    for report in CAC40_REPORTS:
        success = download_pdf(
            report["company"],
            report["url"],
            report["filename"]
        )
        if success:
            downloaded.append(report["filename"])

    print(f"\nTotal downloaded: {len(downloaded)}/{len(CAC40_REPORTS)}")
    return downloaded


if __name__ == "__main__":
    files = download_all_reports()
    print(f"\nFiles ready: {files}")