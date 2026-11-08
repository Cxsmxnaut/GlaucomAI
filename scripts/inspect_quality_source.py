"""Inspect a pinned public quality dataset in Colab without reading test labels."""

import argparse
from hashlib import md5, sha256
from io import BytesIO
import json
from pathlib import Path
import urllib.request
import zipfile

import openpyxl

URL = "https://ndownloader.figshare.com/files/38115186"
EXPECTED_BYTES = 1285292595
EXPECTED_MD5 = "c4f7bbb7979ae918691c35e543900f03"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("data/mshf_v1"))
    parser.add_argument("--report-name", default="inspection.json")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if Path(args.report_name).name != args.report_name or not args.report_name.endswith(".json"):
        raise ValueError("Use a JSON filename inside the inspection folder.")
    report_path = args.work / args.report_name
    if report_path.exists():
        raise ValueError("Inspection report exists. Choose a new --report-name to preserve evidence.")
    destination = args.work / "MSHF_dataset.zip"
    if not destination.exists():
        temporary = destination.with_suffix(".partial")
        received = 0
        with urllib.request.urlopen(URL, timeout=90) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received % (64 * 1024 * 1024) == 0:
                    print(f"Quality-source download: {received / 1024**2:.0f} MiB", flush=True)
        if received != EXPECTED_BYTES:
            raise ValueError("Download size differs from the pinned Figshare file.")
        temporary.rename(destination)
    legacy, strong = md5(), sha256()
    with destination.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            legacy.update(chunk)
            strong.update(chunk)
    if destination.stat().st_size != EXPECTED_BYTES or legacy.hexdigest() != EXPECTED_MD5:
        raise ValueError("Dataset archive differs from the publisher's pinned file.")
    report = {"source": "https://figshare.com/articles/figure/21507564/1", "version": 1,
              "license": "CC BY 4.0", "bytes": EXPECTED_BYTES,
              "publisher_md5": EXPECTED_MD5, "sha256": strong.hexdigest(),
              "inspection_scope": "Archive paths and spreadsheet headers only; no test labels read and no model trained.",
              "spreadsheets": {}, "directory_image_counts": {}, "non_image_files": [],
              "dimension_warnings": {}}
    counts = {}
    with zipfile.ZipFile(destination) as bundle:
        for name in bundle.namelist():
            path = Path(name)
            if "__MACOSX" in path.parts or path.name.startswith("."):
                continue
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                parent = path.parent.as_posix()
                counts[parent] = counts.get(parent, 0) + 1
            elif not name.endswith("/"):
                report["non_image_files"].append(name)
            if path.suffix.lower() == ".xlsx":
                workbook = openpyxl.load_workbook(BytesIO(bundle.read(name)), read_only=True, data_only=True)
                try:
                    report["spreadsheets"][name] = {sheet.title: {
                        "rows": sheet.max_row, "columns": sheet.max_column,
                        "header": [str(value) if value is not None else None
                                   for value in next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))]
                    } for sheet in workbook.worksheets}
                    for sheet in workbook.worksheets:
                        if sheet.max_row and sheet.max_row > 10000:
                            report["dimension_warnings"][sheet.title] = "Declared row dimension exceeds source image counts. Do not materialize sheet.values; use bounded, verified image-name joins."
                finally:
                    workbook.close()
    report["directory_image_counts"] = counts
    report_path.write_text(json.dumps(report, indent=2))
    print("QUALITY SOURCE INSPECTION COMPLETE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
