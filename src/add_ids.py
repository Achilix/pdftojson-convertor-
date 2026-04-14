import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_records(input_path: Path) -> List[Dict[str, Any]]:
	with input_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError(f"Expected a JSON array in {input_path}")

	records: List[Dict[str, Any]] = []
	for item in payload:
		if not isinstance(item, dict):
			raise ValueError(f"Expected each JSON item to be an object in {input_path}")
		records.append(dict(item))

	return records


def _write_json(output_path: Path, records: List[Dict[str, Any]]) -> None:
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(records, handle, ensure_ascii=False, indent=2)


def _write_csv(output_path: Path, records: List[Dict[str, Any]]) -> None:
	df = pd.DataFrame(records)
	df.to_csv(output_path, index=False, encoding="utf-8-sig")


def add_ids(records: List[Dict[str, Any]], id_field: str = "id") -> List[Dict[str, Any]]:
	updated: List[Dict[str, Any]] = []
	for index, record in enumerate(records, start=1):
		row = dict(record)
		row[id_field] = index
		updated.append(row)
	return updated


def process_file(input_path: Path, output_dir: Path, id_field: str = "id") -> None:
	records = _load_records(input_path)
	updated_records = add_ids(records, id_field=id_field)

	output_dir.mkdir(parents=True, exist_ok=True)
	stem = input_path.stem
	json_output = output_dir / f"{stem}_with_ids.json"
	csv_output = output_dir / f"{stem}_with_ids.csv"

	_write_json(json_output, updated_records)
	_write_csv(csv_output, updated_records)

	print(f"Wrote {json_output}")
	print(f"Wrote {csv_output}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Add sequential ids to extracted document records.")
	parser.add_argument(
		"input",
		help="Path to a JSON file containing an array of document records, or a directory of *_articles.json files.",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory to write output files. Defaults to output/with_ids for folder inputs or the input file directory for file inputs.",
	)
	parser.add_argument(
		"--id-field",
		default="id",
		help="Name of the id field to add to each record.",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	if input_path.is_dir():
		output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "with_ids"
		for json_path in sorted(input_path.glob("*_articles.json")):
			process_file(json_path, output_dir, id_field=args.id_field)
		return

	if not input_path.exists():
		raise FileNotFoundError(f"Input path not found: {input_path}")

	output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
	process_file(input_path, output_dir, id_field=args.id_field)


if __name__ == "__main__":
	main()
