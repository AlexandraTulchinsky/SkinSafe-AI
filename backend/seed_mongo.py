# Populate a local MongoDB with eczema trigger data from a
# JSON file structured as:
# {
#   "Top Category": {
#     "Subcategory": [
#       "Product (derivative1, derivative2, ...)",
#       ...
#     ],
#     ...
#   },
#   "Another Top Category": [ "Product (...)", ... ]
# }
#
# One MongoDB document per *product* with fields:
#   - ingredient     : canonical product (lowercase)
#   - examples       : list[str] of derivative names (lowercase)
#   - triggers       : list[str] (e.g., "histamine", "gluten")
#   - groups         : list[str] (top-level categories, e.g., "biogenic amines")
#   - paths          : list[str] ("biogenic amines > histamine")
#   - last_updated   : datetime
#
# Usage (Windows PowerShell):
#   pip install pymongo
#   python seed_mongo.py --json C:\path\to\cross_ref_food.json --db eczema_db --collection ingredients --drop

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple, Any

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# Helper cleaning 

DELIMS = r"[,;/]|(?:\s+\band\b\s+)"

def _normalize_spaces(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_entry(raw: str) -> Tuple[str, List[str]]:
    """
    Input:  "Aged cheese (Cheddar, Parmesan, Blue cheese, Brie)"
    Output: ("aged cheese", ["cheddar","parmesan","blue cheese","brie"])
    If no parentheses → examples = []
    """
    s = _normalize_spaces(raw)
    m = re.match(r"^(.*?)\s*\((.*?)\)$", s)
    if m:
        base = _normalize_spaces(m.group(1))
        inside = _normalize_spaces(m.group(2))
        parts = [p.strip() for p in re.split(DELIMS, inside) if p.strip()]
        examples = [p.lower() for p in parts]
    else:
        base = s
        examples = []
    return base.lower(), examples

# JSON traversal

def walk_json(data: Dict[str, Any]) -> Iterable[Tuple[str, List[str], str, str, str]]:
    """
    Yields tuples: (ingredient_base, examples, trigger, group, path)
    - group  : top-level key (e.g., "biogenic amines")
    - trigger: subcategory if present (e.g., "histamine"), else same as group
    - path   : "group > trigger" or just "group" when no subcategory
    """
    for group, val in data.items():
        group_l = _normalize_spaces(group).lower()
        if isinstance(val, dict):
            for subcat, items in val.items():
                trigger_l = _normalize_spaces(subcat).lower()
                for item in items:
                    base, examples = parse_entry(item)
                    yield base, examples, trigger_l, group_l, f"{group_l} > {trigger_l}"
        elif isinstance(val, list):
            for item in val:
                base, examples = parse_entry(item)
                yield base, examples, group_l, group_l, group_l

# Mongo load 

def seed_mongo(
    records: Iterable[Tuple[str, List[str], str, str, str]],
    uri: str,
    db_name: str,
    coll_name: str,
    drop: bool = False,
    sample: int = 0
) -> None:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    col = db[coll_name]

    if drop:
        col.drop()

    # Indexes for fast lookups
    col.create_index([("ingredient", 1)], unique=True)
    col.create_index([("examples", 1)])
    col.create_index([("triggers", 1)])
    col.create_index([("groups", 1)])

    # Merge documents per ingredient
    merged: Dict[str, Dict[str, Any]] = {}
    for base, examples, trigger, group, path in records:
        doc = merged.setdefault(base, {
            "ingredient": base,
            "examples": set(),
            "triggers": set(),
            "groups": set(),
            "paths": set(),
        })
        if examples:
            doc["examples"].update(examples)
        doc["triggers"].add(trigger)
        doc["groups"].add(group)
        doc["paths"].add(path)

    # Convert sets -> lists; add last_updated
    now = datetime.now(timezone.utc)
    for d in merged.values():
        d["examples"] = sorted(d["examples"])
        d["triggers"] = sorted(d["triggers"])
        d["groups"] = sorted(d["groups"])
        d["paths"] = sorted(d["paths"])
        d["last_updated"] = now

    # Bulk upsert
    ops: List[UpdateOne] = []
    for base, d in merged.items():
        ops.append(
            UpdateOne(
                {"ingredient": base},
                {"$set": d},
                upsert=True,
            )
        )

    if ops:
        try:
            col.bulk_write(ops, ordered=False)
        except BulkWriteError as e:
            print("BulkWriteError:", e.details)

    total = col.count_documents({})
    print(f"✅ Done. Collection: {db_name}.{coll_name} | Documents: {total}")

    if sample > 0:
        print("\n— Samples —")
        for doc in col.find({}, {"_id": 0}).limit(sample):
            print(json.dumps(doc, default=str, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Seed local MongoDB from cross-ref food JSON.")
    ap.add_argument("--json", required=True, help="Path to cross_ref_food.json")
    ap.add_argument("--db", default="eczema_db", help="MongoDB database name")
    ap.add_argument("--collection", default="ingredients", help="MongoDB collection name")
    ap.add_argument("--uri", default="mongodb://localhost:27017/", help="MongoDB URI")
    ap.add_argument("--drop", action="store_true", help="Drop collection before insert")
    ap.add_argument("--sample", type=int, default=0, help="Print N sample docs after load")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = list(walk_json(data))
    seed_mongo(
        records=records,
        uri=args.uri,
        db_name=args.db,
        coll_name=args.collection,
        drop=args.drop,
        sample=args.sample,
    )

if __name__ == "__main__":
    main()
