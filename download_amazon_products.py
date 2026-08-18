#!/usr/bin/env python3
"""Stream and normalize Amazon Reviews 2023 item metadata.

The script downloads product metadata only. It does not download reviews or
user interactions. Output is newline-delimited JSON suitable for a later
database-import or embedding-indexing step.

Example:
    1. Edit PRODUCTS_PER_CATEGORY below to set the shared category quantity.
    2. Run: python3 download_amazon_products.py
"""

from __future__ import annotations

import gzip
import io
import json
import random
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DATASET_BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories"
)

# ---------------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------------
# All fixed Chinese business categories use this same target quantity.
PRODUCTS_PER_CATEGORY = 200                         #每个类别的个数

OUTPUT_PATH = Path("data/amazon_products.jsonl")
SHUFFLE_BUFFER = 1_000
RANDOM_SEED = 42
REQUIRE_IMAGE = False
MIN_RATING_COUNT = 0

# Amazon source category -> CommercePilot display category.
CATEGORY_MAP: dict[str, str] = {
    # Digital products and appliances
    "Electronics": "数码家电",
    "Appliances": "数码家电",
    "Cell_Phones_and_Accessories": "数码家电",
    "Software": "数码家电",
    # Beauty and personal care
    "All_Beauty": "美妆个护",
    "Beauty_and_Personal_Care": "美妆个护",
    # Home
    "Home_and_Kitchen": "家居用品",
    # Food
    "Grocery_and_Gourmet_Food": "食品饮料",
    "Subscription_Boxes": "食品饮料",
    # Sports
    "Sports_and_Outdoors": "运动户外",
    # Books and entertainment
    "Books": "图书音像",
    "Kindle_Store": "图书音像",
    "Magazine_Subscriptions": "图书音像",
    "Movies_and_TV": "图书音像",
    "CDs_and_Vinyl": "图书音像",
    "Digital_Music": "图书音像",
    # Health
    "Health_and_Household": "医药健康",
    "Health_and_Personal_Care": "医药健康",
    # Automotive
    "Automotive": "汽车用品",
    # Fashion
    "Amazon_Fashion": "服饰鞋包",
    "Clothing_Shoes_and_Jewelry": "服饰鞋包",
    # Baby
    "Baby_Products": "母婴用品",
    # Office
    "Office_Products": "办公用品",
    # Home improvement and garden
    "Tools_and_Home_Improvement": "家装工具",
    "Patio_Lawn_and_Garden": "家装工具",
    # Pets
    "Pet_Supplies": "宠物用品",
    # Toys and games
    "Toys_and_Games": "玩具游戏",
    "Video_Games": "玩具游戏",
    # Arts and handmade products
    "Arts_Crafts_and_Sewing": "艺术手工",
    "Handmade_Products": "艺术手工",
    # Musical instruments
    "Musical_Instruments": "乐器",
    # Industrial products
    "Industrial_and_Scientific": "工业科研",
}


def parse_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None

    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    parsed = float(match.group())
    return parsed if parsed > 0 else None


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def first_image(images: Any) -> str:
    if not isinstance(images, dict):
        return ""
    for image_type in ("hi_res", "large", "thumb"):
        urls = images.get(image_type)
        if not isinstance(urls, list):
            continue
        for url in urls:
            if url and str(url).startswith(("http://", "https://")):
                return str(url)
    return ""


def clean_details(value: Any) -> dict[str, Any] | str:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else value.strip()
        except json.JSONDecodeError:
            return value.strip()
    return {}


def normalize_product(
    row: dict[str, Any],
    source_category: str,
    target_category: str,
    *,
    require_image: bool,
    min_rating_count: int,
) -> dict[str, Any] | None:
    product_id = str(row.get("parent_asin") or "").strip()
    title = str(row.get("title") or "").strip()
    price = parse_price(row.get("price"))
    description_parts = string_list(row.get("description"))
    features = string_list(row.get("features"))
    image_url = first_image(row.get("images"))
    rating_count = parse_int(row.get("rating_number"))

    if not product_id or not title or price is None:
        return None
    if not description_parts and not features:
        return None
    if require_image and not image_url:
        return None
    if rating_count < min_rating_count:
        return None

    description = " ".join(description_parts).strip()
    if not description:
        description = " ".join(features).strip()

    return {
        "product_id": product_id,
        "name": title,
        "category": target_category,
        "source_category": source_category,
        "price": price,
        "brand": str(row.get("store") or "未知品牌").strip(),
        "description": description,
        "tags": features[:8],
        "image_url": image_url,
        "average_rating": parse_float(row.get("average_rating")),
        "rating_number": rating_count,
        "details": clean_details(row.get("details")),
        "source": "Amazon Reviews 2023",
    }


def build_download_plan() -> list[tuple[str, int, list[str]]]:
    if (
        isinstance(PRODUCTS_PER_CATEGORY, bool)
        or not isinstance(PRODUCTS_PER_CATEGORY, int)
        or PRODUCTS_PER_CATEGORY <= 0
    ):
        raise ValueError("PRODUCTS_PER_CATEGORY must be a positive integer")

    # dict preserves insertion order and removes repeated mapped categories.
    fixed_categories = list(dict.fromkeys(CATEGORY_MAP.values()))

    return [
        (
            target_category,
            PRODUCTS_PER_CATEGORY,
            [
                source
                for source, target in CATEGORY_MAP.items()
                if target == target_category
            ],
        )
        for target_category in fixed_categories
    ]


def iter_streaming_dataset(
    source_category: str,
    *,
    seed: int,
    shuffle_buffer: int,
) -> Iterable[dict[str, Any]]:
    """Read an official compressed JSONL file without Hugging Face scripts.

    Recent releases of ``datasets`` no longer execute legacy dataset loading
    scripts. Amazon Reviews 2023 still uses such a script on Hugging Face, so
    reading the official category file directly is both simpler and compatible
    with current Python environments.
    """
    url = f"{DATASET_BASE_URL}/meta_{source_category}.jsonl.gz"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CommercePilot-dataset-downloader/1.0"},
    )

    try:
        response = urllib.request.urlopen(
            request,
            timeout=60,
            context=build_ssl_context(),
        )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "TLS certificate verification failed. Install or update the "
                "Python CA bundle with: python3 -m pip install -U certifi"
            ) from exc
        raise RuntimeError(f"Unable to download {url}: {exc.reason}") from exc

    def rows() -> Iterable[dict[str, Any]]:
        with response:
            with gzip.GzipFile(fileobj=response) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                    for line_number, line in enumerate(text, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            print(
                                f"Skipping invalid JSON in {source_category} "
                                f"at line {line_number}",
                                file=sys.stderr,
                            )
                            continue
                        if isinstance(row, dict):
                            yield row

    return buffered_shuffle(rows(), seed=seed, buffer_size=shuffle_buffer)


def build_ssl_context() -> ssl.SSLContext:
    """Build a verified TLS context, preferring certifi's portable CA bundle."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def buffered_shuffle(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    buffer_size: int,
) -> Iterable[dict[str, Any]]:
    """Shuffle a stream approximately while keeping memory bounded."""
    rng = random.Random(seed)
    buffer: list[dict[str, Any]] = []

    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row

    rng.shuffle(buffer)
    yield from buffer


def main() -> int:
    try:
        categories = build_download_plan()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen_product_ids: set[str] = set()
    category_counts: dict[str, int] = {}
    failed_categories: list[str] = []

    # A dedicated RNG yields a stable, distinct shuffle seed for each source.
    rng = random.Random(RANDOM_SEED)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for target_category, requested_count, source_categories in categories:
            category_retained = 0
            if requested_count == 0:
                category_counts[target_category] = 0
                print(f"Skipping {target_category}: target is 0")
                continue

            print(
                f"Downloading fixed category {target_category} "
                f"(target: {requested_count}) ..."
            )
            for source_index, source_category in enumerate(source_categories):
                remaining = requested_count - category_retained
                if remaining <= 0:
                    break

                sources_left = len(source_categories) - source_index
                source_target = (remaining + sources_left - 1) // sources_left
                source_retained = 0
                scanned = 0
                print(
                    f"  Source {source_category} "
                    f"(target contribution: {source_target}) ..."
                )
                try:
                    dataset = iter_streaming_dataset(
                        source_category,
                        seed=rng.randrange(0, 2**31),
                        shuffle_buffer=SHUFFLE_BUFFER,
                    )
                    for row in dataset:
                        scanned += 1
                        product = normalize_product(
                            row,
                            source_category,
                            target_category,
                            require_image=REQUIRE_IMAGE,
                            min_rating_count=MIN_RATING_COUNT,
                        )
                        if not product:
                            continue
                        product_id = product["product_id"]
                        if product_id in seen_product_ids:
                            continue

                        output.write(json.dumps(product, ensure_ascii=False) + "\n")
                        seen_product_ids.add(product_id)
                        source_retained += 1
                        category_retained += 1
                        if (
                            source_retained >= source_target
                            or category_retained >= requested_count
                        ):
                            break
                except Exception as exc:  # Keep other sources usable on one failure.
                    failed_categories.append(source_category)
                    print(f"Failed {source_category}: {exc}", file=sys.stderr)

                print(
                    f"  Finished {source_category}: "
                    f"kept {source_retained}, scanned {scanned}"
                )

            category_counts[target_category] = category_retained
            print(
                f"Finished fixed category {target_category}: "
                f"kept {category_retained}/{requested_count}"
            )

    print(f"\nSaved {len(seen_product_ids)} unique products to {OUTPUT_PATH}")
    for source_category, retained in category_counts.items():
        print(f"  {source_category}: {retained}")
    if failed_categories:
        print(
            "Failed categories: " + ", ".join(failed_categories),
            file=sys.stderr,
        )
    return 0 if seen_product_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
