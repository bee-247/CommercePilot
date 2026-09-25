#!/usr/bin/env python3
"""Replace the SQL catalog and index the same Chinese products in Milvus."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "generated_chinese_products.jsonl"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.session import init_db, product_engine  # noqa: E402, I001
from services.embedding import EmbeddingService  # noqa: E402, I001
from services.vector_store import MilvusVectorStore  # noqa: E402, I001
from sqlalchemy import inspect, text  # noqa: E402, I001


REQUIRED_FIELDS = {
    "product_id",
    "name",
    "category",
    "subcategory",
    "brand",
    "price",
    "description",
    "tags",
    "stock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用 JSONL 替换 SQLite 商品及库存，并将同一批商品写入 Milvus。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 JSONL，默认：{DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先删除当前模型对应的商品向量集合；默认保留集合并自动跳过已写入商品。",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        choices=range(1, 9),
        metavar="1-8",
        help="并发调用 embedding 接口的请求数，默认：2。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="单次 embedding 请求的商品数；默认使用 EMBEDDING_BATCH_SIZE。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="限流或临时网络错误的最大重试次数，默认：5。",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help="指数退避的初始秒数，默认：2。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验 JSONL，不修改 SQLite 或 Milvus。",
    )
    parser.add_argument(
        "--database-only",
        action="store_true",
        help="只替换 SQLite 商品及库存，不生成向量。",
    )
    parser.add_argument(
        "--skip-database",
        action="store_true",
        help="不写 SQLite，只生成或续跑商品向量。",
    )
    args = parser.parse_args()
    if args.reset and args.keep_existing:
        parser.error("--reset 与旧参数 --keep-existing 不能同时使用")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.max_retries < 0:
        parser.error("--max-retries 不能小于 0")
    if args.retry_base_delay <= 0:
        parser.error("--retry-base-delay 必须大于 0")
    if args.database_only and args.skip_database:
        parser.error("--database-only 与 --skip-database 不能同时使用")
    if args.database_only and args.reset:
        parser.error("--database-only 不接受 --reset")
    return args


def iter_products(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                product = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(product, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象")

            missing = sorted(REQUIRED_FIELDS - product.keys())
            if missing:
                raise ValueError(f"第 {line_number} 行缺少字段：{', '.join(missing)}")
            if not str(product["product_id"]).strip():
                raise ValueError(f"第 {line_number} 行的 product_id 不能为空")
            if not isinstance(product["tags"], list):
                raise ValueError(f"第 {line_number} 行的 tags 必须是数组")
            for field in ("name", "category", "subcategory", "brand", "description"):
                if not str(product[field]).strip():
                    raise ValueError(f"第 {line_number} 行的 {field} 不能为空")
            if not isinstance(product["price"], (int, float)) or isinstance(
                product["price"], bool
            ):
                raise ValueError(f"第 {line_number} 行的 price 必须是数字")
            if product["price"] < 0:
                raise ValueError(f"第 {line_number} 行的 price 不能小于 0")
            if not isinstance(product["stock"], int) or isinstance(
                product["stock"], bool
            ):
                raise ValueError(f"第 {line_number} 行的 stock 必须是整数")
            if product["stock"] < 0:
                raise ValueError(f"第 {line_number} 行的 stock 不能小于 0")
            yield product


def product_text(product: dict[str, Any]) -> str:
    tags = "、".join(
        dict.fromkeys(str(tag).strip() for tag in product["tags"] if str(tag).strip())
    )
    return "\n".join(
        [
            f"商品名称：{str(product['name']).strip()}",
            f"品牌：{str(product['brand']).strip()}",
            "商品分类："
            f"{str(product['category']).strip()} > "
            f"{str(product['subcategory']).strip()}",
            f"商品价格：{float(product['price']):.2f} 元",
            f"商品描述：{str(product['description']).strip()}",
            f"商品标签：{tags}",
            f"库存状态：{'有货' if product['stock'] > 0 else '缺货'}",
        ]
    )


def batches(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def validate_unique_product_ids(products: list[dict[str, Any]]) -> None:
    counts = Counter(str(product["product_id"]).strip() for product in products)
    duplicates = sorted(product_id for product_id, count in counts.items() if count > 1)
    if duplicates:
        preview = "、".join(duplicates[:10])
        suffix = "……" if len(duplicates) > 10 else ""
        raise ValueError(f"输入文件存在重复 product_id：{preview}{suffix}")


def _sqlite_database_path() -> Path | None:
    if product_engine.dialect.name != "sqlite":
        return None
    database = product_engine.url.database
    return Path(database).resolve() if database and database != ":memory:" else None


def backup_sqlite_database() -> Path | None:
    """Create a consistent online backup before replacing a different catalog."""
    source_path = _sqlite_database_path()
    if source_path is None or not source_path.is_file():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = source_path.with_name(
        f"{source_path.stem}.before-chinese-products-{timestamp}{source_path.suffix}"
    )
    with sqlite3.connect(source_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path


def replace_sqlite_catalog(products: list[dict[str, Any]]) -> tuple[int, int]:
    """Atomically make products and inventory exactly match the JSONL input."""
    init_db()
    inspector = inspect(product_engine)
    product_columns = {
        column["name"] for column in inspector.get_columns("products")
    }
    if "subcategory" not in product_columns:
        with product_engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE products ADD COLUMN subcategory "
                    "VARCHAR(100) DEFAULT ''"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_products_subcategory "
                    "ON products (subcategory)"
                )
            )

    input_ids = {str(product["product_id"]).strip() for product in products}
    with product_engine.connect() as connection:
        current_ids = set(
            connection.execute(text("SELECT product_id FROM products")).scalars()
        )

    if current_ids != input_ids:
        backup_path = backup_sqlite_database()
        if backup_path is not None:
            print(f"SQLite 备份：{backup_path}", file=sys.stderr)

    now = datetime.now()
    product_rows = []
    inventory_rows = []
    for product in products:
        subcategory = str(product["subcategory"]).strip()
        searchable_tags = list(
            dict.fromkeys(
                [subcategory]
                + [str(tag).strip() for tag in product["tags"] if str(tag).strip()]
            )
        )
        product_id = str(product["product_id"]).strip()
        product_rows.append(
            {
                "product_id": product_id,
                "name": str(product["name"]).strip(),
                "category": str(product["category"]).strip(),
                "subcategory": subcategory,
                "price": float(product["price"]),
                "description": str(product["description"]).strip(),
                "brand": str(product["brand"]).strip(),
                "seller_id": "",
                "tags_json": json.dumps(searchable_tags, ensure_ascii=False),
                "image_url": "",
                "status": "active",
                "hot_score": 0.0,
                "created_at": now,
                "updated_at": now,
            }
        )
        inventory_rows.append(
            {
                "product_id": product_id,
                "stock": int(product["stock"]),
                "reserved_stock": 0,
                "updated_at": now,
            }
        )

    with product_engine.begin() as connection:
        connection.execute(text("DELETE FROM inventory"))
        connection.execute(text("DELETE FROM products"))
        connection.execute(
            text(
                "INSERT INTO products "
                "(product_id, name, category, subcategory, price, description, "
                "brand, seller_id, tags_json, image_url, status, hot_score, "
                "created_at, updated_at) VALUES "
                "(:product_id, :name, :category, :subcategory, :price, "
                ":description, :brand, :seller_id, :tags_json, :image_url, "
                ":status, :hot_score, :created_at, :updated_at)"
            ),
            product_rows,
        )
        connection.execute(
            text(
                "INSERT INTO inventory "
                "(product_id, stock, reserved_stock, updated_at) VALUES "
                "(:product_id, :stock, :reserved_stock, :updated_at)"
            ),
            inventory_rows,
        )

    with product_engine.connect() as connection:
        product_count = connection.execute(
            text("SELECT COUNT(*) FROM products")
        ).scalar_one()
        inventory_count = connection.execute(
            text("SELECT COUNT(*) FROM inventory")
        ).scalar_one()
        stored_ids = set(
            connection.execute(text("SELECT product_id FROM products")).scalars()
        )
    if product_count != len(products) or inventory_count != len(products):
        raise RuntimeError(
            "SQLite 写入后数量不一致："
            f"products={product_count}，inventory={inventory_count}"
        )
    if stored_ids != input_ids:
        raise RuntimeError("SQLite 写入后的 product_id 与 JSONL 不一致")
    return int(product_count or 0), int(inventory_count or 0)


def existing_product_ids(vector_store: MilvusVectorStore) -> set[str]:
    """Return durable IDs already present so an interrupted run can resume."""
    if not vector_store._connect():
        raise RuntimeError("无法连接 Milvus")

    from pymilvus import Collection, utility

    if not utility.has_collection(vector_store.product_collection):
        return set()

    collection = Collection(vector_store.product_collection)
    collection.load()
    iterator = collection.query_iterator(
        batch_size=1000,
        expr='product_id != ""',
        output_fields=["product_id"],
    )
    product_ids: set[str] = set()
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            product_ids.update(
                str(row["product_id"]).strip()
                for row in rows
                if row.get("product_id")
            )
    finally:
        iterator.close()
    return product_ids


def retryable_embedding_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).casefold()
    permanent_markers = (
        "freetieronly",
        "free quota exhausted",
        "insufficient_quota",
        "invalid api key",
        "authentication",
        "permission denied",
    )
    if any(marker in message for marker in permanent_markers):
        return False
    return status_code not in {400, 401, 403, 404, 422}


async def embed_with_retry(
    product_batch: list[dict[str, Any]],
    embedding_service: EmbeddingService,
    max_retries: int,
    retry_base_delay: float,
) -> list[list[float]]:
    texts = [product_text(product) for product in product_batch]
    for attempt in range(max_retries + 1):
        try:
            return await embedding_service.embed_documents_async(texts)
        except Exception as exc:
            if attempt >= max_retries or not retryable_embedding_error(exc):
                raise
            delay = retry_base_delay * (2**attempt) + random.uniform(0, 0.5)
            first_id = str(product_batch[0]["product_id"])
            print(
                f"批次 {first_id} 调用失败，{delay:.1f} 秒后进行第 "
                f"{attempt + 1}/{max_retries} 次重试：{exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def embed_and_store(
    products: list[dict[str, Any]],
    embedding_service: EmbeddingService,
    vector_store: MilvusVectorStore,
    concurrency: int,
    batch_size: int,
    max_retries: int,
    retry_base_delay: float,
    already_indexed: int,
    total_products: int,
) -> int:
    indexed = already_indexed
    next_milestone = (already_indexed * 100 // total_products // 10 + 1) * 10
    product_batches = list(batches(products, batch_size))
    for start in range(0, len(product_batches), concurrency):
        request_window = product_batches[start : start + concurrency]
        results = await asyncio.gather(
            *(
                embed_with_retry(
                    product_batch,
                    embedding_service,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                )
                for product_batch in request_window
            ),
            return_exceptions=True,
        )

        rows = []
        first_error: BaseException | None = None
        for product_batch, result in zip(request_window, results, strict=True):
            if isinstance(result, BaseException):
                first_error = first_error or result
                continue
            vectors = result
            if len(vectors) != len(product_batch):
                raise RuntimeError("Embedding 返回数量与当前商品批次不一致")
            rows.extend(
                (str(product["product_id"]).strip(), vector)
                for product, vector in zip(product_batch, vectors, strict=True)
            )
        if rows and not await vector_store.upsert_product_embeddings(rows):
            raise RuntimeError("商品向量写入 Milvus 失败")

        indexed += len(rows)
        progress = indexed * 100 // total_products
        while next_milestone <= min(progress, 100):
            print(
                f"Milvus 写入进度：{next_milestone}% "
                f"({indexed}/{total_products})",
                file=sys.stderr,
            )
            next_milestone += 10
        if first_error is not None:
            raise RuntimeError(
                "部分批次调用 embedding 接口失败；已成功的批次已经写入，"
                "修复接口问题后直接重新执行即可续跑。"
            ) from first_error
    return indexed


async def async_main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    products = list(iter_products(input_path))
    if not products:
        raise ValueError("输入文件没有商品记录")
    validate_unique_product_ids(products)

    print(f"JSONL 校验完成：{len(products)} 个商品。", file=sys.stderr)
    if args.dry_run:
        print("校验完成：未修改 SQLite 或 Milvus。", file=sys.stderr)
        return 0

    if not args.skip_database:
        product_count, inventory_count = replace_sqlite_catalog(products)
        print(
            f"SQLite 写入完成：products={product_count}，"
            f"inventory={inventory_count}。",
            file=sys.stderr,
        )
    if args.database_only:
        return 0

    embedding_service = EmbeddingService()
    if not (
        embedding_service.api_key
        and embedding_service.base_url
        and embedding_service.embedding_model
    ):
        raise ValueError(
            "请先在 .env 配置 EMBEDDING_MODEL、EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL"
        )
    vector_store = MilvusVectorStore()
    print(
        f"商品数：{len(products)}；模型：{embedding_service.embedding_model}；"
        f"维度：{embedding_service.embedding_dimension}；"
        f"Milvus 集合：{vector_store.product_collection}",
        file=sys.stderr,
    )
    batch_size = args.batch_size or embedding_service.batch_size
    print(
        f"请求批量：{batch_size}；并发：{args.concurrency}；"
        f"最大重试：{args.max_retries}",
        file=sys.stderr,
    )

    if args.reset:
        print(
            f"正在删除旧商品向量集合：{vector_store.product_collection}",
            file=sys.stderr,
        )
        if not vector_store.drop_collection(vector_store.product_collection):
            raise RuntimeError("无法连接 Milvus 或删除旧商品向量集合失败")

    input_ids = {str(product["product_id"]).strip() for product in products}
    stored_ids = existing_product_ids(vector_store)
    completed_ids = input_ids & stored_ids
    pending_products = [
        product
        for product in products
        if str(product["product_id"]).strip() not in completed_ids
    ]
    print(
        f"断点状态：已完成 {len(completed_ids)}，待处理 {len(pending_products)}。",
        file=sys.stderr,
    )
    if not pending_products:
        print("完成：输入文件中的商品向量均已存在。", file=sys.stderr)
        return 0

    indexed = await embed_and_store(
        pending_products,
        embedding_service,
        vector_store,
        concurrency=args.concurrency,
        batch_size=batch_size,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        already_indexed=len(completed_ids),
        total_products=len(products),
    )
    print(
        f"完成：已将 {indexed} 个商品写入 {vector_store.product_collection}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
