#!/usr/bin/env python3
"""按类目规划调用 DeepSeek，增量生成中文商品 JSONL。

默认行为：

1. 读取 ``chinese_product_taxonomy.json`` 和 ``product_generation_plan.json``；
2. 按规划顺序选择尚未完成的 1 至 2 个小类，并逐个处理；
3. 一个小类达到目标数量后才处理下一个，每个 API 请求只生成一个小类；
4. 输出恰好包含 product_id、name、category、subcategory、brand、price、
   description、tags、stock 九个字段。

示例：

    # 查看本次将处理的小类，不调用 API
    python data/generate_chinese_products.py --dry-run

    # 自动处理接下来的两个未完成小类
    python data/generate_chinese_products.py

    # 按规划顺序处理所有未完成小类
    python data/generate_chinese_products.py --all

    # 指定处理一个或两个小类
    python data/generate_chinese_products.py \
        --category 数码家电 --subcategory 耳机 --subcategory 音响
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_TAXONOMY = Path("data/chinese_product_taxonomy.json")
DEFAULT_PLAN = Path("data/product_generation_plan.json")
MAX_ATTEMPTS = 5
CHINESE_CHARACTER = re.compile(r"[\u4e00-\u9fff]")

OUTPUT_FIELDS = (
    "product_id",
    "name",
    "category",
    "subcategory",
    "brand",
    "price",
    "description",
    "tags",
    "stock",
)

CATEGORY_PREFIXES = {
    "数码家电": "DIGITAL",
    "服饰鞋包": "FASHION",
    "美妆个护": "BEAUTY",
    "食品生鲜": "FOOD",
    "家居家装": "HOME",
    "母婴用品": "BABY",
    "图书文娱": "CULTURE",
    "运动户外": "SPORT",
    "医药健康": "HEALTH",
    "宠物用品": "PET",
    "汽车用品": "AUTO",
}

# 这是兜底校验范围，不是要求模型均匀随机取值的范围。提示词会要求价格与
# 商品类型、规格和定位一致，避免低价配件出现整机价格。
CATEGORY_PRICE_LIMITS = {
    "数码家电": (5.0, 50000.0),
    "服饰鞋包": (5.0, 20000.0),
    "美妆个护": (3.0, 5000.0),
    "食品生鲜": (1.0, 5000.0),
    "家居家装": (2.0, 50000.0),
    "母婴用品": (3.0, 10000.0),
    "图书文娱": (1.0, 30000.0),
    "运动户外": (3.0, 30000.0),
    "医药健康": (1.0, 10000.0),
    "宠物用品": (2.0, 10000.0),
    "汽车用品": (3.0, 50000.0),
}

CATEGORY_FOCUS = {
    "数码家电": "写清核心参数、兼容性、接口、尺寸或容量，以及包装内容。",
    "服饰鞋包": "写清材质、版型、尺码建议、适用季节和洗护方式。",
    "美妆个护": "写清容量、肤质或发质、使用方式、成分特点及注意事项。",
    "食品生鲜": "写清净含量、口味或产地类型、储存方式、食用建议和过敏原。",
    "家居家装": "写清材质、尺寸、承重或适配范围、安装方式和养护方法。",
    "母婴用品": "写清适用年龄或阶段、材质、规格、使用方法和安全提醒。",
    "图书文娱": "写清内容或玩法、规格、适用年龄、配件和使用场景。",
    "运动户外": "写清材质、尺寸、承重或防护等级、适用场景和安全提醒。",
    "医药健康": "写清规格、适用范围、用法或测量方式，并避免诊疗和功效承诺。",
    "宠物用品": "写清适用宠物、规格、材质或配方、使用方法和注意事项。",
    "汽车用品": "写清适用车型或接口、尺寸、材质、安装方式和安全提醒。",
}

# 容易被模型混淆的小类需要补充“商品本体”边界。通用边界仍从类目 JSON
# 读取；这里同时提供可程序校验的关键词，避免把配件写成整机。
SUBCATEGORY_BOUNDARIES: dict[str, dict[str, Any]] = {
    "电脑": {
        "definition": (
            "商品本体必须是笔记本电脑、台式电脑整机、一体机、迷你电脑或工作站；"
            "键盘、鼠标、显示器、充电器、扩展坞和电脑零部件均不属于本小类。"
        ),
        "required_name_terms": (
            "电脑",
            "笔记本",
            "台式机",
            "一体机",
            "工作站",
        ),
        "forbidden_name_terms": (
            "键盘",
            "鼠标",
            "显示器",
            "充电器",
            "数据线",
            "扩展坞",
            "散热器",
            "鼠标垫",
            "硬盘",
            "内存条",
            "显卡",
        ),
    },
    "电脑外设": {
        "definition": (
            "商品本体应是键盘、鼠标、显示器、摄像头、扩展坞等电脑外设，"
            "不得生成笔记本、台式整机或一体机。"
        )
    },
    "数码配件": {
        "definition": (
            "商品本体应是充电器、数据线、保护壳、支架等通用数码配件，"
            "不得生成电脑整机或家用电器。"
        )
    },
    "游戏主机及配件": {
        "definition": "只生成游戏主机、游戏手柄、主机底座等游戏设备及专用配件。"
    },
    "电子阅读与学习设备": {
        "definition": "只生成电子书阅读器、学习机、点读笔等电子学习设备。"
    },
}

PROHIBITED_TERMS = (
    "国家级",
    "世界第一",
    "绝对有效",
    "百分百治愈",
    "包治",
    "根治",
)

SYSTEM_PROMPT = """你是中文电商演示商品数据生成器。只输出一个合法 JSON 对象，不要使用 Markdown 代码块，不要输出解释。

所有商品均为虚构演示数据。不得冒充真实品牌、真实在售商品、官方认证或获奖商品，不得使用知名品牌、机构、人物、赛事和作品商标。

商品要求：
1. name 是自然的中文商品标题，包含虚构品牌、商品类型和关键区别信息，不堆砌关键词。
2. brand 是简短、自然且虚构的中文品牌名。
3. price 是符合中国市场常识的人民币数字，必须结合具体商品类型、规格和定位判断，不能仅按大类随机定价。
4. description 为 150 至 400 个中文字符，详细说明用途、材质或成分、主要规格、适用场景、包装内容和必要限制；内容必须与标题、价格和标签一致。
5. tags 合并搜索标签和 selling_points。标签数量随商品信息变化，只保留真实有依据、有检索价值且不重复的中文短标签，不凑固定数量。
6. stock 是 0 至 100 的整数。
7. 同一批商品在功能、规格、价格定位、风格和适用人群上应有明显差异。
8. 不写物流、促销、销量、用户评价、绝对化功效或疾病诊断和治疗承诺。
9. 不生成武器、烟草、成人、赌博、危险化学品、处方药等不适合普通电商演示的商品。
10. 用户提供的已有商品名称均为禁止名称，name 不得与其相同，也不得只修改空格或标点后复用。

返回格式：
{"products":[{"sequence":1,"name":"商品标题","brand":"虚构品牌","price":99.00,"description":"详细中文说明","tags":["标签"],"stock":100}]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="只在指定大类中选择；可重复传入。",
    )
    parser.add_argument(
        "--subcategory",
        action="append",
        dest="subcategories",
        help="指定要处理的小类；可重复传入，且必须同时指定一个 --category。",
    )
    parser.add_argument(
        "--subcategory-limit",
        type=int,
        help="本次最多处理的小类数，默认读取规划文件，允许 1 或 2。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="generate_all",
        help="按规划顺序逐个生成所选大类中的全部未完成小类。",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="覆盖本次所选小类的规划目标数量。",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        help="每次 API 请求的商品数，默认读取规划文件。",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到{label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是合法 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}顶层必须是 JSON 对象：{path}")
    return value


def load_taxonomy(
    path: Path,
) -> tuple[int, dict[str, list[str]], list[str]]:
    data = load_json_object(path, "类目文件")
    try:
        version = int(data["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("类目文件缺少有效的 version") from exc

    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("类目文件 categories 必须是非空数组")

    taxonomy: dict[str, list[str]] = {}
    all_subcategories: set[str] = set()
    for raw_category in raw_categories:
        if not isinstance(raw_category, dict):
            raise ValueError("categories 中存在非对象元素")
        category = normalize_text(raw_category.get("name"))
        raw_subcategories = raw_category.get("subcategories")
        if not category or category in taxonomy:
            raise ValueError(f"大类名称缺失或重复：{category or '<空>'}")
        if not isinstance(raw_subcategories, list) or not raw_subcategories:
            raise ValueError(f"{category} 的 subcategories 必须是非空数组")
        subcategories = [normalize_text(value) for value in raw_subcategories]
        if any(not value for value in subcategories):
            raise ValueError(f"{category} 包含空的小类名称")
        if len(subcategories) != len(set(subcategories)):
            raise ValueError(f"{category} 包含重复小类")
        duplicates = all_subcategories.intersection(subcategories)
        if duplicates:
            raise ValueError(f"小类在多个大类中重复：{sorted(duplicates)}")
        taxonomy[category] = subcategories
        all_subcategories.update(subcategories)

    missing_support = set(taxonomy).difference(CATEGORY_PREFIXES)
    if missing_support:
        raise ValueError(f"脚本尚未配置这些大类：{sorted(missing_support)}")

    rules: list[str] = []
    for key in ("boundary_rules", "generation_rules"):
        raw_rules = data.get(key, [])
        if not isinstance(raw_rules, list):
            raise ValueError(f"类目文件 {key} 必须是数组")
        rules.extend(normalize_text(rule) for rule in raw_rules if normalize_text(rule))
    return version, taxonomy, rules


def load_plan(
    path: Path,
    *,
    taxonomy_version: int,
    taxonomy: dict[str, list[str]],
) -> dict[str, Any]:
    plan = load_json_object(path, "目标数量文件")
    if plan.get("taxonomy_version") != taxonomy_version:
        raise ValueError("目标数量文件 taxonomy_version 与类目文件 version 不一致")

    default_target = plan.get("default_target_count")
    subcategory_limit = plan.get("subcategory_batch_limit")
    api_batch_size = plan.get("api_batch_size")
    if (
        isinstance(default_target, bool)
        or not isinstance(default_target, int)
        or default_target < 1
    ):
        raise ValueError("default_target_count 必须是正整数")
    if isinstance(subcategory_limit, bool) or subcategory_limit not in (1, 2):
        raise ValueError("subcategory_batch_limit 必须是 1 或 2")
    if (
        isinstance(api_batch_size, bool)
        or not isinstance(api_batch_size, int)
        or not 1 <= api_batch_size <= 10
    ):
        raise ValueError("api_batch_size 必须是 1 至 10 的整数")

    order = plan.get("category_order")
    if (
        not isinstance(order, list)
        or len(order) != len(taxonomy)
        or set(order) != set(taxonomy)
    ):
        raise ValueError("category_order 必须完整且只包含类目文件中的大类")

    overrides = plan.get("target_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("target_overrides 必须是对象")
    for category, subcategory_targets in overrides.items():
        if category not in taxonomy or not isinstance(subcategory_targets, dict):
            raise ValueError(f"target_overrides 包含无效大类：{category}")
        for subcategory, target in subcategory_targets.items():
            if subcategory not in taxonomy[category]:
                raise ValueError(f"{category} 包含无效目标小类：{subcategory}")
            if isinstance(target, bool) or not isinstance(target, int) or target < 1:
                raise ValueError(f"{category}/{subcategory} 的目标数量必须是正整数")
    return plan


def load_existing_rows(
    path: Path,
    taxonomy: dict[str, list[str]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    set[str],
    set[str],
]:
    rows_by_key = {
        (category, subcategory): []
        for category, subcategories in taxonomy.items()
        for subcategory in subcategories
    }
    product_ids: set[str] = set()
    product_names: set[str] = set()
    if not path.exists():
        return rows_by_key, product_ids, product_names

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"输出文件第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(row, dict) or set(row) != set(OUTPUT_FIELDS):
                raise ValueError(f"输出文件第 {line_number} 行字段不符合要求")

            product_id = normalize_text(row.get("product_id"))
            name = normalize_text(row.get("name"))
            category = normalize_text(row.get("category"))
            subcategory = normalize_text(row.get("subcategory"))
            if not product_id or product_id in product_ids:
                raise ValueError(f"输出文件第 {line_number} 行商品 ID 缺失或重复")
            if not name or name in product_names:
                raise ValueError(f"输出文件第 {line_number} 行商品名称缺失或重复")
            if category not in taxonomy or subcategory not in taxonomy[category]:
                raise ValueError(
                    f"输出文件第 {line_number} 行包含未知类目：{category}/{subcategory}"
                )
            product_ids.add(product_id)
            product_names.add(name)
            rows_by_key[(category, subcategory)].append(row)
    return rows_by_key, product_ids, product_names


def target_for(
    plan: dict[str, Any],
    category: str,
    subcategory: str,
    override: int | None,
) -> int:
    if override is not None:
        return override
    category_overrides = plan.get("target_overrides", {}).get(category, {})
    return int(category_overrides.get(subcategory, plan["default_target_count"]))


def select_work(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    taxonomy: dict[str, list[str]],
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    limit: int | None,
) -> list[tuple[str, str, int]]:
    requested_categories = args.categories or list(plan["category_order"])
    if len(requested_categories) != len(set(requested_categories)):
        raise ValueError("--category 不能重复")
    unknown_categories = set(requested_categories).difference(taxonomy)
    if unknown_categories:
        raise ValueError(f"未知大类：{sorted(unknown_categories)}")

    if args.subcategories:
        if not args.categories or len(args.categories) != 1:
            raise ValueError("指定 --subcategory 时必须且只能指定一个 --category")
        if len(args.subcategories) != len(set(args.subcategories)):
            raise ValueError("--subcategory 不能重复")
        if limit is not None and len(args.subcategories) > limit:
            raise ValueError(f"本次最多处理 {limit} 个小类")
        category = args.categories[0]
        unknown = set(args.subcategories).difference(taxonomy[category])
        if unknown:
            raise ValueError(f"{category} 中不存在这些小类：{sorted(unknown)}")
        candidates = [(category, value) for value in args.subcategories]
    else:
        candidates = [
            (category, subcategory)
            for category in plan["category_order"]
            if category in requested_categories
            for subcategory in taxonomy[category]
        ]

    selected: list[tuple[str, str, int]] = []
    for category, subcategory in candidates:
        target = target_for(plan, category, subcategory, args.target_count)
        if len(rows_by_key[(category, subcategory)]) < target:
            selected.append((category, subcategory, target))
        if limit is not None and len(selected) == limit:
            break
    return selected


def stable_subcategory_code(subcategory: str) -> str:
    return hashlib.sha256(subcategory.encode("utf-8")).hexdigest()[:6].upper()


def allocate_product_ids(
    category: str,
    subcategory: str,
    start_sequence: int,
    count: int,
    existing_ids: set[str],
) -> list[str]:
    prefix = CATEGORY_PREFIXES[category]
    subcategory_code = stable_subcategory_code(subcategory)
    sequence = start_sequence
    product_ids: list[str] = []
    while len(product_ids) < count:
        product_id = f"CP-{prefix}-{subcategory_code}-{sequence:04d}"
        sequence += 1
        if product_id in existing_ids:
            continue
        product_ids.append(product_id)
    return product_ids


def parse_response(raw: str) -> dict[str, Any]:
    cleaned = str(raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not cleaned:
        raise ValueError("DeepSeek 返回了空内容")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("DeepSeek 返回内容不是合法 JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("DeepSeek 顶层输出必须是 JSON 对象")
    return data


def validate_generated_products(
    data: dict[str, Any],
    *,
    category: str,
    subcategory: str,
    start_sequence: int,
    product_ids: list[str],
    existing_names: set[str],
) -> list[dict[str, Any]]:
    raw_products = data.get("products")
    if not isinstance(raw_products, list) or len(raw_products) != len(product_ids):
        raise ValueError("products 数量与请求数量不一致")

    expected_sequences = set(range(start_sequence, start_sequence + len(product_ids)))
    received: dict[int, dict[str, Any]] = {}
    for item in raw_products:
        if not isinstance(item, dict):
            raise ValueError("products 中存在非对象元素")
        try:
            sequence = int(item.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("商品 sequence 无效") from exc
        if sequence not in expected_sequences or sequence in received:
            raise ValueError(f"商品 sequence 缺失、重复或越界：{sequence}")
        received[sequence] = item

    rows: list[dict[str, Any]] = []
    batch_names: set[str] = set()
    low_price, high_price = CATEGORY_PRICE_LIMITS[category]
    boundary = SUBCATEGORY_BOUNDARIES.get(subcategory, {})
    required_name_terms = tuple(boundary.get("required_name_terms", ()))
    forbidden_name_terms = tuple(boundary.get("forbidden_name_terms", ()))
    for sequence in sorted(received):
        item = received[sequence]
        name = normalize_text(item.get("name"))
        brand = normalize_text(item.get("brand"))
        description = normalize_text(item.get("description"))
        if not name or len(name) > 100 or not CHINESE_CHARACTER.search(name):
            raise ValueError(f"序号 {sequence} 的商品名称无效")
        if name in existing_names or name in batch_names:
            raise ValueError(f"商品名称重复：{name}")
        if required_name_terms and not any(
            term in name for term in required_name_terms
        ):
            raise ValueError(
                f"序号 {sequence} 的商品名称不符合{subcategory}定义：{name}"
            )
        if forbidden_name_terms and any(term in name for term in forbidden_name_terms):
            raise ValueError(
                f"序号 {sequence} 的商品实际属于其他小类，不能归入{subcategory}：{name}"
            )
        if not brand or len(brand) > 40 or not CHINESE_CHARACTER.search(brand):
            raise ValueError(f"序号 {sequence} 的品牌无效")
        if not 150 <= len(description) <= 400:
            raise ValueError(
                f"序号 {sequence} 的 description 长度应为 150 至 400 个字符"
            )
        if any(term in name or term in description for term in PROHIBITED_TERMS):
            raise ValueError(f"序号 {sequence} 包含禁止宣传用语")

        if isinstance(item.get("price"), bool):
            raise ValueError(f"序号 {sequence} 的 price 无效")
        try:
            price = round(float(item.get("price")), 2)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"序号 {sequence} 的 price 无效") from exc
        if not low_price <= price <= high_price:
            raise ValueError(f"序号 {sequence} 的 price 超出合理校验范围")

        stock = item.get("stock")
        if (
            isinstance(stock, bool)
            or not isinstance(stock, int)
            or not 0 <= stock <= 100
        ):
            raise ValueError(f"序号 {sequence} 的 stock 必须是 0 至 100 的整数")

        raw_tags = item.get("tags")
        if not isinstance(raw_tags, list) or not raw_tags:
            raise ValueError(f"序号 {sequence} 的 tags 必须是非空数组")
        tags: list[str] = []
        seen_tags: set[str] = set()
        for raw_tag in raw_tags:
            tag = normalize_text(raw_tag)
            normalized_tag = tag.casefold()
            if not tag or len(tag) > 40 or normalized_tag in seen_tags:
                continue
            seen_tags.add(normalized_tag)
            tags.append(tag)
        if not tags:
            raise ValueError(f"序号 {sequence} 没有有效 tag")

        rows.append(
            {
                "product_id": product_ids[sequence - start_sequence],
                "name": name,
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "price": price,
                "description": description,
                "tags": tags,
                "stock": stock,
            }
        )
        batch_names.add(name)
    return rows


def build_user_prompt(
    *,
    category: str,
    subcategory: str,
    start_sequence: int,
    count: int,
    existing_names: set[str],
    same_subcategory_names: set[str],
    sibling_subcategories: list[str],
    taxonomy_rules: list[str],
    attempt: int,
    previous_error: str | None,
) -> str:
    boundary = SUBCATEGORY_BOUNDARIES.get(subcategory, {})
    payload: dict[str, Any] = {
        "task": "生成一个固定小类的中文电商演示商品",
        "category": category,
        "subcategory": subcategory,
        "count": count,
        "required_sequences": list(range(start_sequence, start_sequence + count)),
        "category_writing_focus": CATEGORY_FOCUS[category],
        "target_subcategory_definition": boundary.get(
            "definition",
            f"商品本体必须明确属于{subcategory}，不能只是相关配件或搭配商品。",
        ),
        "excluded_sibling_subcategories": sibling_subcategories,
        "taxonomy_and_generation_rules": taxonomy_rules,
        "existing_names_in_this_subcategory": sorted(same_subcategory_names),
        "strictly_forbidden_names": sorted(same_subcategory_names),
        "other_names_to_avoid": sorted(existing_names - same_subcategory_names)[-50:],
        "generation_attempt": attempt,
        "variation_token": hashlib.sha256(
            (
                f"{category}:{subcategory}:{start_sequence}:{count}:"
                f"{attempt}:{secrets.token_hex(8)}"
            ).encode()
        ).hexdigest()[:12],
        "required_fields_per_product": [
            "sequence",
            "name",
            "brand",
            "price",
            "description",
            "tags",
            "stock",
        ],
        "important": [
            "本次所有商品的商品本体必须属于指定的同一个小类",
            "不得生成 excluded_sibling_subcategories 中的商品",
            "逐项检查 name，绝对不能复用 strictly_forbidden_names 中的名称",
            "重试时必须更换品牌、商品类型或核心规格，不能原样重复上次结果",
            "不要输出 category、subcategory 或 product_id，由脚本统一写入",
            "tags 数量按商品自然变化，不要求固定为 8 至 15 个",
            "价格必须结合具体商品、规格和定位生成",
        ],
    }
    if previous_error:
        payload["previous_output_problem"] = previous_error
        payload["retry_requirement"] = "修正上述问题并重新输出完整 JSON"
    return json.dumps(payload, ensure_ascii=False)


async def request_batch(
    client: AsyncOpenAI,
    *,
    model: str,
    category: str,
    subcategory: str,
    start_sequence: int,
    count: int,
    product_ids: list[str],
    existing_names: set[str],
    same_subcategory_names: set[str],
    sibling_subcategories: list[str],
    taxonomy_rules: list[str],
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            category=category,
                            subcategory=subcategory,
                            start_sequence=start_sequence,
                            count=count,
                            existing_names=existing_names,
                            same_subcategory_names=same_subcategory_names,
                            sibling_subcategories=sibling_subcategories,
                            taxonomy_rules=taxonomy_rules,
                            attempt=attempt,
                            previous_error=str(last_error) if last_error else None,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=min(1.2, 0.7 + attempt * 0.1),
                max_tokens=8192,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("DeepSeek 输出因 max_tokens 被截断")
            return validate_generated_products(
                parse_response(choice.message.content or ""),
                category=category,
                subcategory=subcategory,
                start_sequence=start_sequence,
                product_ids=product_ids,
                existing_names=existing_names,
            )
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(2 ** (attempt - 1))
    raise RuntimeError(
        f"{category}/{subcategory} 批次生成失败：{last_error}"
    ) from last_error


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def generate_subcategory(
    client: AsyncOpenAI,
    *,
    output_path: Path,
    category: str,
    subcategory: str,
    target_count: int,
    api_batch_size: int,
    existing_rows: list[dict[str, Any]],
    catalog_ids: set[str],
    catalog_names: set[str],
    sibling_subcategories: list[str],
    taxonomy_rules: list[str],
    model: str,
) -> None:
    completed = len(existing_rows)
    while completed < target_count:
        count = min(api_batch_size, target_count - completed)
        names_snapshot = set(catalog_names)
        same_subcategory_names = {str(row["name"]) for row in existing_rows}
        product_ids = allocate_product_ids(
            category,
            subcategory,
            completed + 1,
            count,
            catalog_ids,
        )

        rows = await request_batch(
            client,
            model=model,
            category=category,
            subcategory=subcategory,
            start_sequence=completed + 1,
            count=count,
            product_ids=product_ids,
            existing_names=names_snapshot,
            same_subcategory_names=same_subcategory_names,
            sibling_subcategories=sibling_subcategories,
            taxonomy_rules=taxonomy_rules,
        )

        duplicate_names = {row["name"] for row in rows}.intersection(catalog_names)
        duplicate_ids = {row["product_id"] for row in rows}.intersection(catalog_ids)
        if duplicate_names or duplicate_ids:
            raise RuntimeError(
                f"{category}/{subcategory} 生成结果发生重复："
                f"names={sorted(duplicate_names)}, ids={sorted(duplicate_ids)}"
            )

        append_rows(output_path, rows)
        catalog_names.update(str(row["name"]) for row in rows)
        catalog_ids.update(str(row["product_id"]) for row in rows)
        existing_rows.extend(rows)

        completed += len(rows)
        print(f"{category}/{subcategory}：{completed}/{target_count}")


async def async_main() -> int:
    args = parse_args()
    taxonomy_path = resolve_path(args.taxonomy).resolve()
    plan_path = resolve_path(args.plan).resolve()
    taxonomy_version, taxonomy, taxonomy_rules = load_taxonomy(taxonomy_path)
    plan = load_plan(
        plan_path,
        taxonomy_version=taxonomy_version,
        taxonomy=taxonomy,
    )

    raw_output = args.output
    if raw_output is None:
        configured_output = normalize_text(plan.get("output_path"))
        if not configured_output:
            raise ValueError("目标数量文件缺少 output_path")
        raw_output = Path(configured_output)
    output_path = resolve_path(raw_output).resolve()
    if output_path in {taxonomy_path, plan_path}:
        raise ValueError("output_path 不能与类目文件或目标数量文件相同")

    if args.generate_all and args.subcategories:
        raise ValueError("--all 不能与 --subcategory 同时使用")
    if args.generate_all and args.subcategory_limit is not None:
        raise ValueError("--all 不能与 --subcategory-limit 同时使用")

    limit = (
        None
        if args.generate_all
        else args.subcategory_limit or int(plan["subcategory_batch_limit"])
    )
    api_batch_size = args.api_batch_size or int(plan["api_batch_size"])
    if limit is not None and limit not in (1, 2):
        raise ValueError("--subcategory-limit 必须是 1 或 2")
    if not 1 <= api_batch_size <= 10:
        raise ValueError("--api-batch-size 必须在 1 至 10 之间")
    if args.target_count is not None and args.target_count < 1:
        raise ValueError("--target-count 必须大于 0")
    if args.timeout <= 0:
        raise ValueError("--timeout 必须大于 0")

    rows_by_key, catalog_ids, catalog_names = load_existing_rows(
        output_path,
        taxonomy,
    )
    selected = select_work(
        args=args,
        plan=plan,
        taxonomy=taxonomy,
        rows_by_key=rows_by_key,
        limit=limit,
    )
    if not selected:
        print("所选范围内的小类均已达到目标数量，无需生成。")
        return 0

    print(f"输出文件：{output_path}")
    if args.generate_all:
        print(f"全量顺序模式：共 {len(selected)} 个未完成小类")
    else:
        print("本次任务：")
    for category, subcategory, target in selected:
        current = len(rows_by_key[(category, subcategory)])
        print(
            f"- {category}/{subcategory}：已有 {current}，目标 {target}，缺少 {target - current}"
        )
    if args.dry_run:
        print("dry-run：未调用 DeepSeek，未写入文件。")
        return 0

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )

    async def run_selected(category: str, subcategory: str, target: int) -> None:
        await generate_subcategory(
            client,
            output_path=output_path,
            category=category,
            subcategory=subcategory,
            target_count=target,
            api_batch_size=api_batch_size,
            existing_rows=rows_by_key[(category, subcategory)],
            catalog_ids=catalog_ids,
            catalog_names=catalog_names,
            sibling_subcategories=[
                value for value in taxonomy[category] if value != subcategory
            ],
            taxonomy_rules=taxonomy_rules,
            model=args.model,
        )

    try:
        for category, subcategory, target in selected:
            await run_selected(category, subcategory, target)
    finally:
        await client.close()

    print("本次所选小类已达到目标数量。")
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"生成失败：{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
