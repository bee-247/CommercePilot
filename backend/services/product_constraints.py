"""Shared hard constraints used before ranking and before publishing products."""

from typing import Any

from models.schemas import Product


def hard_constraint_failures(product: Product, context: dict[str, Any]) -> list[str]:
    failures = []
    try:
        budget = float(context.get("budget") or 0)
    except (TypeError, ValueError):
        budget = 0
    if budget > 0 and product.price > budget:
        failures.append(f"价格 {product.price:g} 超过预算 {budget:g}")
    if product.stock <= 0:
        failures.append("库存不足")
    if product.category in (context.get("avoid_categories") or []):
        failures.append("属于用户排除的类目")
    categories = set(context.get("categories") or [])
    if context.get("category"):
        categories.add(context["category"])
    for goal in context.get("shopping_goals") or []:
        if isinstance(goal, dict) and goal.get("category"):
            categories.add(goal["category"])
    if (
        categories
        and not context.get("skip_category_filter")
        and product.category not in categories
    ):
        failures.append("不属于本轮购物目标的类目")
    return failures
