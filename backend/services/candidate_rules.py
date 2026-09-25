from __future__ import annotations

from typing import Any

from services.product_constraints import hard_constraint_failures

from models.schemas import (
    CandidateEvaluation,
    CandidateEvaluationResult,
    Product,
    ProductEvidence,
)


class RuleCandidateEvaluator:
    """Pure rule strategy used by the internal candidate evaluation service."""

    def evaluate(
        self, *, context: dict[str, Any], products: list[Product]
    ) -> CandidateEvaluationResult:
        evaluations = [self._evaluate(product, context) for product in products]
        return CandidateEvaluationResult(
            success=True,
            evaluations=evaluations,
            data={
                "evaluations": [item.model_dump() for item in evaluations],
                "candidate_count": len(products),
                "evaluated_count": len(evaluations),
                "evaluation_strategy": "deterministic_lightweight",
                "feedback_applied": False,
            },
            confidence=1.0,
        )

    def _evaluate(
        self,
        product: Product,
        context: dict[str, Any],
    ) -> CandidateEvaluation:
        budget = self._positive_float(context.get("budget"))
        categories = self._target_categories(context)
        avoid_categories = {
            str(item).strip()
            for item in context.get("avoid_categories") or []
            if str(item).strip()
        }
        keywords = self._goal_keywords(context)
        constraints = self._constraints(context)
        product_text = " ".join(
            [
                product.name,
                product.category,
                product.brand,
                product.description,
                *product.tags,
            ]
        ).lower()

        score = 50.0
        matched: list[str] = []
        unmet = hard_constraint_failures(product, context)
        evidence: list[ProductEvidence] = []

        if budget is not None:
            evidence.append(ProductEvidence(field="price", value=product.price))
            if product.price <= budget:
                score += 12.0
                matched.append(f"价格不超过预算 ¥{budget:g}")
            else:
                score -= 35.0

        if categories:
            evidence.append(
                ProductEvidence(field="category", value=product.category)
            )
            if product.category in categories:
                score += 18.0
                matched.append("符合目标品类")
            else:
                score -= 30.0

        if product.category in avoid_categories:
            if not any(item.field == "category" for item in evidence):
                evidence.append(
                    ProductEvidence(field="category", value=product.category)
                )
            score -= 40.0

        matched_keywords = [
            keyword for keyword in keywords if keyword.lower() in product_text
        ]
        if matched_keywords:
            matched.append(f"匹配关键词：{'、'.join(matched_keywords[:3])}")
            score += min(15.0, len(matched_keywords) * 5.0)
            keyword_evidence = self._keyword_evidence(product, matched_keywords)
            if keyword_evidence is not None:
                evidence.append(keyword_evidence)

        score += min(max(float(product.score or 0.0), 0.0), 1.0) * 15.0
        hard_constraints_passed = not unmet
        if not hard_constraints_passed:
            score = min(score, 30.0)

        unverified = [
            constraint
            for constraint in constraints
            if constraint.lower() not in product_text
        ]
        reason = (
            "满足明确预算与品类要求，按召回得分和关键词完成轻量评估"
            if hard_constraints_passed
            else "未满足明确预算或品类要求"
        )
        return CandidateEvaluation(
            product_id=product.product_id,
            fit_score=round(min(100.0, max(0.0, score)), 2),
            hard_constraints_passed=hard_constraints_passed,
            matched_preferences=matched,
            unmet_constraints=unmet,
            unverified_requirements=unverified,
            evidence=evidence,
            reason=reason,
        )

    @staticmethod
    def _positive_float(raw: Any) -> float | None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _target_categories(context: dict[str, Any]) -> set[str]:
        if context.get("skip_category_filter"):
            return set()
        categories = {
            str(item).strip()
            for item in context.get("categories") or []
            if str(item).strip()
        }
        if context.get("category"):
            categories.add(str(context["category"]))
        for goal in context.get("shopping_goals") or []:
            if isinstance(goal, dict) and str(goal.get("category") or "").strip():
                categories.add(str(goal["category"]).strip())
        return categories

    @staticmethod
    def _goal_keywords(context: dict[str, Any]) -> list[str]:
        keywords: list[str] = []
        for goal in context.get("shopping_goals") or []:
            if not isinstance(goal, dict):
                continue
            for keyword in goal.get("keywords") or []:
                value = str(keyword).strip()
                if value and value not in keywords:
                    keywords.append(value)
        return keywords[:8]

    @staticmethod
    def _constraints(context: dict[str, Any]) -> list[str]:
        values = list(context.get("constraints") or [])
        for goal in context.get("shopping_goals") or []:
            if isinstance(goal, dict):
                values.extend(goal.get("constraints") or [])
        return list(
            dict.fromkeys(str(item).strip() for item in values if str(item).strip())
        )

    @staticmethod
    def _keyword_evidence(
        product: Product,
        keywords: list[str],
    ) -> ProductEvidence | None:
        fields: tuple[tuple[str, Any], ...] = (
            ("name", product.name),
            ("tags", product.tags),
            ("brand", product.brand),
            ("description", product.description),
        )
        for field, value in fields:
            text = " ".join(value) if isinstance(value, list) else str(value)
            if any(keyword.lower() in text.lower() for keyword in keywords):
                return ProductEvidence(field=field, value=value)
        return None
