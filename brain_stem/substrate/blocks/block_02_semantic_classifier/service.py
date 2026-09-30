from ...contracts import Intent, IntentProfile


class SemanticClassifier:
    _CATEGORY_TERMS = {
        "code": {"code", "implement", "refactor", "debug", "program"},
        "analysis": {"analyze", "explain", "compare", "review", "summarize"},
    }

    def classify(self, intent: Intent) -> IntentProfile:
        words = set(intent.goal.lower().split())
        for category, terms in self._CATEGORY_TERMS.items():
            if words & terms:
                return IntentProfile(category=category)
        return IntentProfile(category="general")