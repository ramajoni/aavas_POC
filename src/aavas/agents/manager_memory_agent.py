"""Manager memory agent - how this area manager decided requests like this one, and
what happened to those customers.

It runs when an area manager opens a lower-rate request, and again at the moment
they decide. Three steps, each a declared capability:

* **observe** (skill ``decision_factors``) - the factors on the screen, each with its
  signal, and the request reduced to the fields that make two requests alike;
* **recall** (tool ``procedure.decision_memory``, read only) - the manager's own past
  decisions most like this one and other managers', with what each customer did;
* **reflect** (skill ``decision_recall``) - how this manager tends to decide, which
  factors move them, and whether today's suggestion sits with that habit.

It writes nothing. The learning agent is the only agent permitted to write the
governance store, and it is the one that remembers the decision once it is taken.

It is outside the priced path, and it does not move the suggestion. The suggestion is
the rate worth the most to the bank; memory sits beside it and says what this manager
did before. An agent that steered the suggestion toward a manager's habit would teach
the engine that manager's mistakes along with their judgement.
"""

from __future__ import annotations

from typing import Any

from ..analytics import decision_factors, decision_recall
from ..learning import decision_memory

METHOD = ("Recalled from the decisions remembered when area managers decided earlier requests: the factors "
          "as they stood then, the suggestion they were shown and what the customer did after. Similar means "
          "the same reason, a cut of similar size and a similar file; it does not change the suggestion.")


class ManagerMemoryAgent:
    # ------------------------------------------------------------- observe
    def observe(self, request: dict[str, Any], payload: dict[str, Any], applicant: dict[str, Any],
                checklist: dict[str, Any] | None, briefing: dict[str, Any] | None) -> dict[str, Any]:
        rows = decision_factors.factors(request, payload, applicant, checklist, briefing)
        return {"factors": rows, "lean": decision_factors.lean(rows),
                "case": decision_factors.case(request, payload, applicant),
                "suggestion": (checklist or {}).get("suggestion")}

    # -------------------------------------------------------------- recall
    def recall(self, observation: dict[str, Any], *, manager: str, request_id: str | None = None,
               top: int = decision_recall.TOP) -> dict[str, Any]:
        past = decision_memory.episodes(exclude_request=request_id)
        mine = [ep for ep in past if ep["manager"] == manager]
        others = [ep for ep in past if ep["manager"] != manager]
        case = observation.get("case") or {}
        my_cases = decision_recall.nearest(case, mine, top)
        other_cases = decision_recall.nearest(case, others, top)
        return {
            "manager": manager, "episodes": len(past), "your_episodes": len(mine),
            "case": case,
            "yours": {"cases": my_cases,
                      "empty": None if my_cases else (
                          "You have not decided a request like this yet. Memory starts with your decisions: "
                          "each one is remembered with the factors you saw and what the customer did."
                          if mine else "You have no decisions in memory yet. Each decision you take is "
                                       "remembered with the factors you saw and what the customer did.")},
            "others": {"cases": other_cases, "managers": sorted({c["manager"] for c in other_cases})},
            "profile": decision_recall.profile(mine),
            "consistency": decision_recall.consistency(my_cases, observation.get("suggestion")),
            "moves_suggestion": False,
            "method": METHOD,
        }

    def recalled_ids(self, recall: dict[str, Any]) -> list[str]:
        """The episodes shown to the manager, so a decision records what memory put in front of it."""
        return [c["episode_id"] for side in ("yours", "others") for c in recall[side]["cases"]]
