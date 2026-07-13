"""The golden set is the trust anchor: 100% pass, 100% reproducible, always."""
from harness import pipeline


def test_golden_set_passes_and_reproduces():
    res = pipeline.run_golden()
    failures = res[~res["pass"]]
    assert failures.empty, f"golden failures:\n{failures.to_string()}"
    not_repro = res[~res["reproducible"]]
    assert not_repro.empty, f"non-reproducible:\n{not_repro.to_string()}"


def test_refusals_carry_reframes():
    """Every refusal explains itself and points somewhere useful."""
    for q in ("Forecast revenue for next quarter",
              "Why is morale down this quarter?",
              "What is our customer happiness index?"):
        art = pipeline.answer(q)
        assert art.tier == "Abstained"
        reason = art.headline.removeprefix("Declined: ")
        assert len(reason) > 40, f"refusal for {q!r} is too thin: {reason!r}"
        for roadmap in ("Alpha", "Beta", "V1"):
            assert roadmap not in reason, f"refusal for {q!r} uses roadmap voice: {reason!r}"
