from nerion_digital_physicist.generation.sampler import TemplateSampler, TemplateSpec


def test_sampler_deterministic_with_seed():
    specs = [TemplateSpec("a", weight=1.0), TemplateSpec("b", weight=3.0)]
    sampler1 = TemplateSampler(specs, seed=123)
    sampler2 = TemplateSampler(specs, seed=123)

    seq1 = [spec.template_id for spec in sampler1.sequence(5)]
    seq2 = [spec.template_id for spec in sampler2.sequence(5)]

    assert seq1 == seq2


def test_sampler_requires_templates():
    try:
        TemplateSampler([], seed=0)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty template list"
