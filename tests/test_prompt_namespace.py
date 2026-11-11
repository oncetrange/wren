from wren.agent.prompt import BASE_PROMPT


def test_prompt_warns_about_shared_test_namespaces() -> None:
    assert "distinctive names" in BASE_PROMPT
    assert "collide" in BASE_PROMPT
