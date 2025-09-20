from __future__ import annotations

from app.parent.tools_manifest import ToolsManifest, ToolSpec, ToolParam


def test_as_prompt_block_includes_when_guidance():
    manifest = ToolsManifest(
        tools=[
            ToolSpec(
                name="read_url",
                description="Fetch and summarize a URL",
                params=[ToolParam(name="url", type="string", description="HTTP/HTTPS URL")],
            ),
            ToolSpec(
                name="web_search",
                description="Search the web",
                params=[ToolParam(name="query", type="string", description="search string")],
            ),
            ToolSpec(
                name="rename_symbol",
                description="Plan safe code renames",
                params=[ToolParam(name="old", type="string"), ToolParam(name="new", type="string")],
            ),
        ]
    )
    block = manifest.as_prompt_block()
    assert "when: use to fetch and summarize a single URL" in block
    assert "when: use for open-ended queries or when no specific URL is provided" in block
    assert "when: use to plan safe code renames" in block

