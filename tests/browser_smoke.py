"""Small Playwright smoke test for a running local laboratory server.

Run it after starting ``emergent-server``::

    python tests/browser_smoke.py

The test is deliberately not collected by pytest because it needs a live web
server and an installed browser. It is a focused interaction check instead of
a large end-to-end suite.
"""

from __future__ import annotations

import argparse


def wait_for_generation(page, element_id: str, expected: int) -> None:
    page.wait_for_function(
        "args => Number(document.getElementById(args.id).textContent) === args.expected",
        arg={"id": element_id, "expected": expected},
    )


def run_smoke(url: str, *, headed: bool = False) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "browser smoke tests require Playwright; install with "
            "`python -m pip install -e '.[browser]'`"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        requested_urls: list[str] = []
        page.on("request", lambda request: requested_urls.append(request.url))
        page.goto(url, wait_until="domcontentloaded")

        page.locator('[data-testid="ca-canvas"]').wait_for(state="visible")
        page.wait_for_function(
            "() => document.getElementById('status-message').textContent.includes('Ready -')"
        )
        page.locator('[data-testid="step-button"]').click()
        wait_for_generation(page, "generation-value", 1)
        page.locator("#rule-text").fill("B36/S23")
        page.locator("#apply-rule-button").click()
        page.wait_for_function(
            "() => document.getElementById('rule-display').textContent === 'B36/S23'"
        )
        assert page.locator("#rule-display").inner_text() == "B36/S23"

        generation_before_playback = int(page.locator("#generation-value").inner_text())
        page.locator('[data-testid="play-button"]').click()
        wait_for_generation(page, "generation-value", generation_before_playback + 1)
        page.locator('[data-mode="3d"]').click()
        page.locator('[id="viewport-3d"] canvas').wait_for(state="visible")
        page.wait_for_timeout(250)
        page.locator('[data-mode="2d"]').click()
        paused_generation = int(page.locator("#generation-value").inner_text())
        page.wait_for_timeout(300)
        assert int(page.locator("#generation-value").inner_text()) == paused_generation

        page.locator('[data-mode="3d"]').click()
        page.locator('[id="viewport-3d"] canvas').wait_for(state="visible")
        assert any(url.endswith("/vendor/three/three.module.js") for url in requested_urls)
        assert not any("cdn.jsdelivr.net" in url for url in requested_urls)
        page.wait_for_function(
            "() => document.getElementById('3d-status-message').textContent.includes('Ready')"
        )
        page.locator('[id="3d-step-button"]').click()
        wait_for_generation(page, "3d-generation-value", 1)
        page.locator('[id="3d-view-menu"] summary').click()
        page.locator('[id="3d-performance-input"]').check()
        page.locator('[id="3d-axis-input"]').select_option("y")
        page.locator('[id="3d-slice-input"]').check()
        page.locator('[id="3d-slice-panel"]').wait_for(state="visible")
        page.locator('[data-camera="top"]').click()
        with page.expect_download() as download_info:
            page.locator('[id="3d-export-button"]').click()
        assert download_info.value.suggested_filename.endswith(".npz")

        page.locator('[data-mode="compare"]').click()
        page.locator('[id="compare-2d-canvas"]').wait_for(state="visible")
        page.wait_for_function(
            "() => document.getElementById('compare-status').textContent.includes('Ready')"
        )
        page.locator('[id="compare-step"]').click()
        page.locator('[id="compare-2d-stats"]').wait_for(state="visible")
        page.wait_for_function(
            "() => document.getElementById('compare-2d-stats').textContent.includes('Generation 1')"
        )

        page.locator('[data-mode="experiments"]').click()
        page.locator("#experiments-heading").wait_for(state="visible")
        page.locator("#experiment-dimensions").select_option("3")
        page.locator("#experiment-rules").fill("100")
        page.locator("#experiment-conditions").fill("100")
        page.locator("#experiment-size").fill("96")
        page.locator("#experiment-steps").fill("2000")
        page.locator("#experiment-run").click()
        page.locator("#experiment-progress").wait_for(state="visible")
        page.wait_for_function(
            "() => document.getElementById('experiment-progress').textContent.includes('too large')"
        )

        page.locator("#experiment-rules").fill("1")
        page.locator("#experiment-conditions").fill("1")
        page.locator("#experiment-size").fill("8")
        page.locator("#experiment-steps").fill("1")
        page.locator("#experiment-run").click()
        page.wait_for_function(
            "() => document.getElementById('experiment-progress').textContent.includes('Completed')"
        )
        with page.expect_download() as download_info:
            page.locator("#experiment-export").click()
        assert download_info.value.suggested_filename.endswith(".csv")
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    run_smoke(args.url, headed=args.headed)
    print("browser smoke passed")


if __name__ == "__main__":
    main()
