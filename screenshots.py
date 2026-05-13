"""
Playwright screenshot script for Skill Selector.
Captures: search state, global pointcloud, local domain view.
"""
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://192.168.88.102:8200"
QUERY = "I need to debug a tricky bug in my code and write tests to prevent regression"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # Desktop viewport
        ctx = await browser.new_context(
            viewport={"width": 1400, "height": 860},
            device_scale_factor=2,
        )
        page = await ctx.new_page()

        print(f"Opening {URL}...")
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # ── 1. Empty state ────────────────────────────────────────────────────
        await page.screenshot(path=OUT / "01-empty.png", full_page=False)
        print("  ✓ 01-empty.png")

        # ── 2. Type query ─────────────────────────────────────────────────────
        textarea = page.locator("#query-input")
        await textarea.click()
        await textarea.fill(QUERY)
        await page.screenshot(path=OUT / "02-query-typed.png", full_page=False)
        print("  ✓ 02-query-typed.png")

        # ── 3. Submit and wait for results ────────────────────────────────────
        await page.locator("#search-btn").click()
        # Wait for cards to appear
        await page.wait_for_selector(".skill-card", timeout=30000)
        await page.wait_for_timeout(600)
        await page.screenshot(path=OUT / "03-results-cards.png", full_page=False)
        print("  ✓ 03-results-cards.png")

        # ── 4. Open the map (global view) ─────────────────────────────────────
        await page.locator("#toggle-map-btn").click()
        # Wait for canvas to appear and pointcloud to load
        await page.wait_for_selector("#map-canvas", state="visible")
        await page.wait_for_timeout(4000)  # let UMAP data load + animate
        await page.screenshot(path=OUT / "04-global-map.png", full_page=False)
        print("  ✓ 04-global-map.png")

        # ── 5. Global map + cards side by side ───────────────────────────────
        # Already in this state, just capture it cleanly
        await page.wait_for_timeout(500)
        await page.screenshot(path=OUT / "05-split-view.png", full_page=False)
        print("  ✓ 05-split-view.png")

        # ── 6. Switch to local domain view ────────────────────────────────────
        local_btn = page.locator("#local-view-btn")
        await local_btn.click()
        await page.wait_for_timeout(3000)  # local UMAP loads
        await page.screenshot(path=OUT / "06-local-domain.png", full_page=False)
        print("  ✓ 06-local-domain.png")

        # ── 7. Hover a skill card to show tooltip ─────────────────────────────
        cards = page.locator(".skill-card")
        first_card = cards.first
        await first_card.hover()
        await page.wait_for_timeout(300)
        await page.screenshot(path=OUT / "07-card-hover.png", full_page=False)
        print("  ✓ 07-card-hover.png")

        # ── 8. Click a card → skill modal ─────────────────────────────────────
        # Fetch skill name then open modal by directly manipulating the DOM
        skill_name = await page.evaluate("document.querySelector('.skill-card .card-name')?.textContent?.trim()")
        print(f"  → opening modal for: {skill_name!r}")
        has_fn = await page.evaluate("typeof window.__openSkillModal")
        print(f"  → window.__openSkillModal type: {has_fn}")
        # Directly fetch and show modal
        await page.evaluate("""async (name) => {
            const res = await fetch('/api/skills/' + encodeURIComponent(name));
            const skill = await res.json();
            document.getElementById('modal-title').textContent = skill.name;
            document.getElementById('modal-description').textContent = skill.description || '';
            document.getElementById('modal-badges').innerHTML = '';
            document.getElementById('modal-content').innerHTML = '';
            document.getElementById('modal-source-link').href = skill.url || '#';
            document.getElementById('modal-overlay').classList.remove('hidden');
        }""", skill_name)
        await page.wait_for_timeout(400)
        await page.screenshot(path=OUT / "08-skill-modal.png", full_page=False)
        print("  ✓ 08-skill-modal.png")

        # ── Mobile versions ───────────────────────────────────────────────────
        await ctx.close()
        mobile_ctx = await browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=3,
        )
        mpage = await mobile_ctx.new_page()
        await mpage.goto(URL, wait_until="networkidle")
        await mpage.wait_for_timeout(800)

        # Mobile: type + search
        await mpage.locator("#query-input").fill(QUERY)
        await mpage.locator("#search-btn").click()
        await mpage.wait_for_selector(".skill-card", timeout=30000)
        await mpage.wait_for_timeout(500)
        await mpage.screenshot(path=OUT / "09-mobile-results.png", full_page=False)
        print("  ✓ 09-mobile-results.png")

        # Mobile: open map
        await mpage.locator("#toggle-map-btn").click()
        await mpage.wait_for_timeout(4000)
        await mpage.screenshot(path=OUT / "10-mobile-map.png", full_page=False)
        print("  ✓ 10-mobile-map.png")

        await mobile_ctx.close()
        await browser.close()

    print(f"\nAll screenshots saved to {OUT}/")


asyncio.run(main())
