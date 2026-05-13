#!/usr/bin/env python3
"""Screenshots of the compare UI — v2."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

BASE_URL = "http://192.168.88.102:8200"
OUT      = Path(__file__).parent / "screenshots_v2"
OUT.mkdir(exist_ok=True)

QUERY = "I need something to keep tabs on what my agent gets up to while I'm not watching"

async def shot(page, name):
    await page.screenshot(path=OUT / name, full_page=False)
    print(f"  ✓ {name}")

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx  = await browser.new_context(viewport={"width": 1400, "height": 860}, device_scale_factor=2)
        page = await ctx.new_page()

        print(f"Opening {BASE_URL}...")
        await page.goto(BASE_URL, wait_until="networkidle")
        await page.wait_for_selector("#loading-overlay", state="hidden", timeout=30000)

        # 01 — empty state, pointcloud rotating
        await page.wait_for_timeout(1200)
        await shot(page, "01-empty.png")

        # 02 — query typed
        await page.click("#query-input")
        await page.fill("#query-input", QUERY)
        await page.wait_for_timeout(400)
        await shot(page, "02-query-typed.png")

        # 03 — submit and wait for results
        await page.click("#search-btn")
        await page.wait_for_selector("#compare-cols:not(.hidden)", timeout=30000)
        await page.wait_for_timeout(2200)   # let camera fly + lines draw
        await shot(page, "03-results-compare.png")

        # 04 — hover first spatial card
        first_spatial = page.locator("#spatial-cards .compare-card").first
        await first_spatial.hover()
        await page.wait_for_timeout(200)
        await shot(page, "04-spatial-hover.png")

        # 05 — hover first semantic card
        first_semantic = page.locator("#semantic-cards .compare-card").first
        await first_semantic.hover()
        await page.wait_for_timeout(200)
        await shot(page, "05-semantic-hover.png")

        # 06 — skill modal (spatial result)
        skill_name = await page.evaluate(
            "document.querySelector('#spatial-cards .cc-name')?.textContent?.trim()"
        )
        print(f"  → modal: {skill_name!r}")
        await page.evaluate(f"window.__openSkillModal({repr(skill_name)})")
        await page.wait_for_function(
            "!document.getElementById('modal-overlay').classList.contains('hidden')",
            timeout=8000
        )
        await page.wait_for_timeout(500)
        await shot(page, "06-skill-modal.png")
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)

        # 07 — toggle cluster wireframes on
        await page.click("#toggle-wires-btn")
        await page.wait_for_timeout(600)
        await shot(page, "07-wires-on.png")
        await page.click("#toggle-wires-btn")  # reset

        # 08 — mobile
        mob_ctx  = await pw.chromium.launch(headless=True)
        mob_page = await (await mob_ctx.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=3
        )).new_page()
        await mob_page.goto(BASE_URL, wait_until="networkidle")
        await mob_page.wait_for_selector("#loading-overlay", state="hidden", timeout=30000)
        await mob_page.wait_for_timeout(800)
        await mob_page.screenshot(path=OUT / "08-mobile-empty.png")
        print("  ✓ 08-mobile-empty.png")

        await mob_page.click("#query-input")
        await mob_page.fill("#query-input", QUERY)
        await mob_page.click("#search-btn")
        await mob_page.wait_for_selector("#compare-cols:not(.hidden)", timeout=30000)
        await mob_page.wait_for_timeout(2000)
        await mob_page.screenshot(path=OUT / "09-mobile-results.png")
        print("  ✓ 09-mobile-results.png")

        await mob_ctx.close()

        print(f"\nAll screenshots → {OUT}/")

asyncio.run(main())
