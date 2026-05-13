#!/usr/bin/env python3
"""
Record a demo video of the Skill Selector compare view.
Uses Playwright's built-in video recording.

Output: screenshots/demo.webm  (then converted to mp4 via ffmpeg)
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

BASE_URL = "http://192.168.88.102:8200"
OUT_DIR  = Path(__file__).parent / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

QUERY = (
    "I need something to keep tabs on what my agent gets up to "
    "while I'm not watching"
)

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        context = await browser.new_context(
            viewport={"width": 1400, "height": 860},
            device_scale_factor=2,
            record_video_dir=str(OUT_DIR),
            record_video_size={"width": 1400, "height": 860},
        )
        page = await context.new_page()

        print(f"Opening {BASE_URL}...")
        await page.goto(BASE_URL, wait_until="networkidle")

        # Wait for pointcloud to finish loading (loading overlay disappears)
        await page.wait_for_selector("#loading-overlay", state="hidden", timeout=30000)
        print("  Pointcloud loaded")

        # Let the auto-rotating pointcloud spin for a few seconds
        await page.wait_for_timeout(4000)

        # Scroll into view and focus the textarea
        query_input = page.locator("#query-input")
        await query_input.click()
        await page.wait_for_timeout(300)

        # Type the query character by character for a natural feel
        print(f"  Typing query...")
        await query_input.type(QUERY, delay=28)
        await page.wait_for_timeout(600)

        # Submit
        print("  Submitting...")
        await page.click("#search-btn")

        # Wait for results to appear
        await page.wait_for_selector("#compare-cols:not(.hidden)", timeout=30000)
        print("  Results appeared")

        # Let the animation play: query sphere appears, camera flies, lines draw
        await page.wait_for_timeout(2500)

        # Pause on results
        await page.wait_for_timeout(3000)

        # Switch to domain/local view to show it
        local_btn = page.locator("#local-view-btn")
        is_disabled = await local_btn.get_attribute("disabled")
        if not is_disabled:
            await local_btn.click()
            await page.wait_for_timeout(2000)
            print("  Local domain view shown")

        # Back to global
        await page.click("#global-view-btn")
        await page.wait_for_timeout(1500)

        # Click the first spatial card to show the modal
        first_card = page.locator("#spatial-cards .compare-card").first
        if await first_card.count() > 0:
            skill_name = await page.evaluate(
                "document.querySelector('#spatial-cards .cc-name')?.textContent?.trim()"
            )
            print(f"  Opening modal for: {skill_name!r}")
            await page.evaluate(f"window.__openSkillModal({repr(skill_name)})")
            await page.wait_for_function(
                "!document.getElementById('modal-overlay').classList.contains('hidden')",
                timeout=6000
            )
            await page.wait_for_timeout(2000)
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(800)

        # Final pause showing the full UI
        await page.wait_for_timeout(2000)

        print("  Closing browser and saving video...")
        await context.close()
        await browser.close()

    # Find the recorded webm (Playwright names it with a hash)
    videos = sorted(OUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if not videos:
        print("No video found!")
        return
    raw = videos[-1]
    out_mp4 = OUT_DIR / "demo.mp4"
    print(f"  Raw video: {raw.name} ({raw.stat().st_size // 1024}KB)")

    # Convert to mp4 with ffmpeg
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw),
         "-vf", "scale=1400:860",
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-movflags", "+faststart",
         str(out_mp4)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  Converted → demo.mp4 ({out_mp4.stat().st_size // 1024}KB)")
        raw.unlink()  # remove the raw webm
    else:
        print(f"  ffmpeg failed: {result.stderr[-300:]}")
        print(f"  Raw webm kept at: {raw}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
