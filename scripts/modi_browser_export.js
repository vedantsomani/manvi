// Export English texts of PM Modi's speeches from pmindia.gov.in (the site only serves browsers).
// 1. Open https://www.pmindia.gov.in/en/tag/pmspeech/ in Chrome.
// 2. Open DevTools > Console, paste this file and press Enter (takes ~1 minute).
// 3. A file catbot_modi_speeches.json is downloaded. Then run:
//      python fetch_data.py modi --modi-json path/to/catbot_modi_speeches.json
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const latin = t => { const L = [...t].filter(c => /\p{L}/u.test(c));
    return L.length ? L.filter(c => c.charCodeAt(0) < 128).length / L.length : 0; };
  const slugs = [];
  for (let n = 1; n <= 8; n++) {
    const html = await (await fetch(`/en/tag/pmspeech/page/${n}/`)).text();
    for (const m of html.matchAll(/news_updates\/([a-z0-9-]+)\//g)) if (!slugs.includes(m[1])) slugs.push(m[1]);
    await sleep(400);
  }
  const texts = {};
  for (const slug of slugs) {
    const doc = new DOMParser().parseFromString(await (await fetch(`/en/news_updates/${slug}/`)).text(), "text/html");
    const box = doc.querySelector(".news-bg");
    if (box) {
      const text = [...box.querySelectorAll("p")].map(p => p.textContent.replace(/\s+/g, " ").trim())
        .filter(t => t && latin(t) > 0.9 && !/DOCTYPE|W3C/.test(t)).join("\n\n");
      if (text.split(/\s+/).length >= 300) texts[slug] = text;  // English text of a real speech
    }
    await sleep(350);
  }
  const order = slugs.filter(s => s in texts);
  const blob = new Blob([JSON.stringify({source: "https://www.pmindia.gov.in/en/news_updates/", order, texts})],
                        {type: "application/json"});
  const a = Object.assign(document.createElement("a"), {href: URL.createObjectURL(blob), download: "catbot_modi_speeches.json"});
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`exported ${order.length} speeches`);
})();
