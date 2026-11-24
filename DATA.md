# Data sources

Active speakers: **Narendra Modi**, **Donald Trump** and **Barack Obama**. Obama is a placeholder until the third speaker is chosen. Each speaker is capped at roughly 80k words, so differences between speakers come from style and not from how much text each one has.

| Speaker | Words (cleaned) | Speeches | Source |
|---|---:|---:|---|
| Narendra Modi | 81,614 | 47 | English texts on the Prime Minister's Office site, [pmindia.gov.in/en/tag/pmspeech](https://www.pmindia.gov.in/en/tag/pmspeech/), 2025–2026. The list is in `data/sources/modi.txt` and the raw export in `data/sources/modi_browser_export.json`. |
| Donald Trump | 85,809 | 34 | Remarks on the archived Trump White House site, [trumpwhitehouse.archives.gov/remarks](https://trumpwhitehouse.archives.gov/remarks/), 2020–2021: rallies at airports, addresses to the nation, the UN General Assembly, commencement speeches, the farewell address. The list is in `data/sources/trump.txt`. |
| Barack Obama | 67,074 | 14 | The archived Obama White House site: both inaugurals, five State of the Union addresses, Cairo, Tucson, Brussels, Selma, Charleston, Havana, and the farewell address. URLs are in `fetch_data.py`. |

## Rebuilding the dataset

```bash
python fetch_data.py trump obama     # downloads into data/raw/<speaker>/
python fetch_data.py modi            # imports data/sources/modi_browser_export.json
python clean.py                      # -> data/speakers/<speaker>.txt
```

- **Trump:** `python fetch_data.py --discover trump` rebuilds the URL list from the archive. It skips formats that are mostly Q&A (departures, roundtables, bilaterals, press conferences, town halls) and stops at about 80k words.
- **Modi:** pmindia.gov.in and pib.gov.in refuse non-browser clients, so the text was exported once from a normal Chrome session with `scripts/modi_browser_export.js`. That script fetches the speech listing and pages at about 3 requests per second and downloads a JSON file. To refresh it, run the script in the browser console on the site, then run `python fetch_data.py modi --modi-json <downloaded file>`.

## Important notes about the text

- **Modi's speeches are mostly translations.** He usually speaks in Hindi, and the site publishes an official English rendering. The model therefore learns the style of the PMO's English translations, which follow his structure and catchphrases ("Friends,", "my dear countrymen", "Bharat Mata ki Jai"). The word choice is partly the translator's. Lines in other scripts are dropped.
- **Trump's transcripts are of delivered remarks**, including ad-libs. Only text spoken by `THE PRESIDENT` is kept: reporter questions (`Q`) and other speakers' turns are removed.
- **Obama:** venue and time headers and audience lines are removed.
- `clean.py` also removes stage directions such as `(Applause.)`, URLs, HTML and dates, and normalises quotes and whitespace.

## Train / validation split

`corpus.py` cuts each speaker's text into 200-word chunks and holds out every 10th chunk (10%). All models use the same split.

## Retired speakers

Gandhi (*Freedom's Battle*, Project Gutenberg #10366) and Lincoln (Gutenberg #48364 and #3253) were used in the first version. Their raw text is in `data/retired/`, and `fetch_data.py gandhi lincoln` still downloads them. To reactivate one, move its folder back to `data/raw/` and run `clean.py`.

## Copyright and responsible use

Trump's and Obama's White House transcripts are works of the US federal government and are in the public domain. Modi's speeches are published by the Government of India and are used here only to train models for a non-commercial class project.

Generated text is synthetic. It must never be presented as a real quote from any of these people, and the app labels every output that way. All three are prominent political figures, so be especially careful when sharing outputs: fake political quotes can spread and mislead.
