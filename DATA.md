# Data sources

Everything in `data/` can be rebuilt with:

```bash
python fetch_data.py      # downloads raw text into data/raw/<speaker>/
python clean.py           # cleans and merges into data/speakers/<speaker>.txt
```

| Speaker | Words (cleaned) | Era / style | Source | Licence |
|---|---:|---|---|---|
| Mahatma Gandhi | 73,551 | 1919–1922 essays and speeches on non-co-operation, Khilafat, swaraj; moral, plain, argumentative | *Freedom's Battle: Being a Comprehensive Collection of Writings and Speeches on the Present Situation* (2nd ed., 1922), [Project Gutenberg #10366](https://www.gutenberg.org/ebooks/10366) | Public domain |
| Abraham Lincoln | 39,912 | 1837–1865 speeches; long legal reasoning, 19th-century rhetoric | [Gutenberg #48364](https://www.gutenberg.org/ebooks/48364) *Noted Speeches of Abraham Lincoln*: Cooper Union address, First and Second Inaugurals, Gettysburg Address, Ottawa debate reply. [Gutenberg #3253](https://www.gutenberg.org/ebooks/3253) *Papers and Writings*: Lyceum Address (1837), Peoria Speech (1854), Farewell Address at Springfield (1861) | Public domain |
| Barack Obama | 67,074 | 2009–2017 presidential addresses; modern, short sentences, applause lines | 14 transcripts from the archived Obama White House site, [obamawhitehouse.archives.gov](https://obamawhitehouse.archives.gov/briefing-room/speeches-and-remarks): both inaugurals, the 2011–2015 State of the Union addresses, Cairo (2009), Tucson memorial (2011), Brussels (2014), Selma 50th anniversary (2015), Charleston eulogy (2015), Havana (2016), and the Farewell Address (2017). Exact URLs are listed in `fetch_data.py`. | Works of the US federal government, public domain in the US (17 U.S.C. § 105) |

Gutenberg files are downloaded from the PGLAF mirror (`gutenberg.pglaf.org`), which serves the same files as gutenberg.org.

## Why not APJ Abdul Kalam?

The first plan used Kalam, Gandhi and Obama. Kalam died in 2015, so his speeches and books are still under copyright and there is no public-domain collection to use. Lincoln replaces him. Lincoln's style differs a lot from both of the others, which makes the "guess the speaker" game and the evaluation more interesting.

## What the pipeline removes

`fetch_data.py`:
- Gandhi: drops Gutenberg headers and the editor's introduction by C. Rajagopalachari. Some letters from other people that Gandhi quoted in full before replying to them are still in the text.
- Lincoln: keeps only Lincoln's own words. It drops the biographical sketches, Douglas's opening speech, the Emancipation Proclamation (a legal document, not a speech) and the editor's narration inside the Peoria speech.
- Obama: drops venue and time headers, `END` markers, page furniture, and lines spoken by anyone other than `THE PRESIDENT`.

`clean.py` (`clean_text`): normalises Unicode and quotes, removes URLs, HTML, stage directions such as `(Applause.)` and `[laughter]`, dates, and Gutenberg `_italic_` markers, then collapses whitespace. Each source document becomes one line of `data/speakers/<speaker>.txt`.

## Train / validation split

`corpus.py` splits every speaker the same way for every model. It cuts the text into 200-word chunks and sends every 10th chunk to validation (10%). Evaluation results therefore use text that neither the Markov chains nor GPT-2 saw during training.

## Responsible use

Generated text is synthetic. Do not present it as a real quote from any of these people. The app labels every output this way.
