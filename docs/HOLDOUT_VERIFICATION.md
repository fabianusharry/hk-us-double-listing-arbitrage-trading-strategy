# Out-of-universe candidates — verification checklist

Pre-registered rule: CLAUDE.md §6 ("Out-of-universe robustness test"). This file is the
student's manual checklist. Liquidity, first trading dates and an implied-ratio
cross-check will come from yfinance data in Session 6b — **not** from this file.

Everything in the "Claude's note" column is from memory and may be wrong. That is
exactly why it needs checking.

## For each candidate, check five things

| # | Question | Where to look | What to record |
|---|---|---|---|
| 1 | **Current ADR ratio** (HK shares per ADR) | Deutsche Bank DR directory (adr.db.com → DR Universe → search the US ticker) — the same source used for the original six | ratio + link |
| 2 | **Did the ratio change between 2021-04-19 and 2026-09-30?** | Search: `"<company> ADS ratio change"`, `"<company> change in ADS ratio"`; the company's 6-K filings on SEC EDGAR (search the 6-Ks for "ratio") | date of any change, old → new |
| 3 | **HK listing date and type** (secondary / dual primary / introduction) | HKEXnews (hkexnews.hk) listing documents; or search `"<company> Hong Kong listing <HK code>"` | date; must be on or before 2022-12-31 |
| 4 | **US ticker or domicile change** in the sample | SEC EDGAR company page (former names / tickers); news search `"<company> redomicile"`, `"<company> new ticker"` | date and what changed |
| 5 | **Still dual-listed on 2026-09-30?** (no delisting, privatisation or HK withdrawal) | HKEXnews; Nasdaq/NYSE quote page; news search `"<company> delisting"` | yes / no |

Exclude the candidate if #2, #4 or #5 fails, or if #3 is after 2022-12-31. If #1
differs from the ratio in `config.HOLDOUT_CANDIDATES`, correct it there. Then set
`HOLDOUT_RATIO_VERIFIED[name] = True` only for names you have checked.

## Candidates

| Name in config | US | HK (Yahoo) | Ratio (Citi, verified) | Citi "Product Milestones" | 1 ratio | 2 change in 2021-04-19..2026-09-30? | 3 HK listing | 4 ticker/domicile | 5 still dual-listed | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Bilibili | BILI | 9626.HK | 1 | New program 2018-04-02 | ✓ 1 | none listed | from data | none | active | pass → liquidity rule |
| Autohome | ATHM | 2518.HK | 4 | Ratio 1→4 effective 2021-02-05 | ✓ 4 | before START: OK | from data | none | active | pass → liquidity rule |
| XPeng | XPEV | 9868.HK | 2 | — | ✓ 2 | none listed | from data | none | active | pass → liquidity rule |
| LiAuto | LI | 2015.HK | 2 | HK dual listing 2021-08-12 | ✓ 2 | none listed | 2021-08-12 | none | active | pass → liquidity rule |
| Hutchmed | HCM | 0013.HK | **5** (Citi page showed 15 in error) | Ratio 2 DR:1 ORD → 1 DR:5 ORD on 2019-05-30 (before START); name change 2021-05-14. Depositary: **Deutsche Bank** | ✓ 5 — company FAQ: *"each ADR represents five ordinary shares"* (hutch-med.com/shareholder-information/investor-faqs, checked 2026-10-10) | none in sample: market-implied 4.9–5.1 every quarter 2021–2026; no Yahoo split | 2021-06-30 | name change only | active | pass → liquidity rule |
| Weibo | WB | 9898.HK | 1 | — | ✓ 1 | none listed | from data | none | active | pass → liquidity rule |
| NIO | NIO | 9866.HK | 1 | New program 2018-09-14 | ✓ 1 | none listed | from data | none | active | pass → liquidity rule |
| KEHoldings | BEKE | 2423.HK | 3 | — | ✓ 3 | none listed | from data | none | active | pass → liquidity rule |
| TencentMusic | TME | 1698.HK | 2 | New program 2018-12-14 | ✓ 2 | none listed | from data | none | active | pass → liquidity rule |
| ZTO | ZTO | 2057.HK | 1 | — | ✓ 1 | none listed | from data | none | active | pass → liquidity rule |
| HWorld | HTHT | 1179.HK | 10 | Ratio 1→10 effective **2021-06-29** | ✓ 10 | **yes** | — | — | active | **exclude (ratio change in sample)** |
| GDS | GDS | 9698.HK | 8 | HK secondary listing 2020-11-02 | ✓ 8 | none listed | 2020-11-02 | none | active | pass → liquidity rule |
| ZaiLab | ZLAB | 9688.HK | 10 | Ratio change effective **2022-03-30** (old 1:1); ORD ISIN changed twice | ✓ 10 | **yes** | — | — | active | **exclude (ratio change in sample)** |
| NewOriental | EDU | 9901.HK | Citi says 4; **market-implied 10** | none listed. Depositary: **DB** | ⚠ conflict | **yes**: Yahoo split records 2022-04-07/08 (ADR ratio → 1:10); 10-for-1 share split 2021-03 | 2020-11-09 | none | active | **exclude (ratio change in sample)** |
| BeiGene | ONC (was BGNE) | 6160.HK | 13 | Ticker BGNE→ONC **2025-01-02**; country Switzerland | ✓ 13 | — | — | **yes** | active | **exclude (ticker + domicile change)** |

Source for every row: Citi DR programme details (`depositaryreceipts.citi.com/adr/guides/pgm_d.aspx?...&cusip=<CUSIP>`),
saved in `results/logs/holdout_citi_programs.csv` (fetched 2026-10-10). Ratios first read by the student on the
Citi DR pages; Claude matched all 15 against the same source. "from data" = the first HK trading date comes
from the yfinance snapshot in Session 6b. "Active" = Citi shows no inactive date.

## After you finish

1. Update `config.HOLDOUT_CANDIDATES` ratios and `HOLDOUT_RATIO_VERIFIED`.
2. Fill in the Verdict column (keep / exclude + reason).
3. Commit. Session 6b then applies the liquidity rule and the implied-ratio check from yfinance data and logs every decision to `results/logs/holdout_selection.csv`.
