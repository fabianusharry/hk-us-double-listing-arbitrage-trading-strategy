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

| Name in config | US | HK (Yahoo) | Ratio in config | Claude's note (verify!) | 1 ratio | 2 change? | 3 HK listing | 4 ticker/domicile | 5 still dual-listed | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Bilibili | BILI | 9626.HK | 1 | HK Mar 2021 (secondary, later primary) | | | | | | |
| Autohome | ATHM | 2518.HK | 4 | HK Mar 2021 | | | | | | |
| XPeng | XPEV | 9868.HK | 2 | HK Jul 2021 (dual primary); borrow may be costly | | | | | | |
| LiAuto | LI | 2015.HK | 2 | HK Aug 2021 (dual primary); borrow may be costly | | | | | | |
| Hutchmed | HCM | 0013.HK | 5 | HK Jun 2021; also had an AIM listing | | | | | | |
| Weibo | WB | 9898.HK | 1 | HK Dec 2021 (secondary) | | | | | | |
| NIO | NIO | 9866.HK | 1 | HK Mar 2022 by introduction; also listed in Singapore | | | | | | |
| KEHoldings | BEKE | 2423.HK | 3 | HK May 2022 by introduction | | | | | | |
| TencentMusic | TME | 1698.HK | 2 | HK Sep 2022 by introduction | | | | | | |
| ZTO | ZTO | 2057.HK | 1 | HK Sep 2020; converted to primary later (status change only — not an exclusion) | | | | | | |
| HWorld | HTHT | 1179.HK | 10 | HK Sep 2020; ratio may have changed around 2020 — check the date is before 2021-04-19 | | | | | | |
| GDS | GDS | 9698.HK | 8 | HK Nov 2020 | | | | | | |
| ZaiLab | ZLAB | 9688.HK | 10 | HK Sep 2020; unsure whether the ratio changed | | | | | | |
| NewOriental | EDU | 9901.HK | 10 | HK Nov 2020; ratio change suspected around 2022 → likely excluded | | | | | | |
| BeiGene | BGNE (now ONC?) | 6160.HK | 13 | 2025 redomicile / rename to BeOne, ticker ONC → likely excluded | | | | | | |

## After you finish

1. Update `config.HOLDOUT_CANDIDATES` ratios and `HOLDOUT_RATIO_VERIFIED`.
2. Fill in the Verdict column (keep / exclude + reason).
3. Commit. Session 6b then applies the liquidity rule and the implied-ratio check from yfinance data and logs every decision to `results/logs/holdout_selection.csv`.
