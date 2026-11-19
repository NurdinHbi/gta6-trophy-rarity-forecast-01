# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/NurdinHbi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/NurdinHbi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate one week
after launch, as five quantiles (5/25/50/75/95%) per trophy, from the
house's recording of the trophy page up to the case checkpoint.

Method, all in logit space: per trophy, a least-squares line of logit(rate)
on ln(1 + hours since release), anchored at the last read; plus a separate
earners-vs-tracked split, because Exophase's rate is earners / players
tracked and the two move for different reasons — players progress through
the game while new players join and dilute every rate — so earners and the
tracked base are forecast on their own and the ratio gives a second rate
forecast, trusted only while the rate is low — its multiplicative growth
has no saturation, so for a common trophy the trend stands alone — and
never allowed far from the trend forecast. Slopes are shrunk toward
their tier's mean (platinum / gold / silver / bronze), so one trophy's noisy
early reads cannot drag its forecast away; with flat early data every slope
shrinks to ~0 and the forecast is honest persistence. The platinum median
is capped at the rarest other trophy's median, since earning every trophy
cannot be more common than earning the rarest one. The spread combines the
fit residual, the slope's standard error over the horizon, a random-walk
term growing with the square root of the horizon, and half the disagreement
between the two forecasts.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case. Python 3.13, standard library only, no
network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7drh8aw0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
