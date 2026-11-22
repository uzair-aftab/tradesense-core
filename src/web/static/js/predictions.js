/* TradeSense v8 — Prediction Markets (Wealthsimple Predict / Kalshi / FinBERT) Controller */

let predInitialized = false;
let predChart = null;
let predPriceSeries = null;
let predFairLine = null;
let predCurrentCategory = 'ALL';
let predCurrentMarkets = [];
let predSelectedMarket = null;
let predSelectedSide = 'YES';
let predCurrentFx = 1.39;
let predAutoPilotEnabled = true;

window.currentPredictionData = null;

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function predCalcFeeUsd(contracts, priceUsd) {
  const p = Math.min(Math.max(Number(priceUsd) || 0.5, 0.01), 0.99);
  const c = Math.max(Number(contracts) || 0, 0);
  if (c <= 0) return 0;
  const raw = 0.07 * c * p * (1.0 - p);
  return Math.ceil(raw * 100.0 - 1e-12) / 100.0;
}

function initPredictChart() {
  const container = document.getElementById('predict-chart-container');
  if (!container || predChart) return;
  if (typeof LightweightCharts === 'undefined') return;

  predChart = LightweightCharts.createChart(container, {
    width: container.clientWidth || 500,
    height: 230,
    layout: {
      background: { type: 'solid', color: '#020617' },
      textColor: '#94a3b8',
      fontSize: 11,
      fontFamily: 'JetBrains Mono, monospace',
    },
    grid: {
      vertLines: { color: '#0f172a' },
      horzLines: { color: '#0f172a' },
    },
    rightPriceScale: {
      borderColor: '#1e293b',
      scaleMargins: { top: 0.12, bottom: 0.12 },
    },
    timeScale: {
      borderColor: '#1e293b',
      timeVisible: true,
    },
  });

  predPriceSeries = predChart.addAreaSeries({
    lineColor: '#e2e8f0',
    topColor: 'rgba(56, 189, 248, 0.22)',
    bottomColor: 'rgba(56, 189, 248, 0.02)',
    lineWidth: 2,
    priceFormat: { type: 'custom', formatter: val => `${val.toFixed(1)}c` },
  });

  window.addEventListener('resize', () => {
    if (predChart && container.clientWidth > 0) {
      predChart.applyOptions({ width: container.clientWidth });
    }
  });
}

function predChartZoomIn() {
  if (!predChart) return;
  const range = predChart.timeScale().getVisibleLogicalRange();
  if (!range) return;
  const span = range.to - range.from;
  if (span <= 6) return;
  const shrink = span * 0.2;
  predChart.timeScale().setVisibleLogicalRange({ from: range.from + shrink, to: range.to });
}

function predChartZoomOut() {
  if (!predChart) return;
  const range = predChart.timeScale().getVisibleLogicalRange();
  if (!range) return;
  const span = range.to - range.from;
  predChart.timeScale().setVisibleLogicalRange({ from: range.from - span * 0.25, to: range.to });
}

function predChartReset() {
  if (predChart) predChart.timeScale().fitContent();
}

async function initPredictionsTab() {
  initPredictChart();
  setTimeout(() => {
    const c = document.getElementById('predict-chart-container');
    if (predChart && c && c.clientWidth > 0) {
      predChart.applyOptions({ width: c.clientWidth });
      predChart.timeScale().fitContent();
    }
  }, 60);
  if (!predInitialized) {
    predInitialized = true;
    document.querySelectorAll('.pred-cat-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.pred-cat-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        predCurrentCategory = btn.dataset.cat || 'ALL';
        loadPredictionMarkets(false);
      });
    });
    const searchEl = document.getElementById('pred-search-input');
    if (searchEl) {
      let timer = null;
      searchEl.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => loadPredictionMarkets(false), 260);
      });
    }
  }

  await Promise.all([
    loadMacroCalendar(false),
    loadPredictionPortfolio(),
    loadPredictionMarkets(false),
  ]);
}

window.initPredictionsTab = initPredictionsTab;

async function loadMacroCalendar(force = false) {
  try {
    const resp = await fetch(`/api/predictions/macro-calendar?force=${force ? 'true' : 'false'}`);
    const data = await resp.json();

    const fb = data.finbert_status || {};
    const beEl = document.getElementById('pred-finbert-backend');
    if (beEl) beEl.innerText = fb.backend && fb.backend.includes('vader') ? 'Lexicon Fallback' : 'FinBERT CUDA GPU';
    const devEl = document.getElementById('pred-finbert-device');
    if (devEl) devEl.innerText = `${fb.device || 'GPU'} · ${fb.cache_size || 0} cached headlines`;
    const fhBadge = document.getElementById('pred-finnhub-badge');
    if (fhBadge) {
      fhBadge.innerText = data.finnhub_active ? 'Finnhub Live' : 'RSS Wire';
      fhBadge.className = data.finnhub_active
        ? 'text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800'
        : 'text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700';
    }

    const items = data.calendar || [];
    const firstMacro = items.find(i => i.kind === 'MACRO_EVENT') || items[0];
    if (firstMacro) {
      document.getElementById('pred-next-catalyst-title').innerText = firstMacro.title;
      document.getElementById('pred-next-catalyst-countdown').innerText = `${firstMacro.date_label} (${firstMacro.days_away}d)`;
      document.getElementById('pred-next-catalyst-lead').innerText = firstMacro.lead_prob_pct !== null
        ? `${firstMacro.lead_outcome} @ ${firstMacro.lead_prob_pct}%`
        : firstMacro.lead_outcome;
    }

    const strip = document.getElementById('pred-macro-calendar-strip');
    if (strip) {
      strip.innerHTML = '';
      items.slice(0, 8).forEach(item => {
        const card = document.createElement('div');
        const clickable = Boolean(item.event_ticker);
        card.className = `shrink-0 min-w-[205px] max-w-[230px] p-2.5 rounded-lg bg-slate-950 border border-slate-800 ${clickable ? 'hover:border-sky-500 cursor-pointer' : 'opacity-80'} transition`;
        const tagClass = item.kind === 'EARNINGS' ? 'text-amber-300 bg-amber-950/60' : 'text-sky-300 bg-sky-950/60';
        card.innerHTML = `
          <div class="flex items-center justify-between text-[10px] font-mono mb-1">
            <span class="px-1.5 py-0.5 rounded ${tagClass} font-semibold">${escHtml(item.series_ticker)}</span>
            <span class="text-slate-400">${escHtml(item.date_label)} · ${item.days_away}d</span>
          </div>
          <div class="text-xs font-semibold text-white truncate">${escHtml(item.title)}</div>
          <div class="text-[11px] font-mono text-slate-400 mt-1 truncate">
            ${escHtml(item.lead_outcome)}${item.lead_prob_pct !== null ? ` · <strong class="text-emerald-400">${item.lead_prob_pct}%</strong>` : ''}
          </div>
        `;
        if (clickable) {
          card.addEventListener('click', () => loadEventDetail(item.event_ticker, null));
        }
        strip.appendChild(card);
      });
    }
  } catch (err) {
    console.warn('Macro calendar load warning:', err);
  }
}

async function loadPredictionMarkets(force = false) {
  const tbody = document.getElementById('pred-scanner-body');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="4" class="py-5 text-center text-slate-500">Scanning Kalshi &amp; running FinBERT GPU scoring...</td></tr>';
  }

  try {
    const wsOnly = document.getElementById('pred-ws-only')?.checked ?? true;
    const q = document.getElementById('pred-search-input')?.value?.trim() || '';
    const params = new URLSearchParams({
      category: predCurrentCategory,
      ws_only: wsOnly ? 'true' : 'false',
      q,
      force: force ? 'true' : 'false',
    });

    const resp = await fetch(`/api/predictions/markets?${params.toString()}`);
    const data = await resp.json();
    predCurrentFx = data.fx_usd_cad || 1.39;
    predCurrentMarkets = data.opportunities || [];

    renderPredictionScanner(predCurrentMarkets);

    if (predCurrentMarkets.length && (!predSelectedMarket || force)) {
      const top = predCurrentMarkets[0];
      await loadEventDetail(top.event_ticker, top.market_ticker);
    }
  } catch (err) {
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="4" class="py-5 text-center text-rose-400">Failed to load Kalshi prediction markets.</td></tr>';
    }
  }
}

function renderPredictionScanner(opps) {
  const tbody = document.getElementById('pred-scanner-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!opps.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="py-5 text-center text-slate-500">No contracts matched filters</td></tr>';
    return;
  }

  opps.forEach(o => {
    const tr = document.createElement('tr');
    const isSel = predSelectedMarket && predSelectedMarket.market_ticker === o.market_ticker;
    tr.className = `cursor-pointer transition ${isSel ? 'bg-sky-950/40' : 'hover:bg-slate-900/60'}`;

    const evColor = o.best_ev_pct >= 4.0 ? 'text-emerald-400 font-bold' : (o.best_ev_pct >= 1.5 ? 'text-sky-400 font-semibold' : 'text-slate-400');
    const sideBadge = o.recommended_side === 'YES'
      ? '<span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 text-[10px] font-bold">BUY YES</span>'
      : (o.recommended_side === 'NO'
        ? '<span class="px-1.5 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800 text-[10px] font-bold">BUY NO</span>'
        : '<span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]">PASS</span>');

    tr.innerHTML = `
      <td class="py-2 pr-2">
        <div class="text-white font-semibold font-sans text-xs leading-snug">${escHtml(o.outcome_label)}</div>
        <div class="text-[10px] text-slate-500 font-sans truncate max-w-[180px]">${escHtml(o.event_title)}</div>
      </td>
      <td class="py-2 text-right whitespace-nowrap">
        <div class="text-slate-200">${o.yes_cents.toFixed(0)}c</div>
        <div class="text-[10px] text-sky-400">${o.ai_fair_pct.toFixed(1)}%</div>
      </td>
      <td class="py-2 text-right ${evColor} whitespace-nowrap">${o.best_ev_pct >= 0 ? '+' : ''}${o.best_ev_pct.toFixed(1)}%</td>
      <td class="py-2 text-right whitespace-nowrap">${sideBadge}</td>
    `;
    tr.addEventListener('click', () => loadEventDetail(o.event_ticker, o.market_ticker));
    tbody.appendChild(tr);
  });
}

async function loadEventDetail(eventTicker, marketTicker = null) {
  if (!eventTicker) return;
  try {
    const params = new URLSearchParams({ lookback: '14' });
    if (marketTicker) params.set('market_ticker', marketTicker);

    const resp = await fetch(`/api/predictions/event/${encodeURIComponent(eventTicker)}?${params.toString()}`);
    if (!resp.ok) return;
    const data = await resp.json();
    window.currentPredictionData = data;
    predCurrentFx = data.fx_usd_cad || 1.39;

    const ev = data.event;
    const sel = data.selected_market || (data.outcomes && data.outcomes[0]);
    predSelectedMarket = sel;

    // Sync orderbook-refined probabilities & EV back into Scanner table so left & right panes agree 100%
    if (sel && Array.isArray(predCurrentMarkets)) {
      const mIdx = predCurrentMarkets.findIndex(x => x.market_ticker === sel.market_ticker);
      if (mIdx >= 0) {
        predCurrentMarkets[mIdx] = {
          ...predCurrentMarkets[mIdx],
          ai_fair_prob: sel.ai_fair_prob,
          ai_fair_pct: sel.ai_fair_pct,
          best_ev_pct: sel.best_ev_pct,
          recommended_side: sel.recommended_side,
          yes_cents: sel.yes_cents,
          no_cents: sel.no_cents,
        };
      }
    }

    document.getElementById('pred-ev-series-badge').innerText = ev.event_ticker;
    document.getElementById('pred-ev-title').innerText = ev.title;
    document.getElementById('pred-ev-subtitle').innerText = `${ev.category} · Overround Sum: ${(ev.overround_sum * 100).toFixed(1)}%`;

    if (sel) {
      document.getElementById('pred-ev-close-badge').innerText = `Closes in ${sel.days_to_close}d (${sel.contract_term_days}d term)`;
      const ciroBadge = document.getElementById('pred-ev-ciro-badge');
      if (ciroBadge) {
        ciroBadge.innerText = sel.ws_predict_eligible ? 'WS Predict Eligible' : 'Kalshi Macro Reference';
        ciroBadge.className = sel.ws_predict_eligible
          ? 'text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-semibold'
          : 'text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-semibold';
      }
    }

    const fredPill = document.getElementById('pred-fred-pill');
    if (fredPill) {
      if (data.fred_snapshot && data.fred_snapshot.value !== null) {
        fredPill.innerText = `FRED ${data.fred_snapshot.series_id}: ${data.fred_snapshot.value}`;
        fredPill.classList.remove('hidden');
      } else {
        fredPill.classList.add('hidden');
      }
    }

    // Multi-Outcome Bars
    const outList = document.getElementById('pred-outcomes-list');
    if (outList) {
      outList.innerHTML = '';
      (data.outcomes || []).forEach(o => {
        const isCurrent = sel && o.market_ticker === sel.market_ticker;
        const row = document.createElement('div');
        row.className = `p-2 rounded-lg border cursor-pointer transition ${isCurrent ? 'bg-slate-800/80 border-sky-500' : 'bg-slate-950 border-slate-800 hover:border-slate-700'}`;
        const mPct = Math.max(0, Math.min(100, o.market_prob_pct));
        const aPct = Math.max(0, Math.min(100, o.ai_fair_pct));
        const diff = o.edge_pts;
        row.innerHTML = `
          <div class="flex items-center justify-between text-xs mb-1">
            <span class="font-semibold text-white truncate pr-2">${escHtml(o.outcome_label)}</span>
            <div class="flex items-center gap-2 font-mono text-[11px] shrink-0">
              <span class="text-slate-300">Mkt ${mPct.toFixed(1)}%</span>
              <span class="text-sky-400 font-bold">AI ${aPct.toFixed(1)}%</span>
              <span class="${diff >= 1.5 ? 'text-emerald-400' : (diff <= -1.5 ? 'text-rose-400' : 'text-slate-500')}">(${diff >= 0 ? '+' : ''}${diff.toFixed(1)})</span>
            </div>
          </div>
          <div class="space-y-1">
            <div class="h-1.5 w-full rounded bg-slate-900 overflow-hidden">
              <div class="h-full bg-slate-500" style="width: ${mPct}%"></div>
            </div>
            <div class="h-1.5 w-full rounded bg-slate-900 overflow-hidden">
              <div class="h-full bg-sky-400" style="width: ${aPct}%"></div>
            </div>
          </div>
        `;
        row.addEventListener('click', () => loadEventDetail(ev.event_ticker, o.market_ticker));
        outList.appendChild(row);
      });
    }

    // Implied Probability History Chart
    initPredictChart();
    if (predPriceSeries) {
      let candles = data.candlesticks || [];
      if (!candles.length && sel) {
        const nowSec = Math.floor(Date.now() / 1000);
        const p0 = (sel.prev_price > 0 ? sel.prev_price : sel.mid_prob) * 100.0;
        const p1 = sel.mid_prob * 100.0;
        candles = [
          { time: nowSec - 86400, value: Number(p0.toFixed(1)) },
          { time: nowSec, value: Number(p1.toFixed(1)) },
        ];
      } else {
        candles = candles.map(c => ({ time: c.time, value: c.close }));
      }
      predPriceSeries.setData(candles);
      if (predFairLine) {
        predPriceSeries.removePriceLine(predFairLine);
        predFairLine = null;
      }
      if (sel) {
        predFairLine = predPriceSeries.createPriceLine({
          price: sel.ai_fair_pct,
          color: '#38bdf8',
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'AI Fair %',
        });
      }
      predChart.timeScale().fitContent();
    }

    // Orderbook Depth Skew
    const ob = data.orderbook || {};
    const yDepth = ob.yes_depth_usd || 0;
    const nDepth = ob.no_depth_usd || 0;
    const tot = Math.max(yDepth + nDepth, 1);
    const yPct = Math.round((yDepth / tot) * 100);
    document.getElementById('pred-ob-yes').innerText = `$${yDepth.toLocaleString()}`;
    document.getElementById('pred-ob-no').innerText = `$${nDepth.toLocaleString()}`;
    document.getElementById('pred-ob-yes-bar').style.width = `${yPct}%`;
    document.getElementById('pred-ob-no-bar').style.width = `${100 - yPct}%`;
    document.getElementById('pred-ob-spread').innerText = ob.spread_cents ? `${ob.spread_cents}c` : 'Tight';

    // FinBERT News Wire
    const fbSum = data.finbert_summary || {};
    const shiftEl = document.getElementById('pred-news-shift-badge');
    if (shiftEl) {
      const sh = Number(fbSum.directional_shift || 0);
      shiftEl.innerText = `FinBERT Shift: ${sh >= 0 ? '+' : ''}${sh.toFixed(2)} (${fbSum.backend || 'finbert'})`;
      shiftEl.className = `text-[11px] font-mono px-2 py-0.5 rounded ${sh > 0.1 ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : (sh < -0.1 ? 'bg-rose-950 text-rose-300 border border-rose-800' : 'bg-slate-800 text-slate-300')}`;
    }

    const newsEl = document.getElementById('pred-news-list');
    if (newsEl) {
      newsEl.innerHTML = '';
      (data.news || []).forEach(n => {
        const div = document.createElement('div');
        div.className = 'p-2.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1';
        const dBadge = n.directional_shift >= 0.1
          ? `<span class="text-emerald-400 font-bold">+${n.directional_shift.toFixed(2)} YES</span>`
          : (n.directional_shift <= -0.1
            ? `<span class="text-rose-400 font-bold">${n.directional_shift.toFixed(2)} NO</span>`
            : `<span class="text-slate-400">0.00 Neutral</span>`);
        div.innerHTML = `
          <div class="flex items-center justify-between text-[10px] text-slate-400">
            <span>${escHtml(n.publisher)} · ${escHtml(n.published_ago)}</span>
            <span class="font-mono">${dBadge}</span>
          </div>
          <a href="${escHtml(n.url)}" target="_blank" rel="noopener" class="text-slate-200 hover:text-sky-400 font-medium block leading-snug">${escHtml(n.title)}</a>
          <div class="flex items-center gap-2 text-[10px] font-mono text-slate-500">
            <span class="text-emerald-400">POS ${n.finbert_pos}%</span>
            <span class="text-rose-400">NEG ${n.finbert_neg}%</span>
            <span class="text-slate-400">NEU ${n.finbert_neu}%</span>
            <span class="ml-auto text-sky-400">${escHtml(n.catalyst)}</span>
          </div>
        `;
        newsEl.appendChild(div);
      });
    }

    // Update Execution Ticket
    if (sel) {
      document.getElementById('pred-ticket-outcome').innerText = `${sel.outcome_label}`;
      document.getElementById('pred-ticket-ticker').innerText = `${sel.market_ticker}`;
      document.getElementById('pred-btn-yes-c').innerText = `${sel.yes_cents.toFixed(1)}c`;
      document.getElementById('pred-btn-no-c').innerText = `${sel.no_cents.toFixed(1)}c`;

      const reasonsEl = document.getElementById('pred-ticket-reasons');
      if (reasonsEl) {
        reasonsEl.innerHTML = (sel.reasoning_bullets || []).map(r => `<div>• ${escHtml(r)}</div>`).join('');
      }

      const kellyBtn = document.getElementById('btn-pred-kelly');
      if (kellyBtn) {
        const rec = sel.recommended_stake_usd > 0 ? sel.recommended_stake_usd : 5.0;
        kellyBtn.innerText = `Kelly ($${rec.toFixed(2)})`;
      }

      selectPredictSide(sel.recommended_side === 'NO' ? 'NO' : 'YES');
    }

    renderPredictionScanner(predCurrentMarkets);
  } catch (err) {
    console.warn('Event detail load error:', err);
  }
}

function selectPredictSide(side) {
  predSelectedSide = side === 'NO' ? 'NO' : 'YES';
  const yBtn = document.getElementById('btn-pred-side-yes');
  const nBtn = document.getElementById('btn-pred-side-no');

  if (yBtn && nBtn) {
    if (predSelectedSide === 'YES') {
      yBtn.className = 'py-2 px-3 rounded-lg bg-emerald-600/20 border border-emerald-500 text-emerald-300 font-bold text-xs transition';
      nBtn.className = 'py-2 px-3 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 font-bold text-xs transition hover:border-rose-500';
    } else {
      nBtn.className = 'py-2 px-3 rounded-lg bg-rose-600/20 border border-rose-500 text-rose-300 font-bold text-xs transition';
      yBtn.className = 'py-2 px-3 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 font-bold text-xs transition hover:border-emerald-500';
    }
  }
  recalcPredictTicket();
}

function setPredictStake(val) {
  const inp = document.getElementById('pred-stake-usd');
  if (inp) inp.value = Math.max(1, Number(val) || 5);
  document.querySelectorAll('.pred-stake-preset').forEach(b => {
    b.classList.toggle('active', b.innerText === `$${inp.value}`);
  });
  recalcPredictTicket();
}

function applyPredictKelly() {
  const rec = (predSelectedMarket && predSelectedMarket.recommended_stake_usd > 0)
    ? predSelectedMarket.recommended_stake_usd
    : 5.0;
  setPredictStake(Math.max(1, Math.round(rec * 100) / 100));
}

function recalcPredictTicket() {
  const stake = Math.max(1.0, parseFloat(document.getElementById('pred-stake-usd')?.value || '5') || 5);
  const btnBuy = document.getElementById('btn-pred-paper-buy');
  if (btnBuy) btnBuy.innerText = `Paper Trade Virtual ($${stake.toFixed(0)} USD)`;

  if (!predSelectedMarket) return;
  const price = predSelectedSide === 'YES' ? (predSelectedMarket.yes_ask || 0.5) : (predSelectedMarket.no_ask || 0.5);
  const p = Math.min(Math.max(price, 0.01), 0.99);
  const fair = predSelectedSide === 'YES' ? (predSelectedMarket.ai_fair_prob || 0.5) : (1.0 - (predSelectedMarket.ai_fair_prob || 0.5));

  const contracts = stake / p;
  const fee = predCalcFeeUsd(contracts, p);
  const totalCost = stake + fee;
  const totalCad = totalCost * predCurrentFx;
  const netProfit = contracts - totalCost;
  const roiPct = totalCost > 0 ? (netProfit / totalCost) * 100.0 : 0;
  const breakeven = (totalCost / Math.max(contracts, 1e-6)) * 100.0;
  const netEvPct = totalCost > 0 ? ((fair * contracts - totalCost) / totalCost) * 100.0 : 0;

  document.getElementById('pred-calc-contracts').innerText = contracts.toFixed(2);
  document.getElementById('pred-calc-fee').innerText = `$${fee.toFixed(2)} USD`;
  document.getElementById('pred-calc-cost-usd').innerText = `$${totalCost.toFixed(2)} USD`;
  document.getElementById('pred-calc-cost-cad').innerText = `($${totalCad.toFixed(2)} CAD)`;
  document.getElementById('pred-calc-profit').innerText = `+$${netProfit.toFixed(2)} (+${roiPct.toFixed(0)}%)`;
  document.getElementById('pred-calc-breakeven').innerText = `${Math.min(99.9, breakeven).toFixed(1)}%`;

  const evEl = document.getElementById('pred-calc-ev');
  if (evEl) {
    evEl.innerText = `${netEvPct >= 0 ? '+' : ''}${netEvPct.toFixed(2)}%`;
    evEl.className = netEvPct >= 1.5 ? 'text-emerald-400 font-bold' : (netEvPct < 0 ? 'text-rose-400 font-bold' : 'text-sky-400 font-bold');
  }
}

async function submitPredictPaperBuy() {
  if (!predSelectedMarket) return showToast('Select a prediction market first');
  const stake = Math.max(1.0, parseFloat(document.getElementById('pred-stake-usd')?.value || '5') || 5);
  try {
    const resp = await fetch('/api/predictions/paper-buy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        market_ticker: predSelectedMarket.market_ticker,
        side: predSelectedSide,
        stake_usd: stake,
        ai_fair_prob: predSelectedMarket.ai_fair_prob,
      }),
    });
    const res = await resp.json();
    showToast(res.message || 'Paper prediction executed');
    await loadPredictionPortfolio();
  } catch (err) {
    showToast('Failed to place virtual prediction trade');
  }
}

async function closePredictPaperPos(positionId) {
  try {
    const resp = await fetch('/api/predictions/paper-close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position_id: positionId }),
    });
    const res = await resp.json();
    showToast(res.message || 'Prediction contract exited');
    await loadPredictionPortfolio();
  } catch (err) {
    showToast('Failed to exit prediction contract');
  }
}

async function togglePredictAutoPilot() {
  const nextState = !predAutoPilotEnabled;
  try {
    const resp = await fetch('/api/predictions/auto-pilot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: nextState, max_stake_usd: 5.0 }),
    });
    const res = await resp.json();
    predAutoPilotEnabled = Boolean(res.auto_pilot);
    updateAutoPilotButton();
    showToast(`Predict Auto-Pilot ${predAutoPilotEnabled ? 'Armed ($5 @ >=4% EV)' : 'Paused'}`);
  } catch (err) {
    showToast('Failed to update Auto-Pilot');
  }
}

function updateAutoPilotButton() {
  const btn = document.getElementById('btn-pred-autopilot');
  if (!btn) return;
  if (predAutoPilotEnabled) {
    btn.innerText = 'Auto-Pilot: ON';
    btn.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800';
  } else {
    btn.innerText = 'Auto-Pilot: OFF';
    btn.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700';
  }
}

async function resetPredictSandbox() {
  if (!confirm('Reset your Predict Sandbox to $500.00 USD and clear virtual prediction history?')) return;
  try {
    await fetch('/api/predictions/reset-sandbox', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ starting_usd: 500.0 }),
    });
    showToast('Predict Sandbox reset to $500.00 USD');
    await loadPredictionPortfolio();
  } catch (err) {
    showToast('Reset failed');
  }
}

function copyPredictTicket() {
  if (!predSelectedMarket) return showToast('Select a market first');
  const stake = Math.max(1.0, parseFloat(document.getElementById('pred-stake-usd')?.value || '5') || 5);
  const priceCents = predSelectedSide === 'YES' ? predSelectedMarket.yes_cents : predSelectedMarket.no_cents;
  const text = [
    'Wealthsimple Predict Order Ticket',
    `Event: ${predSelectedMarket.event_title}`,
    `Outcome: ${predSelectedMarket.outcome_label}`,
    `Action: BUY ${predSelectedSide} @ ${priceCents.toFixed(1)}c`,
    `Stake: $${stake.toFixed(2)} USD (~$${(stake * predCurrentFx).toFixed(2)} CAD)`,
    `AI Fair Prob: ${predSelectedMarket.ai_fair_pct.toFixed(1)}% (Mkt ${predSelectedMarket.market_prob_pct.toFixed(1)}%)`,
    `Net EV after fee: ${predSelectedMarket.best_ev_pct >= 0 ? '+' : ''}${predSelectedMarket.best_ev_pct.toFixed(2)}%`,
  ].join('\n');

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => showToast('Copied ticket for Wealthsimple Predict app!'));
  } else {
    showToast('Ready for Wealthsimple Predict entry');
  }
}

async function submitRealPredictFill() {
  if (!predSelectedMarket) return showToast('Select a contract first');
  const stake = Math.max(1.0, parseFloat(document.getElementById('pred-stake-usd')?.value || '5') || 5);
  const price = predSelectedSide === 'YES' ? (predSelectedMarket.yes_ask || 0.5) : (predSelectedMarket.no_ask || 0.5);
  try {
    const resp = await fetch('/api/predictions/real-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_ticker: predSelectedMarket.event_ticker,
        market_ticker: predSelectedMarket.market_ticker,
        event_title: predSelectedMarket.event_title,
        outcome_label: predSelectedMarket.outcome_label,
        side: predSelectedSide,
        entry_price_usd: Math.min(Math.max(price, 0.01), 0.99),
        stake_usd: stake,
        notes: 'Logged from WS Predict Ticket',
      }),
    });
    const res = await resp.json();
    if (res.status === 'SUCCESS') {
      showToast(`Real WS Predict fill logged (${res.tx_id})`);
      await loadPredictionPortfolio();
    }
  } catch (err) {
    showToast('Failed to log real Predict fill');
  }
}

async function loadPredictionPortfolio() {
  try {
    const resp = await fetch('/api/predictions/portfolio');
    const data = await resp.json();
    const acct = data.account || {};
    const cal = data.calibration || {};

    predAutoPilotEnabled = Boolean(acct.auto_pilot);
    updateAutoPilotButton();

    document.getElementById('pred-sandbox-equity').innerText = `$${Number(acct.total_equity_usd || 500).toFixed(2)} USD`;
    document.getElementById('pred-sandbox-cash').innerText = `$${Number(acct.cash_usd || 500).toFixed(2)}`;
    const retPct = Number(acct.net_return_pct || 0);
    const retEl = document.getElementById('pred-sandbox-ret');
    retEl.innerText = `${retPct >= 0 ? '+' : ''}${retPct.toFixed(1)}%`;
    retEl.className = retPct >= 0 ? 'text-emerald-400' : 'text-rose-400';

    const brierEl = document.getElementById('pred-brier-ai');
    if (brierEl) {
      brierEl.innerText = cal.avg_brier_ai !== null && cal.avg_brier_ai !== undefined
        ? `${cal.avg_brier_ai.toFixed(4)} vs ${cal.avg_brier_market.toFixed(4)}`
        : 'Calibrating (Pending Resolve)';
    }
    document.getElementById('pred-win-rate').innerText = `${Number(cal.win_rate_pct || 0).toFixed(1)}%`;
    document.getElementById('pred-settled-count').innerText = `${cal.total_trades || 0} Exits (${cal.settled_count || 0} Resolved)`;

    // Open Positions Table
    const openPos = data.open_positions || [];
    document.getElementById('pred-open-count-badge').innerText = `${openPos.length} Open`;
    const openBody = document.getElementById('pred-open-pos-body');
    openBody.innerHTML = '';
    if (!openPos.length) {
      openBody.innerHTML = '<tr><td colspan="8" class="py-4 text-center text-slate-500">No open prediction contracts yet. Place a $1–$50 virtual bet above!</td></tr>';
    } else {
      openPos.forEach(p => {
        const tr = document.createElement('tr');
        const pnl = Number(p.unrealized_pnl_usd || 0);
        const sideCls = p.side === 'YES' ? 'text-emerald-400' : 'text-rose-400';
        tr.innerHTML = `
          <td class="py-2">
            <span class="font-bold text-white block font-sans">${escHtml(p.outcome_label)}</span>
            <span class="text-[10px] text-slate-500 font-sans">${escHtml(p.event_title)}</span>
          </td>
          <td class="py-2 font-bold ${sideCls}">${p.side}</td>
          <td class="py-2 text-right text-slate-200">${Number(p.contracts).toFixed(2)}</td>
          <td class="py-2 text-right text-slate-300">${(p.entry_price_usd * 100).toFixed(1)}c</td>
          <td class="py-2 text-right text-white font-bold">${(p.current_price_usd * 100).toFixed(1)}c</td>
          <td class="py-2 text-right text-slate-300">$${Number(p.stake_usd).toFixed(2)}</td>
          <td class="py-2 text-right font-bold ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${p.unrealized_pnl_pct.toFixed(1)}%)</td>
          <td class="py-2 text-center">
            <button onclick="closePredictPaperPos(${p.id})" class="px-2 py-0.5 rounded bg-rose-950 hover:bg-rose-900 border border-rose-800 text-rose-300 text-xs">Early Exit</button>
          </td>
        `;
        openBody.appendChild(tr);
      });
    }

    // Settled Ledger
    const settledBody = document.getElementById('pred-settled-body');
    const trades = data.settled_trades || [];
    settledBody.innerHTML = '';
    if (!trades.length) {
      settledBody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500">No settled or exited contracts yet</td></tr>';
    } else {
      trades.forEach(t => {
        const tr = document.createElement('tr');
        const net = Number(t.net_pnl_usd || 0);
        const bTxt = t.brier_ai !== null && t.brier_ai !== undefined ? `${Number(t.brier_ai).toFixed(3)} / ${Number(t.brier_market).toFixed(3)}` : 'Early';
        tr.innerHTML = `
          <td class="py-2 font-sans text-white truncate max-w-[140px]">${escHtml(t.outcome_label)}</td>
          <td class="py-2 ${t.side === 'YES' ? 'text-emerald-400' : 'text-rose-400'} font-bold">${t.side}</td>
          <td class="py-2 text-right text-slate-300">${(t.entry_price_usd * 100).toFixed(0)}c / ${(t.exit_price_usd * 100).toFixed(0)}c</td>
          <td class="py-2 text-right font-bold ${net >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${net >= 0 ? '+' : ''}$${net.toFixed(2)}</td>
          <td class="py-2 text-right text-slate-400">${bTxt}</td>
          <td class="py-2 text-[10px] text-slate-500">${escHtml(t.exit_reason)}</td>
        `;
        settledBody.appendChild(tr);
      });
    }

    // Real WS Fills
    const realBody = document.getElementById('pred-real-fills-body');
    const realFills = data.real_fills || [];
    realBody.innerHTML = '';
    if (!realFills.length) {
      realBody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500">No real Wealthsimple Predict fills logged yet</td></tr>';
    } else {
      realFills.forEach(r => {
        const tr = document.createElement('tr');
        const pnl = Number(r.realized_pnl_usd || 0);
        tr.innerHTML = `
          <td class="py-2 text-[11px] text-slate-400">${escHtml(r.tx_id)}</td>
          <td class="py-2 font-sans text-white truncate max-w-[130px]">${escHtml(r.outcome_label)}</td>
          <td class="py-2 ${r.side === 'YES' ? 'text-emerald-400' : 'text-rose-400'} font-bold">${r.side}</td>
          <td class="py-2 text-right text-slate-300">$${Number(r.stake_usd).toFixed(2)}</td>
          <td class="py-2"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px]">${escHtml(r.status)}</span></td>
          <td class="py-2 text-right ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${r.status === 'CLOSED' ? (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) : '—'}</td>
        `;
        realBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.warn('Prediction portfolio load error:', err);
  }
}

window.loadPredictionPortfolioSummary = function (board) {
  const tbody = document.getElementById('portfolios-predict-lb-body');
  if (!tbody) return;
  tbody.innerHTML = '';
  (board || []).forEach(l => {
    const tr = document.createElement('tr');
    const rc = l.net_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400';
    tr.innerHTML = `
      <td class="py-2 font-bold text-white">${escHtml(l.display_name)}</td>
      <td class="py-2 text-right text-slate-400">$${Number(l.starting_usd).toFixed(2)}</td>
      <td class="py-2 text-right font-bold text-white">$${Number(l.total_equity_usd).toFixed(2)}</td>
      <td class="py-2 text-right font-bold ${rc}">${l.net_return_pct >= 0 ? '+' : ''}${Number(l.net_return_pct).toFixed(2)}%</td>
      <td class="py-2 text-center text-sky-400 font-bold">${l.open_count}</td>
      <td class="py-2 text-center text-slate-300">${l.trades_count}</td>
      <td class="py-2 text-right text-slate-400">${l.avg_brier_ai !== null && l.avg_brier_ai !== undefined ? Number(l.avg_brier_ai).toFixed(4) : '—'}</td>
    `;
    tbody.appendChild(tr);
  });
};
