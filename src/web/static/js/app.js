/**
 * TradeSense v7 — Client Application Logic
 * TradingView Lightweight Charts v4 Integration, 0ms Local Controls, Fast JSON APIs
 */

let chart = null;
let candleSeries = null;
let lineSeries = null;
let volumeSeries = null;
let ema9Series = null;
let ema21Series = null;
let vwapSeries = null;
let bbUpperSeries = null;
let bbLowerSeries = null;
let forecastSeries = null;

let currentWorkspaceData = null;
let rawBars = [];
let currentChartMode = 'candles';
let isChartInitialized = false;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Chart Initialization & 0ms Local Toolbar Controls
// ─────────────────────────────────────────────────────────────────────────────
function initChart() {
  const container = document.getElementById('chart-container');
  if (!container || isChartInitialized) return;

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 440;

  chart = LightweightCharts.createChart(container, {
    width: width,
    height: height,
    layout: {
      background: { type: 'solid', color: '#020617' },
      textColor: '#94a3b8',
      fontSize: 11,
      fontFamily: 'JetBrains Mono, monospace',
    },
    grid: {
      vertLines: { color: 'rgba(30, 41, 59, 0.5)' },
      horzLines: { color: 'rgba(30, 41, 59, 0.5)' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: '#38bdf8', width: 1, style: 3, labelBackgroundColor: '#0f172a' },
      horzLine: { color: '#38bdf8', width: 1, style: 3, labelBackgroundColor: '#0f172a' },
    },
    rightPriceScale: {
      borderColor: '#1e293b',
      scaleMargins: { top: 0.1, bottom: 0.2 },
    },
    timeScale: {
      borderColor: '#1e293b',
      timeVisible: true,
      secondsVisible: false,
    },
  });

  // Candle Series
  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981',
    downColor: '#f43f5e',
    borderVisible: false,
    wickUpColor: '#10b981',
    wickDownColor: '#f43f5e',
  });

  // Line Series (hidden by default)
  lineSeries = chart.addLineSeries({
    color: '#38bdf8',
    lineWidth: 2,
    visible: false,
  });

  // Volume Series
  volumeSeries = chart.addHistogramSeries({
    color: '#334155',
    priceFormat: { type: 'volume' },
    priceScaleId: '',
    scaleMargins: { top: 0.82, bottom: 0 },
  });

  // Overlays
  ema9Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, title: 'EMA 9' });
  ema21Series = chart.addLineSeries({ color: '#fbbf24', lineWidth: 1.5, title: 'EMA 21' });
  vwapSeries = chart.addLineSeries({ color: '#a855f7', lineWidth: 1.5, title: 'VWAP' });
  bbUpperSeries = chart.addLineSeries({ color: 'rgba(148, 163, 184, 0.4)', lineWidth: 1, lineStyle: 2, title: 'BB Upper' });
  bbLowerSeries = chart.addLineSeries({ color: 'rgba(148, 163, 184, 0.4)', lineWidth: 1, lineStyle: 2, title: 'BB Lower' });

  // Forecast Path Series
  forecastSeries = chart.addLineSeries({
    color: '#38bdf8',
    lineWidth: 2,
    lineStyle: 3,
    title: 'P50 Forecast Path',
  });

  // Crosshair move handler
  chart.subscribeCrosshairMove(param => {
    if (!param || !param.time || !param.seriesPrices) {
      return;
    }
    const ohlc = param.seriesPrices.get(candleSeries) || param.seriesPrices.get(lineSeries);
    const vol = param.seriesPrices.get(volumeSeries);
    if (ohlc) {
      document.getElementById('legend-o').innerText = ohlc.open !== undefined ? '$' + formatNumber(ohlc.open) : '—';
      document.getElementById('legend-h').innerText = ohlc.high !== undefined ? '$' + formatNumber(ohlc.high) : '—';
      document.getElementById('legend-l').innerText = ohlc.low !== undefined ? '$' + formatNumber(ohlc.low) : '—';
      document.getElementById('legend-c').innerText = ohlc.close !== undefined ? '$' + formatNumber(ohlc.close) : '$' + formatNumber(ohlc);
    }
    if (vol !== undefined) {
      document.getElementById('legend-v').innerText = formatCompact(vol);
    }
  });

  // Resize listener
  window.addEventListener('resize', () => {
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth });
    }
  });

  isChartInitialized = true;
}

// 0ms Client-Side Zoom Controls
function chartZoomIn() {
  if (!chart) return;
  const range = chart.timeScale().getVisibleLogicalRange();
  if (!range) return;
  const bars = range.to - range.from;
  const newBars = Math.max(10, bars * 0.7);
  const center = (range.from + range.to) / 2;
  chart.timeScale().setVisibleLogicalRange({
    from: center - newBars / 2,
    to: center + newBars / 2,
  });
}

function chartZoomOut() {
  if (!chart) return;
  const range = chart.timeScale().getVisibleLogicalRange();
  if (!range) return;
  const bars = range.to - range.from;
  const newBars = bars * 1.35;
  const center = (range.from + range.to) / 2;
  chart.timeScale().setVisibleLogicalRange({
    from: center - newBars / 2,
    to: center + newBars / 2,
  });
}

function chartResetZoom() {
  if (!chart) return;
  chart.timeScale().fitContent();
}

// 0ms Logical Range Windows
function setWindowBars(count) {
  if (!chart || !rawBars.length) return;
  if (count === 'all') {
    chart.timeScale().fitContent();
    return;
  }
  const n = parseInt(count, 10);
  const total = rawBars.length;
  chart.timeScale().setVisibleLogicalRange({
    from: Math.max(0, total - n),
    to: total,
  });
}

// 0ms Chart Style Switcher
function changeChartStyle(mode) {
  currentChartMode = mode;
  if (!chart || !candleSeries || !lineSeries) return;

  if (mode === 'line') {
    candleSeries.applyOptions({ visible: false });
    lineSeries.applyOptions({ visible: true });
    lineSeries.setData(rawBars.map(b => ({ time: b.time, value: b.close })));
  } else if (mode === 'heikin_ashi') {
    candleSeries.applyOptions({ visible: true });
    lineSeries.applyOptions({ visible: false });
    const haBars = computeHeikinAshi(rawBars);
    candleSeries.setData(haBars);
  } else {
    // Standard Candles
    candleSeries.applyOptions({ visible: true });
    lineSeries.applyOptions({ visible: false });
    candleSeries.setData(rawBars);
  }
}

function computeHeikinAshi(bars) {
  if (!bars.length) return [];
  const ha = [];
  let prevHaOpen = bars[0].open;
  let prevHaClose = bars[0].close;

  for (let i = 0; i < bars.length; i++) {
    const b = bars[i];
    const haClose = (b.open + b.high + b.low + b.close) / 4;
    const haOpen = i === 0 ? (b.open + b.close) / 2 : (prevHaOpen + prevHaClose) / 2;
    const haHigh = Math.max(b.high, haOpen, haClose);
    const haLow = Math.min(b.low, haOpen, haClose);

    ha.push({ time: b.time, open: haOpen, high: haHigh, low: haLow, close: haClose });
    prevHaOpen = haOpen;
    prevHaClose = haClose;
  }
  return ha;
}

// 0ms Overlay Visibility Toggles
function toggleOverlay(overlay, isVisible) {
  if (!chart) return;
  if (overlay === 'ema') {
    ema9Series.applyOptions({ visible: isVisible });
    ema21Series.applyOptions({ visible: isVisible });
  } else if (overlay === 'vwap') {
    vwapSeries.applyOptions({ visible: isVisible });
  } else if (overlay === 'bb') {
    bbUpperSeries.applyOptions({ visible: isVisible });
    bbLowerSeries.applyOptions({ visible: isVisible });
  } else if (overlay === 'cone') {
    forecastSeries.applyOptions({ visible: isVisible });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Workspace Data Fetching & UI State Synchronization
// ─────────────────────────────────────────────────────────────────────────────
async function loadWorkspace(forceRetrain = false) {
  try {
    initChart();
    const btn = document.getElementById('btn-refresh-all');
    if (btn) btn.classList.add('animate-spin');

    const params = new URLSearchParams({
      retrain: forceRetrain ? 'true' : 'false',
    });

    const resp = await fetch(`/api/workspace?${params.toString()}`);
    if (resp.status === 401) {
      window.location.replace('/login');
      return;
    }
    const data = await resp.json();
    currentWorkspaceData = data;
    window.currentWorkspaceData = data;

    renderWorkspace(data);

    const activeTab = document.querySelector('.tab-btn.active');
    if (activeTab && activeTab.dataset.tab === 'model-lab') {
      const activeSym = data.asset ? data.asset.symbol : 'BTC-CAD';
      const symEl = document.getElementById('benchmark-active-symbol');
      if (symEl) symEl.innerText = activeSym;
      renderBenchmarkTable(data.model_results || data.all_model_results || []);
    }

    if (btn) btn.classList.remove('animate-spin');
  } catch (err) {
    console.error('Workspace load error:', err);
    showToast('Failed to load workspace data: ' + err.message);
  }
}

function renderWorkspace(d) {
  const asset = d.asset;
  const execAsset = d.exec_asset;
  const quote = d.quote;
  const consensus = d.consensus;
  const ticket = d.order_ticket;
  const bot = d.virtual_bot;

  // Header badges & clocks
  document.getElementById('header-market-clock').innerText = d.market_clock.toronto_time + ' · ' + (d.market_clock.is_open ? '🟢 Open' : '🔴 Closed');
  document.getElementById('active-asset-category').innerText = asset.category.replace('_', ' ');
  document.getElementById('asset-search-input').value = asset.symbol;
  document.getElementById('stat-symbol').innerText = asset.symbol;
  document.getElementById('stat-price').innerText = '$' + formatNumber(quote.price) + ' ' + asset.currency;
  
  const chgEl = document.getElementById('stat-change');
  chgEl.innerText = (quote.change_pct >= 0 ? '+' : '') + quote.change_pct.toFixed(2) + '%';
  chgEl.className = 'font-mono font-semibold ' + (quote.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400');
  document.getElementById('stat-market-session').innerText = d.market_clock.msg;

  // Consensus Stat
  const dirEl = document.getElementById('stat-consensus-dir');
  const dirColor = consensus.direction === 'BULLISH' ? 'text-emerald-400' : (consensus.direction === 'BEARISH' ? 'text-rose-400' : 'text-amber-400');
  dirEl.innerHTML = `<span class="${dirColor}">${consensus.direction} (${consensus.exp_return_pct >= 0 ? '+' : ''}${consensus.exp_return_pct.toFixed(2)}%)</span>`;
  document.getElementById('stat-consensus-p50').innerText = '$' + formatNumber(consensus.p50);
  document.getElementById('stat-consensus-p10').innerText = '$' + formatNumber(consensus.p10);
  document.getElementById('stat-consensus-p90').innerText = '$' + formatNumber(consensus.p90);
  document.getElementById('stat-horizon').innerText = d.profile.horizon_bars + 'B';

  // Pair Proof-of-Edge Stat
  const eff = bot.efficiency_summary;
  const edgeEl = document.getElementById('stat-edge-badge');
  edgeEl.innerHTML = `<span class="px-2 py-0.5 rounded ${eff.readiness_state === 'PROVEN_EDGE' ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800' : (eff.readiness_state === 'CALIBRATING' ? 'bg-amber-950/80 text-amber-300 border border-amber-800' : 'bg-rose-950/80 text-rose-300 border border-rose-800')} text-xs font-bold">${eff.readiness_badge}</span>`;
  document.getElementById('stat-edge-winrate').innerText = eff.win_rate_pct.toFixed(1) + '% Win';
  document.getElementById('stat-edge-pf').innerText = eff.profit_factor.toFixed(2);
  document.getElementById('stat-edge-net').innerText = (eff.net_return_pct >= 0 ? '+' : '') + eff.net_return_pct.toFixed(1) + '%';

  // User Equity Stat
  document.getElementById('user-badge-equity').innerText = '$' + formatNumber(bot.total_equity_cad) + ' CAD';
  document.getElementById('stat-equity-cad').innerText = '$' + formatNumber(bot.total_equity_cad);
  document.getElementById('stat-cash-cad').innerText = '$' + formatNumber(bot.virtual_cash_cad);
  document.getElementById('stat-user-name').innerText = d.profile.display_name;

  const botBtn = document.getElementById('btn-toggle-bot-armed');
  if (bot.is_armed) {
    botBtn.className = 'px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition shadow-sm';
    botBtn.innerText = 'Armed ●';
    document.getElementById('stat-bot-active-badge').innerText = 'Armed ●';
    document.getElementById('stat-bot-active-badge').className = 'text-emerald-400 font-semibold';
  } else {
    botBtn.className = 'px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold transition border border-slate-700';
    botBtn.innerText = 'Paused';
    document.getElementById('stat-bot-active-badge').innerText = 'Paused';
    document.getElementById('stat-bot-active-badge').className = 'text-slate-500 font-semibold';
  }

  // Populate Chart Data
  rawBars = d.bars || [];
  if (candleSeries && rawBars.length) {
    changeChartStyle(currentChartMode);

    // Volume histogram
    const vols = rawBars.map(b => ({
      time: b.time,
      value: b.volume || 1000,
      color: b.close >= b.open ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)',
    }));
    volumeSeries.setData(vols);

    // EMA overlays
    if (d.indicators) {
      if (d.indicators.ema9) ema9Series.setData(d.indicators.ema9);
      if (d.indicators.ema21) ema21Series.setData(d.indicators.ema21);
      if (d.indicators.vwap) vwapSeries.setData(d.indicators.vwap);
      if (d.indicators.bb_upper) bbUpperSeries.setData(d.indicators.bb_upper);
      if (d.indicators.bb_lower) bbLowerSeries.setData(d.indicators.bb_lower);
    }

    // Forecast Path Line
    if (d.forecast_cone && d.forecast_cone.path) {
      forecastSeries.setData(d.forecast_cone.path);
    }

    // Trade Markers
    if (eff.chart_markers && eff.chart_markers.length) {
      candleSeries.setMarkers(eff.chart_markers);
    }

    // Default view: last 120 bars
    setWindowBars(120);
  }

  // Active Ensemble Weights Bar
  const weightsBar = document.getElementById('ensemble-weights-bar');
  if (weightsBar) {
    weightsBar.innerHTML = '';
    const mResults = d.model_results || d.all_model_results || [];
    mResults.forEach(m => {
      if (m.is_active || m.weight_pct > 0) {
        const chip = document.createElement('div');
        chip.className = 'px-2 py-1 rounded bg-slate-900 border border-slate-800 flex items-center gap-1.5 font-mono';
        chip.innerHTML = `
          <span class="text-slate-300 font-semibold">${m.short_label}</span>
          <span class="text-sky-400 font-bold">${m.weight_pct}%</span>
        `;
        weightsBar.appendChild(chip);
      }
    });
  }

  // Top Feature Drivers
  const driversBar = document.getElementById('feature-drivers-bar');
  driversBar.innerHTML = '';
  (d.feature_importance || []).slice(0, 4).forEach(f => {
    const box = document.createElement('div');
    box.className = 'p-2 rounded bg-slate-950 border border-slate-800 font-mono';
    box.innerHTML = `
      <span class="text-slate-500 block text-[10px] truncate">${f.label}</span>
      <span class="text-sky-300 font-bold text-xs">${f.importance_pct}%</span>
    `;
    driversBar.appendChild(box);
  });

  // Render Virtual Position Box
  const vBox = document.getElementById('virtual-position-box');
  const pos = bot.active_position;
  if (pos) {
    const unPnl = pos.unrealized_pnl_cad;
    const unPct = pos.unrealized_pnl_pct;
    const pnlClass = unPnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
    vBox.innerHTML = `
      <div class="flex items-center justify-between pb-1.5 border-b border-slate-800">
        <div>
          <span class="font-bold text-white">${pos.ticker}</span>
          <span class="text-slate-500 text-[10px] ml-1">${pos.quantity.toFixed(4)} Units</span>
        </div>
        <span class="font-bold font-mono ${pnlClass}">${unPnl >= 0 ? '+' : ''}$${formatNumber(unPnl)} CAD (${unPct >= 0 ? '+' : ''}${unPct.toFixed(2)}%)</span>
      </div>
      <div class="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 pt-1">
        <div>Entry: <strong class="text-slate-200">$${formatNumber(pos.entry_price)}</strong></div>
        <div>Mark: <strong class="text-slate-200">$${formatNumber(pos.current_price)}</strong></div>
        <div>Stop P10: <strong class="text-rose-400">$${formatNumber(pos.stop_p10)}</strong></div>
        <div>Target P90: <strong class="text-emerald-400">$${formatNumber(pos.target_p90)}</strong></div>
      </div>
      <div class="pt-2 flex items-center justify-between">
        <span class="text-[10px] px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">${pos.stop_status_desc || 'Stop Active'}</span>
        <button type="button" onclick="closeVirtualPosition('${pos.ticker}')" class="px-2.5 py-1 rounded bg-rose-950 hover:bg-rose-900 border border-rose-800 text-rose-300 text-xs font-semibold">
          Close Position
        </button>
      </div>
    `;
  } else {
    vBox.innerHTML = `
      <div class="text-center py-2 text-slate-400">
        <span class="block font-semibold text-slate-300">No Open Position in ${asset.symbol}</span>
        <span class="text-[11px] text-slate-500">Autonomous bot is monitoring live ticks for high-conviction entries.</span>
      </div>
    `;
  }

  // Mini Walk-Forward stats
  document.getElementById('mini-wf-trades').innerText = eff.trades_count + ' Trades';
  document.getElementById('mini-wf-winrate').innerText = eff.win_rate_pct.toFixed(1) + '%';
  document.getElementById('mini-wf-pf').innerText = eff.profit_factor.toFixed(2);
  document.getElementById('mini-wf-fees').innerText = '$' + eff.total_ws_fees_cad.toFixed(2) + ' CAD';

  // Render Wealthsimple Order Ticket
  const proxyCheckbox = document.getElementById('proxy-route-checkbox');
  proxyCheckbox.checked = d.profile.use_proxy_route;
  if (asset.alt_cad_proxy) {
    document.getElementById('proxy-route-banner').classList.remove('hidden');
    document.getElementById('proxy-switch-label').innerText = `Route via ${asset.alt_cad_proxy}`;
    document.getElementById('proxy-savings-text').innerText = ticket.proxy_savings_badge || 'Optimal low-fee route';
  } else {
    document.getElementById('proxy-route-banner').classList.add('hidden');
  }

  document.getElementById('ticket-ws-tier-badge').innerText = `WS ${d.profile.ws_tier} (${ticket.fee_estimate.badge_text})`;
  document.getElementById('copy-entry-price').innerText = '$' + formatNumber(ticket.limit_entry);
  document.getElementById('copy-tp-price').innerText = '$' + formatNumber(ticket.take_profit_p90);
  document.getElementById('copy-sl-price').innerText = '$' + formatNumber(ticket.stop_loss_p10);
  document.getElementById('ticket-shares-count').innerText = ticket.recommended_shares.toFixed(4);
  document.getElementById('ticket-exec-unit').innerText = ticket.exec_symbol;
  document.getElementById('kelly-calc-text').innerText = '$' + formatNumber(ticket.quarter_kelly_cad);
  document.getElementById('btn-paper-alloc').innerText = document.getElementById('ticket-capital-cad').value;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. User Actions & Interactive Handlers
// ─────────────────────────────────────────────────────────────────────────────
function copyToClipboard(text, label) {
  const clean = text.replace('$', '').trim();
  navigator.clipboard.writeText(clean).then(() => {
    showToast(`Copied ${label} (${clean}) to clipboard`);
  }).catch(() => {
    showToast(`Copied ${clean}`);
  });
}

function showToast(msg) {
  const t = document.getElementById('toast');
  document.getElementById('toast-text').innerText = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

function formatNumber(val) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  const num = Number(val);
  if (Math.abs(num) >= 1000) {
    return num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  } else if (Math.abs(num) >= 1) {
    return num.toFixed(2);
  } else {
    return num.toFixed(4);
  }
}

function formatCompact(num) {
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toString();
}

async function refreshCurrentWorkspace() {
  await loadWorkspace(false);
}

async function retrainAllModels() {
  showToast('Retraining all 8 models on GPU...');
  await loadWorkspace(true);
  showToast('8-Model Zoo Retrained & Calibrated');
}

async function applyQuarterKelly() {
  if (!currentWorkspaceData) return;
  const kellyCad = currentWorkspaceData.order_ticket.quarter_kelly_cad;
  document.getElementById('ticket-capital-cad').value = Math.max(100, Math.round(kellyCad));
  recalculateTicket();
}

function recalculateTicket() {
  if (!currentWorkspaceData) return;
  const alloc = parseFloat(document.getElementById('ticket-capital-cad').value) || 500;
  document.getElementById('btn-paper-alloc').innerText = Math.round(alloc);
  const execPrice = currentWorkspaceData.order_ticket.exec_price || 1.0;
  const shares = alloc / execPrice;
  document.getElementById('ticket-shares-count').innerText = shares.toFixed(4);
}

async function toggleProxyRoute(isChecked) {
  try {
    await fetch('/api/preferences', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_proxy_route: isChecked }),
    });
    showToast(isChecked ? 'CAD Proxy Routing Enabled' : 'Direct Asset Trading Enabled');
    loadWorkspace(false);
  } catch (err) {
    showToast('Failed to update proxy preference');
  }
}

async function toggleBotArmed() {
  try {
    const isCurrentlyArmed = currentWorkspaceData ? currentWorkspaceData.virtual_bot.is_armed : true;
    const resp = await fetch('/api/bot/arm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_armed: !isCurrentlyArmed }),
    });
    const res = await resp.json();
    showToast(res.is_armed ? 'Virtual Bot Armed & Active' : 'Virtual Bot Paused');
    loadWorkspace(false);
  } catch (err) {
    showToast('Error toggling bot status');
  }
}

async function submitPaperBuy() {
  const alloc = parseFloat(document.getElementById('ticket-capital-cad').value) || 500;
  const sym = currentWorkspaceData ? currentWorkspaceData.asset.symbol : 'BTC-CAD';
  const useProxy = document.getElementById('proxy-route-checkbox').checked;

  try {
    const resp = await fetch('/api/bot/paper-buy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: sym, capital_cad: alloc, use_proxy: useProxy }),
    });
    const res = await resp.json();
    if (res.status === 'SUCCESS') {
      showToast(res.message);
      loadWorkspace(false);
    } else {
      showToast('Error: ' + res.message);
    }
  } catch (err) {
    showToast('Failed to execute paper buy: ' + err.message);
  }
}

async function closeVirtualPosition(ticker) {
  try {
    const resp = await fetch('/api/bot/paper-close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker: ticker }),
    });
    const res = await resp.json();
    if (res.status === 'SUCCESS') {
      showToast(res.message);
      loadWorkspace(false);
    } else {
      showToast('Error closing position');
    }
  } catch (err) {
    showToast('Error: ' + err.message);
  }
}

async function resetVirtualWallet() {
  if (!confirm('Reset virtual portfolio wallet back to $10,000.00 CAD?')) return;
  try {
    const resp = await fetch('/api/bot/reset-wallet', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ starting_cad: 10000.0 }),
    });
    const res = await resp.json();
    showToast('Virtual wallet reset to $10,000 CAD');
    loadWorkspace(false);
  } catch (err) {
    showToast('Reset failed');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Tab 2: Alpha Scanner & Benchmark Logic
// ─────────────────────────────────────────────────────────────────────────────
async function runAlphaScanner(force = false) {
  const tbody = document.getElementById('scanner-table-body');
  tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500">Scanning liquid Wealthsimple universe...</td></tr>';

  try {
    const resp = await fetch(`/api/scanner?force=${force ? 'true' : 'false'}`);
    const data = await resp.json();
    renderScannerTable(data.opportunities || []);
    renderNewsWire(data.news_articles || [], data.news_sentiment || 0.0);
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-rose-500">Scanner failed</td></tr>';
  }
}

function renderScannerTable(opps) {
  const tbody = document.getElementById('scanner-table-body');
  tbody.innerHTML = '';
  if (!opps.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500">No scanner opportunities found</td></tr>';
    return;
  }

  opps.forEach(o => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-900/60 transition';
    const evColor = o.net_ev_pct > 0.5 ? 'text-emerald-400' : (o.net_ev_pct >= 0 ? 'text-slate-300' : 'text-rose-400');
    const actBadge = o.action === 'STRONG_BUY' ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : (
      o.action === 'TACTICAL_BUY' ? 'bg-sky-950 text-sky-300 border border-sky-800' : 'bg-slate-800 text-slate-400 border border-slate-700'
    );

    tr.innerHTML = `
      <td class="py-2.5">
        <span class="font-bold text-white block">${o.symbol}</span>
        <span class="text-[10px] text-slate-500 font-sans">${o.name}</span>
      </td>
      <td class="py-2.5">
        <span class="text-sky-300 text-[11px]">${o.optimal_route_symbol}</span>
        <span class="text-[9px] text-slate-500 block font-sans">${o.tfsa_status}</span>
      </td>
      <td class="py-2.5 text-right text-slate-200">$${formatNumber(o.current_price)}</td>
      <td class="py-2.5 text-right font-bold ${evColor}">${o.net_ev_pct >= 0 ? '+' : ''}${o.net_ev_pct.toFixed(2)}%</td>
      <td class="py-2.5 text-center">
        <span class="px-1.5 py-0.5 rounded text-[10px] font-semibold ${actBadge}">${o.action.replace('_', ' ')}</span>
      </td>
      <td class="py-2.5 text-right">
        <button
          type="button"
          onclick="selectSymbol('${o.symbol}')"
          class="px-2 py-1 rounded bg-slate-800 hover:bg-sky-500 hover:text-slate-950 text-sky-300 text-[11px] font-semibold transition"
        >
          Open &rarr;
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  } );
}

function renderNewsWire(articles, sentiment) {
  const badge = document.getElementById('news-sentiment-badge');
  badge.innerText = `Sentiment: ${sentiment >= 0 ? '+' : ''}${sentiment.toFixed(2)}`;
  badge.className = `text-[10px] font-mono px-2 py-0.5 rounded ${sentiment > 0.1 ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : (sentiment < -0.1 ? 'bg-rose-950 text-rose-300 border border-rose-800' : 'bg-slate-800 text-slate-400')}`;

  const list = document.getElementById('news-wire-list');
  list.innerHTML = '';
  if (!articles.length) {
    list.innerHTML = '<div class="text-slate-500 text-center py-2">No breaking headlines recorded</div>';
    return;
  }
  articles.slice(0, 6).forEach(a => {
    const item = document.createElement('a');
    item.href = a.url || '#';
    item.target = '_blank';
    item.className = 'block p-2 rounded bg-slate-950 hover:bg-slate-900 border border-slate-800/80 transition';
    item.innerHTML = `
      <div class="flex items-center justify-between text-[10px] text-slate-500 mb-0.5">
        <span>${a.source || 'Market News'}</span>
        <span class="text-sky-400 uppercase font-semibold">${a.catalyst || 'MARKET'}</span>
      </div>
      <div class="text-slate-200 text-xs font-medium line-clamp-2">${a.title}</div>
    `;
    list.appendChild(item);
  });
}

function renderBenchmarkTable(results) {
  const tbody = document.getElementById('benchmark-table-body');
  tbody.innerHTML = '';
  (results || []).forEach(m => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-900/60 transition';
    const skillColor = m.skill_pct > 0 ? 'text-emerald-400 font-bold' : (m.skill_pct >= -15 ? 'text-slate-300' : 'text-rose-400');
    tr.innerHTML = `
      <td class="py-2.5">
        <span class="font-bold text-white block">${m.name}</span>
        <span class="text-[10px] text-slate-500 font-sans">${m.family} · ${m.suitability}</span>
      </td>
      <td class="py-2.5 text-right text-slate-300">${m.mape_pct.toFixed(2)}%</td>
      <td class="py-2.5 text-right ${skillColor}">${m.skill_pct >= 0 ? '+' : ''}${m.skill_pct.toFixed(1)}%</td>
      <td class="py-2.5 text-right font-bold ${m.hit_rate_pct >= 55 ? 'text-emerald-400' : 'text-slate-400'}">${m.hit_rate_pct.toFixed(1)}%</td>
      <td class="py-2.5 text-right text-slate-200 font-bold">$${formatNumber(m.p50)}</td>
      <td class="py-2.5 text-right text-sky-400 font-bold">${m.weight_pct.toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Tab 3: Portfolios & Leaderboard Logic
// ─────────────────────────────────────────────────────────────────────────────
async function loadPortfolios() {
  try {
    const resp = await fetch('/api/portfolios');
    const data = await resp.json();

    // 1. Leaderboard
    const lbBody = document.getElementById('leaderboard-table-body');
    lbBody.innerHTML = '';
    (data.leaderboard || []).forEach(l => {
      const tr = document.createElement('tr');
      const retColor = l.net_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400';
      tr.innerHTML = `
        <td class="py-2.5 font-bold text-white">${l.display_name}</td>
        <td class="py-2.5 text-slate-400">${l.ws_tier}</td>
        <td class="py-2.5 text-right text-slate-300">$${formatNumber(l.starting_cad)}</td>
        <td class="py-2.5 text-right font-bold text-white">$${formatNumber(l.current_equity_cad)}</td>
        <td class="py-2.5 text-right font-bold ${retColor}">${l.net_return_pct >= 0 ? '+' : ''}${l.net_return_pct.toFixed(2)}%</td>
        <td class="py-2.5 text-center text-sky-400 font-bold">${l.open_virtual_count}</td>
        <td class="py-2.5 text-center text-slate-300">${l.completed_trades_count}</td>
        <td class="py-2.5 text-center text-slate-400">${l.real_ws_trades_count}</td>
      `;
      lbBody.appendChild(tr);
    });

    // 2. Open Positions
    const posBody = document.getElementById('open-virtual-table-body');
    posBody.innerHTML = '';
    const openPos = data.open_virtual_positions || [];
    document.getElementById('open-positions-count-badge').innerText = `${openPos.length} Open`;
    if (!openPos.length) {
      posBody.innerHTML = '<tr><td colspan="8" class="py-4 text-center text-slate-500">No active virtual positions</td></tr>';
    } else {
      openPos.forEach(p => {
        const tr = document.createElement('tr');
        const pnl = p.unrealized_pnl_cad;
        const pnlColor = pnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
        tr.innerHTML = `
          <td class="py-2.5 font-bold text-white">${p.ticker}</td>
          <td class="py-2.5 text-slate-300">${p.quantity.toFixed(4)}</td>
          <td class="py-2.5 text-right text-slate-300">$${formatNumber(p.entry_price)}</td>
          <td class="py-2.5 text-right font-bold text-white">$${formatNumber(p.current_price)}</td>
          <td class="py-2.5 text-right font-bold ${pnlColor}">${pnl >= 0 ? '+' : ''}$${formatNumber(pnl)} (${p.unrealized_pnl_pct.toFixed(2)}%)</td>
          <td class="py-2.5 text-right text-rose-400">$${formatNumber(p.stop_p10)}</td>
          <td class="py-2.5 text-right text-emerald-400">$${formatNumber(p.target_p90)}</td>
          <td class="py-2.5 text-center">
            <button onclick="closeVirtualPosition('${p.ticker}')" class="px-2 py-0.5 rounded bg-rose-950 hover:bg-rose-900 border border-rose-800 text-rose-300 text-xs">Close</button>
          </td>
        `;
        posBody.appendChild(tr);
      });
    }

    // 3. Closed Trades History
    const closedBody = document.getElementById('closed-virtual-table-body');
    closedBody.innerHTML = '';
    const history = data.completed_virtual_trades || [];
    if (!history.length) {
      closedBody.innerHTML = '<tr><td colspan="9" class="py-4 text-center text-slate-500">No closed trades recorded</td></tr>';
    } else {
      history.forEach(h => {
        const tr = document.createElement('tr');
        const netColor = h.net_pnl_cad >= 0 ? 'text-emerald-400' : 'text-rose-400';
        tr.innerHTML = `
          <td class="py-2 font-bold text-white">${h.ticker}</td>
          <td class="py-2 text-right text-slate-300">$${formatNumber(h.entry_price)}</td>
          <td class="py-2 text-right text-slate-300">$${formatNumber(h.exit_price)}</td>
          <td class="py-2 text-right text-slate-400">$${formatNumber(h.gross_pnl_cad)}</td>
          <td class="py-2 text-right text-rose-400">-$${formatNumber(h.ws_fees_cad)}</td>
          <td class="py-2 text-right font-bold ${netColor}">${h.net_pnl_cad >= 0 ? '+' : ''}$${formatNumber(h.net_pnl_cad)}</td>
          <td class="py-2 text-right font-bold ${netColor}">${h.net_pnl_pct.toFixed(2)}%</td>
          <td class="py-2 text-slate-400 text-[11px]">${h.exit_reason || '—'}</td>
          <td class="py-2 text-slate-500 text-[10px]">${h.closed_at || '—'}</td>
        `;
        closedBody.appendChild(tr);
      });
    }

    // 4. Real Wealthsimple Trades
    const realBody = document.getElementById('real-ws-table-body');
    realBody.innerHTML = '';
    const realTrades = data.real_ws_positions || [];
    if (!realTrades.length) {
      realBody.innerHTML = '<tr><td colspan="9" class="py-4 text-center text-slate-500">No real Wealthsimple fills logged</td></tr>';
    } else {
      realTrades.forEach(r => {
        const tr = document.createElement('tr');
        const isClosed = r.status === 'CLOSED';
        const pnl = r.realized_pnl_cad;
        const pnlText = isClosed && pnl !== null ? `${pnl >= 0 ? '+' : ''}$${formatNumber(pnl)}` : '—';
        const pnlColor = isClosed && pnl >= 0 ? 'text-emerald-400 font-bold' : (isClosed ? 'text-rose-400 font-bold' : 'text-slate-500');

        tr.innerHTML = `
          <td class="py-2 font-mono text-[11px] text-slate-400">${r.tx_id}</td>
          <td class="py-2 font-bold text-white">${r.ticker}</td>
          <td class="py-2"><span class="px-1.5 py-0.5 rounded text-[10px] ${isClosed ? 'bg-slate-800 text-slate-400' : 'bg-emerald-950 text-emerald-300'} font-semibold">${r.status}</span></td>
          <td class="py-2 text-right text-slate-300">$${formatNumber(r.entry_price)}</td>
          <td class="py-2 text-right text-slate-300">${isClosed && r.exit_price ? '$' + formatNumber(r.exit_price) : '—'}</td>
          <td class="py-2 text-right text-slate-300">$${formatNumber(r.capital_cad)}</td>
          <td class="py-2 text-right ${pnlColor}">${pnlText}</td>
          <td class="py-2 text-right ${pnlColor}">${isClosed && r.realized_pnl_pct !== null ? r.realized_pnl_pct.toFixed(2) + '%' : '—'}</td>
          <td class="py-2 text-center">
            ${isClosed ? '<span class="text-slate-600">—</span>' : `<button onclick="closeRealWsTrade('${r.tx_id}')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-sky-300 text-xs">Close</button>`}
          </td>
        `;
        realBody.appendChild(tr);
      });
    }

    // 5. Hardware Telemetry
    if (data.telemetry) {
      const gpu = data.telemetry.gpu;
      if (gpu && gpu.available) {
        document.getElementById('telemetry-gpu').innerText = gpu.device_name;
        document.getElementById('telemetry-gpu-vram').innerText = `${gpu.free_gb} GB Free / ${gpu.total_gb} GB Total`;
      }
      const mem = data.telemetry.memory;
      if (mem) {
        document.getElementById('telemetry-ram').innerText = `${mem.total} Total`;
        document.getElementById('telemetry-ram-load').innerText = `${mem.used} Used (${mem.percent}%)`;
      }
    }
  } catch (err) {
    console.error('Error loading portfolios:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Real Wealthsimple Modal & Logging
// ─────────────────────────────────────────────────────────────────────────────
function openLogRealTradeModal() {
  if (!currentWorkspaceData) return;
  const tkr = currentWorkspaceData.order_ticket.exec_symbol || currentWorkspaceData.asset.symbol;
  document.getElementById('log-real-ticker').value = tkr;
  document.getElementById('log-real-entry').value = currentWorkspaceData.order_ticket.limit_entry;
  document.getElementById('log-real-qty').value = currentWorkspaceData.order_ticket.recommended_shares;
  document.getElementById('log-real-target').value = currentWorkspaceData.order_ticket.take_profit_p90;
  document.getElementById('log-real-stop').value = currentWorkspaceData.order_ticket.stop_loss_p10;
  document.getElementById('modal-log-trade').classList.remove('hidden');
}

function closeLogRealTradeModal() {
  document.getElementById('modal-log-trade').classList.add('hidden');
}

async function submitLogRealTrade() {
  const tkr = document.getElementById('log-real-ticker').value;
  const entry = parseFloat(document.getElementById('log-real-entry').value);
  const qty = parseFloat(document.getElementById('log-real-qty').value);
  const target = parseFloat(document.getElementById('log-real-target').value);
  const stop = parseFloat(document.getElementById('log-real-stop').value);

  if (!entry || !qty) {
    showToast('Please enter fill price and quantity');
    return;
  }

  try {
    const resp = await fetch('/api/real-trades/open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ticker: tkr,
        entry_price: entry,
        quantity: qty,
        target_price: target,
        stop_price: stop,
      }),
    });
    const res = await resp.json();
    closeLogRealTradeModal();
    showToast('Real Wealthsimple Fill Logged');
    loadWorkspace(false);
  } catch (err) {
    showToast('Failed to log real fill: ' + err.message);
  }
}

async function closeRealWsTrade(txId) {
  const exitPriceStr = prompt('Enter your Wealthsimple exit fill price ($ CAD):');
  if (!exitPriceStr) return;
  const exitPrice = parseFloat(exitPriceStr);
  if (isNaN(exitPrice) || exitPrice <= 0) {
    showToast('Invalid price entered');
    return;
  }

  try {
    const resp = await fetch('/api/real-trades/close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tx_id: txId, exit_price: exitPrice }),
    });
    showToast('Real position closed and realized P&L saved');
    loadPortfolios();
  } catch (err) {
    showToast('Error closing real position');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Asset Search & Profile Switching
// ─────────────────────────────────────────────────────────────────────────────
async function selectSymbol(symbol) {
  try {
    await fetch('/api/preferences', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active_symbol: symbol }),
    });
    // Switch to Terminal Tab
    switchTab('terminal');
    await loadWorkspace(false);
  } catch (err) {
    showToast('Failed to change symbol');
  }
}

async function logoutSession() {
  await fetch('/api/auth/logout', { method: 'POST' });
  window.location.replace('/login');
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('hidden', pane.id !== `tab-${tabName}`);
  });

  if (tabName === 'model-lab') {
    runAlphaScanner(false);
    if (currentWorkspaceData) {
      const activeSym = currentWorkspaceData.asset ? currentWorkspaceData.asset.symbol : 'BTC-CAD';
      const symEl = document.getElementById('benchmark-active-symbol');
      if (symEl) symEl.innerText = activeSym;
      renderBenchmarkTable(currentWorkspaceData.model_results || currentWorkspaceData.all_model_results || []);
    }
  } else if (tabName === 'portfolios') {
    loadPortfolios();
  } else if (tabName === 'terminal' && chart) {
    setTimeout(() => {
      chart.applyOptions({ width: document.getElementById('chart-container').clientWidth });
      chart.timeScale().fitContent();
    }, 50);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Event Listeners & Startup
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Timeframe pills
  document.querySelectorAll('.timeframe-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tf = btn.dataset.tf;
      await fetch('/api/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active_timeframe: tf }),
      });
      loadWorkspace(false);
    });
  });

  // Horizon pills
  document.querySelectorAll('.horizon-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.horizon-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const h = parseInt(btn.dataset.horizon, 10);
      await fetch('/api/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ horizon_bars: h }),
      });
      loadWorkspace(false);
    });
  });

  // Window pills
  document.querySelectorAll('.window-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.window-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setWindowBars(btn.dataset.bars);
    });
  });

  // Model Preset pills
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.dataset.preset;
      await fetch('/api/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_preset: preset }),
      });
      loadWorkspace(false);
    });
  });

  // Asset Search Input
  const searchInput = document.getElementById('asset-search-input');
  const dropdown = document.getElementById('asset-search-dropdown');

  let debounceTimer = null;
  searchInput.addEventListener('input', e => {
    clearTimeout(debounceTimer);
    const q = e.target.value.trim();
    if (q.length < 1) {
      dropdown.classList.add('hidden');
      return;
    }
    debounceTimer = setTimeout(async () => {
      try {
        const resp = await fetch(`/api/catalog?q=${encodeURIComponent(q)}`);
        const data = await resp.json();
        dropdown.innerHTML = '';
        if (data.assets && data.assets.length) {
          data.assets.forEach(a => {
            const row = document.createElement('div');
            row.className = 'px-3 py-2 hover:bg-slate-800 cursor-pointer flex items-center justify-between border-b border-slate-800/40 last:border-0';
            row.innerHTML = `
              <div>
                <span class="font-bold text-white font-mono uppercase">${a.symbol}</span>
                <span class="text-slate-400 text-[11px] block truncate">${a.name}</span>
              </div>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-semibold">${a.category.replace('_', ' ')}</span>
            `;
            row.addEventListener('click', () => {
              dropdown.classList.add('hidden');
              selectSymbol(a.symbol);
            });
            dropdown.appendChild(row);
          });
          dropdown.classList.remove('hidden');
        } else {
          dropdown.classList.add('hidden');
        }
      } catch (err) {}
    }, 200);
  });

  document.addEventListener('click', e => {
    if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.add('hidden');
    }
  });

  // Initial Load
  loadWorkspace(false);
});
