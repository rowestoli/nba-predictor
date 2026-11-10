import './style.css';
import { americanOdds, defaultLine, probOver, type ResidualModel } from './odds';

type Stat = 'pts' | 'reb' | 'ast' | 'fg3';
const STATS: Stat[] = ['pts', 'reb', 'ast', 'fg3'];
const LABEL: Record<Stat, string> = { pts: 'Points', reb: 'Rebounds', ast: 'Assists', fg3: '3-pointers' };
const SHORT: Record<Stat, string> = { pts: 'PTS', reb: 'REB', ast: 'AST', fg3: '3PM' };
const POPULAR = ['Nikola Jokic', 'Shai Gilgeous-Alexander', 'Jayson Tatum', 'Stephen Curry', 'LeBron James', 'Anthony Edwards'];

interface IndexEntry { id: number; name: string; team: string | null; position: string | null; pts: number; games: number }
interface Game { date: string; opp: string; home: number; min: number; pts: number; reb: number; ast: number; fg3: number }
interface Player {
  id: number; name: string; team: string | null; position: string | null; games: number;
  season: Record<Stat | 'minutes', number>;
  last_season: Record<Stat, number | null>;
  recent: Game[];
  scenarios: { opp: string; home: number }[];
  predictions: Record<Stat, number[]>;
}
interface Metric {
  label: string; mae_model: number; mae_baseline: number; improvement: number;
  pick_accuracy: number; always_under: number; recent_form_rule: number;
  brier: number; brier_base_rate: number;
  calibration: { predicted: number; actual: number; n: number }[];
}
interface Model {
  snapshot: string; season: string; split: Record<string, number>;
  metrics: Record<Stat, Metric>;
  residuals: Record<Stat, ResidualModel>;
  importance: { feature: string; mae_increase: number }[];
  teams: Record<string, string>;
  defense: Record<string, Record<Stat, number>>;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (x: number) => `${Math.round(x * 100)}%`;

let model: Model;
let players: IndexEntry[] = [];
let player: Player | null = null;
let opp = '';
let home = 1;
let chartStat: Stat = 'pts';
let calStat: Stat = 'pts';
const lines: Partial<Record<Stat, number>> = {};
const cache = new Map<number, Promise<Player>>();

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`/data/${path}`);
  if (!r.ok) throw new Error(`Couldn't load ${path}`);
  return r.json() as Promise<T>;
}
const loadPlayer = (id: number) => {
  if (!cache.has(id)) cache.set(id, getJSON<Player>(`players/${id}.json`));
  return cache.get(id)!;
};

// ---------- boot ----------
(async () => {
  [model, players] = await Promise.all([getJSON<Model>('model.json'), getJSON<IndexEntry[]>('players.json')]);
  $('snapshot').textContent = `${model.season} season, games through ${model.snapshot}. Projections are for each player's next game after that.`;
  $('test-games').textContent = model.split.test.toLocaleString();
  renderOpponents();
  renderPopular();
  renderMetrics();
  renderSeg('chart-stat', chartStat, (s) => { chartStat = s; drawChart(); });
  renderSeg('cal-stat', calStat, (s) => { calStat = s; drawCalibration(); });
  drawCalibration();
  drawImportance();
  const fromHash = Number(new URLSearchParams(location.hash.slice(1)).get('player'));
  const first = players.find((p) => p.id === fromHash) ?? players.find((p) => p.name === POPULAR[0]) ?? players[0];
  selectPlayer(first.id);
})();

// ---------- player picking ----------
const input = $('player-input') as HTMLInputElement;
const list = $('player-list');
input.addEventListener('input', () => {
  const q = input.value.trim().toLowerCase();
  if (!q) { list.hidden = true; return; }
  const found = players.filter((p) => p.name.toLowerCase().includes(q)).slice(0, 8);
  list.innerHTML = found.length
    ? found.map((p) => `<li role="option" tabindex="-1" data-id="${p.id}">${esc(p.name)}<span>${p.team ?? ''} · ${p.pts} ppg</span></li>`).join('')
    : '<li class="empty">No player by that name in this snapshot</li>';
  list.hidden = false;
});
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { const li = list.querySelector<HTMLElement>('li[data-id]'); if (li) pick(Number(li.dataset.id)); }
  if (e.key === 'ArrowDown') { list.querySelector<HTMLElement>('li[data-id]')?.focus(); e.preventDefault(); }
  if (e.key === 'Escape') list.hidden = true;
});
list.addEventListener('keydown', (e) => {
  const li = e.target as HTMLElement;
  if (e.key === 'Enter' && li.dataset.id) pick(Number(li.dataset.id));
  if (e.key === 'ArrowDown') (li.nextElementSibling as HTMLElement | null)?.focus();
  if (e.key === 'ArrowUp') ((li.previousElementSibling as HTMLElement | null) ?? input).focus();
});
list.addEventListener('click', (e) => {
  const li = (e.target as HTMLElement).closest<HTMLElement>('li[data-id]');
  if (li) pick(Number(li.dataset.id));
});
document.addEventListener('click', (e) => { if (!(e.target as HTMLElement).closest('.combo')) list.hidden = true; });
function pick(id: number) {
  input.value = '';
  list.hidden = true;
  selectPlayer(id);
}

function renderPopular() {
  const pop = POPULAR.map((n) => players.find((p) => p.name === n)).filter(Boolean) as IndexEntry[];
  $('popular').innerHTML = pop.map((p) => `<button type="button" data-id="${p.id}">${esc(p.name)}</button>`).join('');
  $('popular').addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>('button[data-id]');
    if (b) selectPlayer(Number(b.dataset.id));
  });
}

async function selectPlayer(id: number) {
  const p = await loadPlayer(id);
  player = p;
  for (const s of STATS) lines[s] = defaultLine(p.season[s]);
  // Keep the chosen opponent unless it's this player's own team
  const opps = [...new Set(p.scenarios.map((s) => s.opp))];
  if (!opps.includes(opp)) opp = opps[0];
  ($('opp') as HTMLSelectElement).value = opp;
  [...($('opp') as HTMLSelectElement).options].forEach((o) => (o.disabled = !opps.includes(o.value)));
  history.replaceState(null, '', `#player=${id}`);
  $('p-name').textContent = p.name;
  $('p-meta').textContent = [p.team && model.teams[p.team], p.position, `${p.games} games this season`, `${p.season.minutes} min per game`]
    .filter(Boolean).join(' · ');
  document.querySelectorAll<HTMLElement>('#popular button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.id) === id)));
  render(true);
}

// ---------- matchup ----------
function renderOpponents() {
  const sel = $('opp') as HTMLSelectElement;
  sel.innerHTML = Object.entries(model.teams)
    .sort((a, b) => a[1].localeCompare(b[1]))
    .map(([code, name]) => `<option value="${code}">${esc(name)}</option>`).join('');
  sel.addEventListener('change', () => { opp = sel.value; render(); });
}
$('venue').addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest<HTMLElement>('button[data-home]');
  if (!b) return;
  home = Number(b.dataset.home);
  $('venue').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  render();
});

function projection(stat: Stat): number {
  const i = player!.scenarios.findIndex((s) => s.opp === opp && s.home === home);
  return player!.predictions[stat][i];
}

// ---------- prediction cards ----------
function render(fresh = false) {
  if (!player) return;
  const d = model.defense[opp];
  const pts = d ? d.pts - 1 : 0;
  $('matchup').innerHTML = `${home ? 'vs' : '@'} <strong>${esc(model.teams[opp])}</strong><span class="muted small">${
    Math.abs(pts) < 0.01 ? 'Average defense' : `Allows ${pct(Math.abs(pts))} ${pts > 0 ? 'more' : 'fewer'} points than average`}</span>`;

  if (fresh || !$('cards').children.length) {
    $('cards').innerHTML = STATS.map((s) => `
      <article class="card" data-stat="${s}">
        <header><span>${LABEL[s]}</span><span class="muted small" data-avg></span></header>
        <p class="proj"><span data-proj>0</span><span class="delta" data-delta></span></p>
        <div class="line">
          <span class="muted small">Line</span>
          <button type="button" data-step="-1" aria-label="Lower the ${LABEL[s]} line">−</button>
          <input type="number" step="0.5" min="0.5" data-line aria-label="${LABEL[s]} line" />
          <button type="button" data-step="1" aria-label="Raise the ${LABEL[s]} line">+</button>
        </div>
        <div class="ou">
          <div class="ou__bar"><span data-over></span></div>
          <div class="ou__labels"><span data-over-label></span><span data-under-label></span></div>
        </div>
      </article>`).join('');
  }
  for (const s of STATS) updateCard(s);
  drawChart();
}

function updateCard(s: Stat) {
  const card = $('cards').querySelector<HTMLElement>(`[data-stat="${s}"]`)!;
  const proj = projection(s);
  const avg = player!.season[s];
  const line = lines[s]!;
  const p = probOver(proj, line, model.residuals[s]);
  card.querySelector<HTMLElement>('[data-avg]')!.textContent = `${avg.toFixed(1)} season avg`;
  countTo(card.querySelector<HTMLElement>('[data-proj]')!, proj);
  const delta = proj - avg;
  const dEl = card.querySelector<HTMLElement>('[data-delta]')!;
  dEl.textContent = Math.abs(delta) < 0.05 ? 'even' : `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)}`;
  dEl.className = `delta ${delta > 0.05 ? 'up' : delta < -0.05 ? 'down' : ''}`;
  (card.querySelector('[data-line]') as HTMLInputElement).value = String(line);
  card.querySelector<HTMLElement>('[data-over]')!.style.width = `${p * 100}%`;
  card.querySelector<HTMLElement>('[data-over-label]')!.innerHTML = `Over <strong>${pct(p)}</strong> <span class="muted">${americanOdds(p)}</span>`;
  card.querySelector<HTMLElement>('[data-under-label]')!.innerHTML = `<span class="muted">${americanOdds(1 - p)}</span> <strong>${pct(1 - p)}</strong> Under`;
}

$('cards').addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest<HTMLElement>('button[data-step]');
  if (!b) return;
  const s = b.closest<HTMLElement>('[data-stat]')!.dataset.stat as Stat;
  lines[s] = Math.max(0.5, lines[s]! + Number(b.dataset.step));
  updateCard(s);
});
$('cards').addEventListener('change', (e) => {
  const inp = e.target as HTMLInputElement;
  if (!inp.matches('[data-line]')) return;
  const s = inp.closest<HTMLElement>('[data-stat]')!.dataset.stat as Stat;
  const v = Number(inp.value);
  // Lines sit on .5 so there's no push
  lines[s] = Number.isFinite(v) && v > 0 ? Math.floor(v) + 0.5 : lines[s];
  updateCard(s);
});

function countTo(el: HTMLElement, target: number) {
  const from = Number(el.dataset.v ?? 0);
  el.dataset.v = String(target);
  if (reduceMotion || Math.abs(target - from) < 0.05) { el.textContent = target.toFixed(1); return; }
  const start = performance.now();
  const step = (t: number) => {
    const k = Math.min(1, (t - start) / 450);
    const e = 1 - (1 - k) ** 3;
    el.textContent = (from + (target - from) * e).toFixed(1);
    if (k < 1 && el.dataset.v === String(target)) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------- charts (plain SVG) ----------
const SVG = 'http://www.w3.org/2000/svg';
function svgEl(tag: string, attrs: Record<string, string | number>, parent: Element, text?: string) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  parent.appendChild(n);
  return n;
}

function drawChart() {
  if (!player) return;
  const svg = $('chart') as unknown as SVGSVGElement;
  const W = 760, H = 220, L = 34, B = 38, T = 14;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const games = [...player.recent];
  const vals = games.map((g) => g[chartStat]);
  const proj = projection(chartStat);
  const line = lines[chartStat]!;
  const max = Math.max(4, ...vals, proj, line) * 1.15;
  const y = (v: number) => H - B - (v / max) * (H - B - T);
  const slots = games.length + 1;
  const bw = (W - L) / slots;
  svg.setAttribute('aria-label', `${LABEL[chartStat]} in the last ${games.length} games, with the next-game projection of ${proj.toFixed(1)}`);
  for (const t of niceTicks(max)) {
    svgEl('line', { x1: L, x2: W, y1: y(t), y2: y(t), class: 'grid' }, svg);
    svgEl('text', { x: L - 8, y: y(t) + 4, class: 'tick', 'text-anchor': 'end' }, svg, String(t));
  }
  games.forEach((g, i) => {
    const x = L + i * bw + bw * 0.18;
    const h = Math.max(1.5, H - B - y(g[chartStat]));
    const over = g[chartStat] > line;
    svgEl('rect', { x, y: y(g[chartStat]), width: bw * 0.64, height: h, rx: 3, class: `bar ${over ? 'bar--over' : ''}`, style: `animation-delay:${i * 25}ms` }, svg);
    svgEl('text', { x: x + bw * 0.32, y: y(g[chartStat]) - 5, class: 'val', 'text-anchor': 'middle' }, svg, String(g[chartStat]));
    svgEl('text', { x: x + bw * 0.32, y: H - B + 15, class: 'tick', 'text-anchor': 'middle' }, svg, `${g.home ? '' : '@'}${g.opp}`);
    svgEl('text', { x: x + bw * 0.32, y: H - B + 28, class: 'tick tick--dim', 'text-anchor': 'middle' }, svg, g.date);
  });
  // Next game projection
  const x = L + games.length * bw + bw * 0.18;
  svgEl('rect', { x, y: y(proj), width: bw * 0.64, height: H - B - y(proj), rx: 3, class: 'bar bar--proj' }, svg);
  svgEl('text', { x: x + bw * 0.32, y: y(proj) - 5, class: 'val val--proj', 'text-anchor': 'middle' }, svg, proj.toFixed(1));
  svgEl('text', { x: x + bw * 0.32, y: H - B + 15, class: 'tick tick--proj', 'text-anchor': 'middle' }, svg, `${home ? '' : '@'}${opp}`);
  svgEl('text', { x: x + bw * 0.32, y: H - B + 28, class: 'tick tick--proj', 'text-anchor': 'middle' }, svg, 'Next');
  // The line
  svgEl('line', { x1: L, x2: W, y1: y(line), y2: y(line), class: 'ref' }, svg);
  svgEl('text', { x: L + 6, y: y(line) - 6, class: 'ref-label' }, svg, `Line ${line}`);
}

function niceTicks(max: number): number[] {
  const step = max > 40 ? 10 : max > 20 ? 5 : max > 8 ? 2 : 1;
  const out: number[] = [];
  for (let t = 0; t <= max; t += step) out.push(t);
  return out;
}

function renderSeg(id: string, current: Stat, onPick: (s: Stat) => void) {
  const el = $(id);
  el.innerHTML = STATS.map((s) => `<button type="button" data-s="${s}" aria-pressed="${s === current}">${SHORT[s]}</button>`).join('');
  el.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>('button[data-s]');
    if (!b) return;
    el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    onPick(b.dataset.s as Stat);
  });
}

// ---------- accuracy section ----------
function renderMetrics() {
  $('metrics').innerHTML = `
    <table>
      <thead><tr><th>Stat</th><th>Average miss<br><span>model</span></th><th>Average miss<br><span>season average</span></th><th>Better by</th><th>Over/under picks right<br><span>model</span></th><th>Over/under picks right<br><span>always under</span></th></tr></thead>
      <tbody>${STATS.map((s) => {
        const m = model.metrics[s];
        return `<tr><th scope="row">${m.label}</th><td><strong>${m.mae_model.toFixed(2)}</strong></td><td>${m.mae_baseline.toFixed(2)}</td>
          <td class="good">${(m.improvement * 100).toFixed(1)}%</td><td><strong>${pct(m.pick_accuracy)}</strong></td><td>${pct(m.always_under)}</td></tr>`;
      }).join('')}</tbody>
    </table>
    <p class="muted small">"Average miss" is mean absolute error per game. Over/under picks are graded against a line at each player's season average; stat lines skew low, so always taking the under is the strategy to beat.</p>`;
}

function drawCalibration() {
  const svg = $('calibration') as unknown as SVGSVGElement;
  const S = 300, P = 36;
  svg.setAttribute('viewBox', `0 0 ${S} ${S}`);
  svg.setAttribute('aria-label', `Calibration of ${LABEL[calStat]} over/under odds`);
  svg.replaceChildren();
  const xy = (v: number) => P + v * (S - 2 * P);
  const inv = (v: number) => S - xy(v);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl('line', { x1: xy(t), x2: xy(t), y1: inv(0), y2: inv(1), class: 'grid' }, svg);
    svgEl('line', { x1: xy(0), x2: xy(1), y1: inv(t), y2: inv(t), class: 'grid' }, svg);
    svgEl('text', { x: xy(t), y: S - 12, class: 'tick', 'text-anchor': 'middle' }, svg, pct(t));
    svgEl('text', { x: P - 6, y: inv(t) + 4, class: 'tick', 'text-anchor': 'end' }, svg, pct(t));
  }
  svgEl('line', { x1: xy(0), y1: inv(0), x2: xy(1), y2: inv(1), class: 'diag' }, svg);
  const pts = model.metrics[calStat].calibration;
  const maxN = Math.max(...pts.map((p) => p.n));
  svgEl('polyline', { points: pts.map((p) => `${xy(p.predicted)},${inv(p.actual)}`).join(' '), class: 'cal-line' }, svg);
  pts.forEach((p, i) => {
    const c = svgEl('circle', { cx: xy(p.predicted), cy: inv(p.actual), r: 3 + 6 * Math.sqrt(p.n / maxN), class: 'cal-dot', style: `animation-delay:${i * 40}ms` }, svg);
    svgEl('title', {}, c, `Model said ${pct(p.predicted)}, over hit ${pct(p.actual)} (${p.n} games)`);
  });
  svgEl('text', { x: S / 2, y: S - 0, class: 'axis', 'text-anchor': 'middle' }, svg, 'Model’s chance of the over');
}

function drawImportance() {
  const names: Record<string, string> = {
    pts_ewm: 'Recent scoring (weighted)', pts_last_season: 'Last season’s scoring', fga_l10: 'Shot attempts, last 10',
    pts_season: 'Season scoring average', minutes_l5: 'Minutes, last 5', pts_l10: 'Scoring, last 10',
    minutes_l10: 'Minutes, last 10', rest_days: 'Days of rest', pts_l5: 'Scoring, last 5', fta_l10: 'Free throws, last 10',
    opp_pts_factor: 'Opponent defense', minutes_season: 'Season minutes', home: 'Home or away', age: 'Age', game_num: 'Games into season',
  };
  const max = model.importance[0].mae_increase;
  $('importance').innerHTML = model.importance.map((f, i) => `
    <div class="imp"><span>${names[f.feature] ?? f.feature}</span>
      <span class="imp__bar"><span style="width:${(f.mae_increase / max) * 100}%;animation-delay:${i * 60}ms"></span></span>
      <span class="muted small">+${f.mae_increase.toFixed(2)}</span></div>`).join('');
}
