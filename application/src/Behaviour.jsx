import React, { useEffect, useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, BarController, LineController, Tooltip, Legend } from 'chart.js';
import { Chart } from 'react-chartjs-2';
import { ArrowClockwise, ChatsCircle, Handshake, Lightbulb, Scales, Storefront, Timer, UsersThree } from '@phosphor-icons/react';
import { api, number, percent } from './api';
import { Button, Empty, InfoBox, Panel } from './components';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, BarController, LineController, Tooltip, Legend);

const share = v => v == null ? '—' : percent(v * 100, 1);
const axis = { grid: { color: '#eef3fa' }, ticks: { font: { size: 10 } } };
const legend = { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } };
const BLUE = '#0756ff', GREEN = '#159a59', ORANGE = '#b3541e', GREY = '#d8e1ef';
// Chart.js draws an array label on several lines; long category names would overlap.
const wrap = (text, width = 14) => text.split(' ').reduce((lines, word) => {
  const last = lines[lines.length - 1];
  if (last && (last + ' ' + word).length <= width) lines[lines.length - 1] = `${last} ${word}`; else lines.push(word);
  return lines;
}, []);
const RESPONSE_COLOURS = { accepted: GREEN, negotiated: BLUE, deferred: '#a16207', rejected: '#b91c1c' };

// ------------------------------------------------------------ market position

// Where each lender prices against us, and what their presence does to our win
// rate. Read from the league table the page already has.
export function MarketPosition({ league, base }) {
  const rows = league.filter(r => r.gap_to_our_rate_bps?.median != null).sort((a, b) => a.gap_to_our_rate_bps.median - b.gap_to_our_rate_bps.median);
  const gaps = rows.map(r => r.gap_to_our_rate_bps.median);
  const lifts = rows.map(r => r.win_rate?.lift_pp ?? null);
  const gapData = { labels: rows.map(r => r.competitor), datasets: [{ label: 'Median quote minus our rate (bps)', data: gaps, backgroundColor: gaps.map(g => g < 0 ? ORANGE : GREEN), borderRadius: 3 }] };
  const liftData = { labels: rows.map(r => r.competitor), datasets: [{ label: 'Win rate vs book (pp)', data: lifts, backgroundColor: lifts.map(l => l == null ? GREY : l < 0 ? ORANGE : GREEN), borderRadius: 3 }] };
  const horizontal = title => ({ indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
    scales: { x: { ...axis, title: { display: true, text: title, font: { size: 10 } } }, y: { ...axis, ticks: { font: { size: 10 } } } } });
  return <div className="ci-two">
    <Panel title="Where each lender prices against us" icon={Scales}>
      <p className="small-copy">Median quote in the notes minus the rate we sanctioned. Orange undercuts us; green is dearer than us.</p>
      <div className="ci-chart tall"><Chart type="bar" data={gapData} options={horizontal('bps')}/></div>
    </Panel>
    <Panel title="What their presence does to our win rate" icon={Storefront}>
      <p className="small-copy">Our win rate on files that name the lender, against the book's {share(base)}. Grey: too few files to say.</p>
      <div className="ci-chart tall"><Chart type="bar" data={liftData} options={horizontal('percentage points')}/></div>
    </Panel>
  </div>;
}

// ------------------------------------------------------- customer behaviour

// Win rate as bars, a second measure as a line on its own axis. Cells the API
// suppressed for having too few files are drawn grey and empty, never as zero.
function BarsAndLine({ cells, line, lineLabel, lineUnit, lineScale = v => v, height }) {
  const data = { labels: cells.map(c => wrap(c.label)), datasets: [
    { type: 'bar', label: 'Win rate %', data: cells.map(c => c.suppressed ? null : c.win_rate * 100), backgroundColor: cells.map(c => c.suppressed ? GREY : BLUE), borderRadius: 3, yAxisID: 'y', order: 2 },
    { type: 'line', label: lineLabel, data: cells.map(c => c.suppressed || c[line] == null ? null : lineScale(c[line])), borderColor: ORANGE, backgroundColor: ORANGE, pointRadius: 3, tension: .25, yAxisID: 'y1', order: 1 }] };
  const options = { responsive: true, maintainAspectRatio: false, plugins: { legend,
    tooltip: { callbacks: { afterBody: items => { const c = cells[items[0].dataIndex]; return c.suppressed ? `n = ${c.n} (too few to show)` : `n = ${number(c.n)}`; } } } },
    scales: { x: { ...axis, ticks: { font: { size: 10 }, maxRotation: 0, autoSkip: false } }, y: { ...axis, min: 0, max: 100, title: { display: true, text: 'Win rate %', font: { size: 10 } } },
      y1: { ...axis, position: 'right', beginAtZero: true, grid: { drawOnChartArea: false }, title: { display: true, text: lineUnit, font: { size: 10 } } } } };
  return <div className={`ci-chart ${height || ''}`}><Chart type="bar" data={data} options={options}/></div>;
}

function ResponseMix({ cells }) {
  const shown = cells.filter(c => !c.suppressed);
  const data = { labels: shown.map(c => c.label), datasets: Object.entries(RESPONSE_COLOURS).map(([k, colour]) => (
    { label: k[0].toUpperCase() + k.slice(1), data: shown.map(c => (c.responses[k] || 0) * 100), backgroundColor: colour, borderRadius: 2 })) };
  const options = { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend },
    scales: { x: { ...axis, stacked: true, max: 100, title: { display: true, text: '% of files', font: { size: 10 } } }, y: { ...axis, stacked: true } } };
  return <div className="ci-chart small"><Chart type="bar" data={data} options={options}/></div>;
}

export function CustomerBehaviour() {
  const [body, setBody] = useState(null);
  const [error, setError] = useState('');
  const load = () => { setError(''); api('/competitors/behaviour').then(setBody).catch(e => setError(e.message)); };
  useEffect(load, []);

  if (error) return <Panel title="Customer behaviour" icon={UsersThree}><InfoBox warning><span role="alert">{error}</span> <Button onClick={load}><ArrowClockwise/> Retry</Button></InfoBox></Panel>;
  if (!body) return <Panel title="Customer behaviour" icon={UsersThree}><Empty icon={UsersThree} title="Reading the loan book">One moment…</Empty></Panel>;
  const h = body.headline;
  return <section className="ci-behaviour">
    <div className="ci-section-head">
      <h2><UsersThree size={22}/> Customer behaviour</h2>
      <p>How customers act around a competitor offer, across {number(h.files)} files: whether evidence, negotiation and sales pressure change the outcome, and whether the files we win stay.</p>
    </div>
    <div className="model-metrics ci-metrics">
      <div><small>Win rate</small><b>{share(h.win_rate)}</b></div>
      <div><small>Customers who negotiate</small><b>{share(h.negotiating_share)}</b></div>
      <div><small>Average discount: negotiating / not</small><b>{h.avg_discount_when_negotiating_bps} / {h.avg_discount_otherwise_bps} bps</b></div>
      <div><small>Competitor quote documented</small><b>{share(h.documented_quote_share)}</b></div>
      <div><small>Won files moved away in 24 months</small><b>{share(h.bt_out_24m)}</b></div>
    </div>
    {body.findings?.length > 0 && <InfoBox><b><Lightbulb size={16}/> What the book shows</b><ul className="ci-list">{body.findings.map(f => <li key={f}>{f}</li>)}</ul></InfoBox>}

    <div className="ci-two">
      <Panel title="Price gap: who accepts, and who stays" icon={Scales}>
        <p className="small-copy">Our offered rate against the competitor's quote. Bars: share of files won. Line: share of won files that moved to another lender within 24 months.</p>
        <BarsAndLine cells={body.by_gap} line="bt_out_24m" lineLabel="Moved away in 24 months %" lineUnit="Moved away %" lineScale={v => v * 100}/>
      </Panel>
      <Panel title="Does negotiating win the file?" icon={Handshake}>
        <p className="small-copy">By rounds of negotiation before the offer. Bars: win rate. Line: average discount below the system recommendation.</p>
        <BarsAndLine cells={body.by_negotiation} line="avg_discount_bps" lineLabel="Average discount (bps)" lineUnit="Discount bps"/>
      </Panel>
    </div>
    <div className="ci-two">
      <Panel title="Sales pressure: discounts the customer did not ask for" icon={Timer}>
        <p className="small-copy">Month end and progress against the sales target. If discounts rise here but the win rate does not, the discount is buying the target, not the customer.</p>
        <BarsAndLine cells={[...body.by_month_end, ...body.by_target]} line="avg_discount_bps" lineLabel="Average discount (bps)" lineUnit="Discount bps"/>
      </Panel>
      <Panel title="How customers respond" icon={ChatsCircle}>
        <p className="small-copy">The customer's response to our offer, by how the competitor quote was backed and whether they already bank with us.</p>
        <ResponseMix cells={[...body.by_evidence, ...body.by_relationship]}/>
      </Panel>
    </div>
    <Panel title="By sourcing channel" icon={Storefront}>
      <p className="small-copy">Bars: win rate. Line: share of files discounted by more than 5 bps.</p>
      <BarsAndLine cells={body.by_channel} line="discounted_share" lineLabel="Discounted share %" lineUnit="Discounted %" lineScale={v => v * 100} height="small"/>
    </Panel>
    <p className="small-copy">{Object.entries(body.definitions).map(([k, v]) => <span key={k}><b>{k.replace(/_/g, ' ')}</b>: {v}. </span>)} Cells with fewer than {body.settings.min_n_rate} files are suppressed.</p>
  </section>;
}
