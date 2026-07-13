// KPIs + composite 0-100 score. The tracking component uses the same normalized
// controlled-output error as the Python backend: controlled outputs first, each
// divided by its declared output range, plus wasted energy and interlock time.
import { splitYSp } from './controllers.js?v=15';

const W_TRACKING = 20.0, W_ENERGY = 0.5, W_SAFETY = 60;
const r = (v, d = 2) => +v.toFixed(d);

const OUTPUT_BOUNDS = {
  cascade: { temps: [[25, 80], [30, 82], [35, 85]] },
  quadruple: { temps: [[25, 72], [25, 72], [20, 58], [20, 58]] },
  cstr: { temps: [[45, 90]] },
  hvac: { temps: [[18, 26], [18, 26]] },
};

// Economic objective (mirrors aiogym/env.py ECON): value − energy-cost − soft-band
// violation. Controlled vars have soft acceptance bands (not a fixed SP); within the
// band, economics are optimised. This is the metric the RL policies actually optimise
// — so a fixed-SP PID/MPC scores low here while the (工况-adaptive) RL scores high.
const ECON = {
  cascade:   { temp_band: [[34, 44], [48, 58], [60, 72]], level_band: [[0.32, 0.58], [0.32, 0.58], [0.32, 0.58]], value: 'none', w_value: 0.0, w_energy: 0.6, w_viol: 25.0 },
  quadruple: { temp_band: [[46, 58], [46, 58], [32, 46], [32, 46]], level_band: [[0.32, 0.56], [0.32, 0.56]], value: 'none', w_value: 0.0, w_energy: 0.6, w_viol: 25.0 },
  cstr:      { temp_band: [[null, 88]], level_band: [], value: 'production', w_value: 900.0, w_energy: 0.4, w_viol: 8.0 },
  hvac:      { temp_band: [[20, 24], [20, 24]], level_band: [], value: 'none', w_value: 0.0, w_energy: 1.2, w_viol: 14.0 },
};
// Per-scenario [worst, best] profit-rate (per control step) for a 0-100 economic score.
// worst ≈ fixed-SP PID, best ≈ the economic optimum the RL targets (from aiogym/runs).
// per-step profit-rate [worst, best] from aiogym/runs (supervisory RL vs PID/MPC).
// Honest: RL clearly leads on cstr/hvac (real economic headroom); on cascade/quad it's
// competitive (regulation problems where PID/MPC are near-optimal — no gaming).
const ECON_REF = { cascade: [-260, -120], quadruple: [-185, -115], cstr: [-3.0, 1.2], hvac: [-9.0, -4.0] };

export class ScoreKeeper {
  constructor(model) { this.bind(model); }
  bind(model) {
    this.model = model; this.n = model.n; this.levelSlots = model.legacyLevelTargetSlots();
    this.scoreEnergy = model.energyScored ? model.energyScored() : true;
    this.scenario = model.metadata ? model.metadata().scenario : 'cascade';
    this.econ = ECON[this.scenario] || ECON.cascade;
    this.econRef = ECON_REF[this.scenario] || ECON_REF.cascade;
    this.reset();
  }
  reset() {
    this.elapsed = 0; this.outputErrorIae = 0; this.meanOutputErrorIae = 0; this.energy = 0; this.excess = 0;
    this.alarmSec = 0; this.interlockSec = 0; this.trips = 0; this.prevIl = false;
    this.instOutputErr = new Array(this.levelSlots.length + this.n).fill(0);
    this.econProfit = 0; this.econSteps = 0; this.prod = 0; this.econEMA = null;
  }

  update(s, sp, mask, nAlarms, dt, act) {
    this.elapsed += dt;
    const y = [...this.levelSlots.map((i) => s.levels[i]), ...s.temps];
    const targets = splitYSp(this.model, sp.y_sp);
    const ysp = [...this.levelSlots.map((i) => targets.level_targets[i]), ...targets.output_targets];
    const scales = this._outputScales();
    const err = y.map((v, i) => Math.abs(v - ysp[i]) / Math.max(scales[i] || 1, 1e-12));
    const outputErr = err.reduce((a, b) => a + b, 0);
    const meanErr = outputErr / Math.max(1, err.length);
    this.instOutputErr = err;
    this.outputErrorIae += outputErr * dt;
    this.meanOutputErrorIae += meanErr * dt;
    const heat = s.heater_power.reduce((a, b) => a + b, 0);
    this.energy += (heat + s.pump_power.reduce((a, b) => a + b, 0)) * dt / 3.6e6;
    if (this.scoreEnergy) this.excess += Math.max(0, heat - this.model.idealPower(s, targets.output_targets)) * dt / 3.6e6;
    if (nAlarms > 0) this.alarmSec += dt;
    const il = mask.pump_trip || mask.heater_trip.some(Boolean);
    if (il) { this.interlockSec += dt; if (!this.prevIl) this.trips++; }
    this.prevIl = il;
    this._economic(s, heat, act);
  }

  // Per-step economic profit (mirrors aiogym _economic_profit): value − energy − band-violation.
  _economic(s, heat, act) {
    const c = this.econ, p = this.model.p || {};
    let value = 0, prod = 0;
    if (c.value === 'production' && act) {
      const Ca = (s.conc && s.conc[0] != null) ? s.conc[0] : 0;
      const D = (act.pumps[0] || 0) * (p.Dmax || 0);
      prod = D * ((p.Caf || 0) - Ca);
      value = prod;
    }
    this.prod = prod;
    const energyKw = heat / 1000;
    let viol = 0;
    for (let i = 0; i < c.temp_band.length; i++) {
      const [lo, hi] = c.temp_band[i], T = s.temps[i];
      if (lo != null && T < lo) viol += (lo - T) / 10;
      if (hi != null && T > hi) viol += (T - hi) / 10;
    }
    this.levelSlots.forEach((idx, j) => {
      const b = c.level_band[j]; if (!b) return;
      const [lo, hi] = b, L = s.levels[idx];
      if (lo != null && L < lo) viol += (lo - L) / 0.1;
      if (hi != null && L > hi) viol += (L - hi) / 0.1;
    });
    const profit = c.w_value * value - c.w_energy * energyKw - c.w_viol * viol;
    this.econProfit += profit; this.econSteps += 1;
    // EMA (~100-step window) for the live score: reflects steady-state, not the
    // start-up transient (e.g. HVAC heating from 10°C into the comfort band).
    this.econEMA = this.econEMA == null ? profit : 0.99 * this.econEMA + 0.01 * profit;
  }

  _outputScales() {
    const meta = this.model.metadata ? this.model.metadata() : {};
    const hmax = meta.height_max || this.model.heightMax || [];
    const tempBounds = (OUTPUT_BOUNDS[this.scenario] || {}).temps || [];
    return [
      ...this.levelSlots.map((idx) => Math.max(hmax[idx] || 1, 1e-12)),
      ...Array.from({ length: this.n }, (_, i) => {
        const b = tempBounds[i];
        return b ? Math.max(b[1] - b[0], 1e-12) : 1;
      }),
    ];
  }

  report() {
    const e = Math.max(this.elapsed, 1e-6), hours = e / 3600;
    const avgY = this.meanOutputErrorIae / e;
    const avgP = hours > 0 ? this.energy / hours : 0, avgX = hours > 0 ? this.excess / hours : 0;
    const sFrac = this.interlockSec / e;
    const pY = W_TRACKING * avgY, pE = this.scoreEnergy ? W_ENERGY * avgX : 0, pS = W_SAFETY * sFrac;
    const score = Math.max(0, Math.min(100, 100 - pY - pE - pS));
    // economic score: per-episode AVERAGE profit-rate (the scorer is reset each episode
    // by the engine) mapped to 0-100 against [worst-PID, best-optimum]
    const rate = this.econSteps > 0 ? this.econProfit / this.econSteps : 0;
    const [lo, hi] = this.econRef;
    const econScore = Math.max(0, Math.min(100, 100 * (rate - lo) / (hi - lo)));
    return {
      elapsed: r(this.elapsed, 1), score: r(score, 1), econ_score: r(econScore, 1),
      components: { tracking: r(pY, 1), energy: r(pE, 1), safety: r(pS, 1) },
      kpis: {
        avg_output_error: r(avgY), output_tracking_iae: r(this.outputErrorIae, 3), energy_kwh: r(this.energy, 3),
        avg_power_kw: r(avgP), excess_kwh: r(this.excess, 3), interlock_seconds: r(this.interlockSec, 1), trip_events: this.trips,
      },
      econ: { score: r(econScore, 1), profit_rate: r(rate, 2), production: r(this.prod * 1000, 2), value: this.econ.value },
      inst_output_err: this.instOutputErr.map((x) => r(x, 4)),
    };
  }
}
