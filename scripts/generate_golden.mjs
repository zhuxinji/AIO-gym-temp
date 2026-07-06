#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const DEFAULT_OUT = path.join(ROOT, 'aiogym', 'tests', 'golden.json');
const SCENARIOS = ['cascade', 'quadruple', 'cstr', 'hvac'];
const DEFAULT_ACTIONS = {
  cascade: { pumps: [0.6], valves: [0.5, 0.4, 0.3], heaters: [0.4, 0.3, 0.5] },
  quadruple: { pumps: [0.5, 0.6], valves: [], heaters: [0.4, 0.3, 0.5, 0.2] },
  cstr: { pumps: [0.5], valves: [], heaters: [0.4] },
  hvac: { pumps: [], valves: [], heaters: [0.7, 0.3] },
};
const DEFAULT_CHECKPOINTS = [1, 2, 5, 10, 25, 50, 100, 200];

function parseArgs(argv) {
  const args = {
    out: DEFAULT_OUT,
    dt: 0.05,
    steps: 200,
    checkpoints: DEFAULT_CHECKPOINTS,
    check: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--check') args.check = true;
    else if (arg === '--out') args.out = path.resolve(argv[++i]);
    else if (arg === '--dt') args.dt = Number(argv[++i]);
    else if (arg === '--steps') args.steps = Number.parseInt(argv[++i], 10);
    else if (arg === '--checkpoints') args.checkpoints = argv[++i].split(',').map((v) => Number.parseInt(v, 10));
    else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isFinite(args.dt) || args.dt <= 0) throw new Error('--dt must be a positive number');
  if (!Number.isInteger(args.steps) || args.steps <= 0) throw new Error('--steps must be a positive integer');
  if (args.checkpoints.some((v) => !Number.isInteger(v) || v <= 0 || v > args.steps)) {
    throw new Error('--checkpoints must be positive integers no larger than --steps');
  }
  args.checkpoints = [...new Set(args.checkpoints)].sort((a, b) => a - b);
  return args;
}

function printHelp() {
  console.log(`Generate aiogym/tests/golden.json from the browser JS source models.

Usage:
  node scripts/generate_golden.mjs [--out PATH] [--dt 0.05] [--steps 200]
  node scripts/generate_golden.mjs --check

Options:
  --check              Compare generated output with the existing golden file without writing.
  --out PATH           Output JSON path. Default: aiogym/tests/golden.json
  --dt NUMBER          Integration step used by parity tests. Default: 0.05
  --steps INTEGER      Number of integration steps. Default: 200
  --checkpoints LIST   Comma-separated checkpoints. Default: 1,2,5,10,25,50,100,200`);
}

async function writeModuleCopy(srcRel, dstPath, replacements = []) {
  const srcPath = path.join(ROOT, srcRel);
  let text = await fs.readFile(srcPath, 'utf8');
  for (const [from, to] of replacements) text = text.split(from).join(to);
  await fs.mkdir(path.dirname(dstPath), { recursive: true });
  await fs.writeFile(dstPath, text, 'utf8');
}

async function loadBrowserModules() {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'aiogym-golden-'));
  const tmpJs = path.join(tmp, 'frontend', 'js');
  const tmpSim = path.join(tmpJs, 'sim');

  await writeModuleCopy('frontend/js/i18n.js', path.join(tmpJs, 'i18n.mjs'));
  await writeModuleCopy('frontend/js/sim/kernel.js', path.join(tmpSim, 'kernel.mjs'));
  await writeModuleCopy('frontend/js/sim/models.js', path.join(tmpSim, 'models.mjs'), [
    ["../i18n.js?v=15", "../i18n.mjs"],
  ]);

  const [{ makeModel }, { Integrator }] = await Promise.all([
    import(pathToFileURL(path.join(tmpSim, 'models.mjs')).href),
    import(pathToFileURL(path.join(tmpSim, 'kernel.mjs')).href),
  ]);
  return { makeModel, Integrator };
}

function envFor(model) {
  return {
    t_cold: model.p.t_cold,
    t_amb: model.p.t_amb,
    extra_outflow: 0.0,
  };
}

function generateScenario(makeModel, Integrator, scenario, args) {
  const model = makeModel(scenario);
  const integ = new Integrator(model);
  const action = structuredClone(DEFAULT_ACTIONS[scenario]);
  const env = envFor(model);
  const checkpointSet = new Set(args.checkpoints);
  const checkpoints = {};

  for (let i = 1; i <= args.steps; i++) {
    integ.step(args.dt, action, env);
    if (checkpointSet.has(i)) checkpoints[String(i)] = integ.x.slice();
  }
  return { action, dt: args.dt, checkpoints };
}

function buildGolden(makeModel, Integrator, args) {
  const golden = {
    _comment: 'Golden state-trajectory checkpoints generated from the browser JS engine (frontend/js/sim/models.js + kernel.js, the source of truth). Each scenario is integrated with a fixed action for 200 steps of dt; full raw state x recorded at the listed step indices. test_parity.py asserts the native numpy port reproduces these.',
  };
  for (const scenario of SCENARIOS) {
    golden[scenario] = generateScenario(makeModel, Integrator, scenario, args);
  }
  return golden;
}
function compareNumbers(pathLabel, actual, expected, mismatches) {
  const atol = 1e-12;
  const rtol = 1e-12;
  const err = Math.abs(actual - expected);
  const tol = atol + rtol * Math.abs(expected);
  if (err > tol) mismatches.push(`${pathLabel}: ${actual} vs ${expected} (abs_delta=${err})`);
}

function compareValue(pathLabel, actual, expected, mismatches) {
  if (typeof actual === 'number' && typeof expected === 'number') {
    compareNumbers(pathLabel, actual, expected, mismatches);
    return;
  }
  if (Array.isArray(actual) || Array.isArray(expected)) {
    if (!Array.isArray(actual) || !Array.isArray(expected) || actual.length !== expected.length) {
      mismatches.push(`${pathLabel}: array shape mismatch`);
      return;
    }
    actual.forEach((value, i) => compareValue(`${pathLabel}[${i}]`, value, expected[i], mismatches));
    return;
  }
  if (actual && expected && typeof actual === 'object' && typeof expected === 'object') {
    const keys = new Set([...Object.keys(actual), ...Object.keys(expected)].filter((k) => k !== '_comment'));
    for (const key of [...keys].sort()) compareValue(`${pathLabel}.${key}`, actual[key], expected[key], mismatches);
    return;
  }
  if (actual !== expected) mismatches.push(`${pathLabel}: ${actual} vs ${expected}`);
}

function compareGolden(actual, expected) {
  const mismatches = [];
  compareValue('golden', actual, expected, mismatches);
  return mismatches;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const { makeModel, Integrator } = await loadBrowserModules();
  const golden = buildGolden(makeModel, Integrator, args);
  const json = `${JSON.stringify(golden, null, 2)}\n`;

  if (args.check) {
    const current = JSON.parse(await fs.readFile(args.out, 'utf8'));
    const mismatches = compareGolden(golden, current);
    if (mismatches.length) {
      console.error(`Golden mismatch: ${path.relative(ROOT, args.out)}`);
      for (const mismatch of mismatches.slice(0, 10)) console.error(`  ${mismatch}`);
      if (mismatches.length > 10) console.error(`  ... ${mismatches.length - 10} more mismatches`);
      console.error('Run: node scripts/generate_golden.mjs');
      process.exit(1);
    }
    console.log(`Golden is up to date: ${path.relative(ROOT, args.out)}`);
    return;
  }

  await fs.mkdir(path.dirname(args.out), { recursive: true });
  await fs.writeFile(args.out, json, 'utf8');
  console.log(`Wrote ${path.relative(ROOT, args.out)}`);
}

main().catch((err) => {
  console.error(err?.stack || err);
  process.exit(1);
});