#!/usr/bin/env node
process.title = "pi";
process.emitWarning = (() => {}) as typeof process.emitWarning;

await import("../cli.js");
